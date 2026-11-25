"""Track A: own-transfer rules -> TF-IDF+LogReg (when confident) -> Haiku (unseen merchants, deduped, parallel) -> unknown.
Run: python classify.py validation.csv   (expects transactions.csv next to it; ANTHROPIC_API_KEY in env)
Without an API key the Haiku step is skipped (those rows become unknown)."""
import io, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

TIMES, _last = {}, [time.perf_counter()]


def lap(name):
    """Record wall time since the previous lap under `name`."""
    now = time.perf_counter()
    TIMES[name] = TIMES.get(name, 0) + now - _last[0]
    _last[0] = now

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


def load_env(path=".env"):
    """Read the key from .env (never committed). Accepts ANTHROPIC_API_KEY=... or api_key=..."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8-sig"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k.upper() in ("ANTHROPIC_API_KEY", "API_KEY") and v:
                os.environ.setdefault("ANTHROPIC_API_KEY", v)


HERE = os.path.dirname(os.path.abspath(__file__))  # training data and .env live next to this script
load_env(os.path.join(HERE, ".env"))


def load(path):
    """Handles normal CSVs and CSVs where every line was wrapped in quotes when pasted."""
    df = pd.read_csv(io.StringIO(open(path, encoding="utf-8-sig", errors="replace").read()))
    if df.shape[1] == 1:
        col = df.columns[0]
        df = pd.read_csv(io.StringIO("\n".join([col] + df[col].astype(str).tolist())))
    return df


# å/ä/ö may have been lost as '?' when copying; drop both so "ÖVERFÖRING" and "?VERF?RING" normalise the same
NOISE = r"\b(KORTKP|KP|RESERVERAT|PRELIMINRT|BETALNING|VERFRING)\b"


def norm(t):
    t = re.sub(r"[ÅÄÖ?]", "", str(t).upper())
    t = re.sub(r"\d+", " ", t)
    t = re.sub(NOISE, " ", t)
    return re.sub(r"[^\w*]+", " ", t).strip()


def sign(df):
    return np.where(df.amount_sek < 0, "OUT", "IN")


def feat(df):
    return df.text.map(norm) + " SIGN_" + sign(df) + " ACC_" + df.account_type.astype(str).str.upper()


def own_transfers(df):
    """Same customer, same day, same absolute amount, opposite signs, different accounts."""
    out = {}
    d = df.assign(a=df.amount_sek.abs().round(2))
    for _, g in d.groupby(["customer", "date", "a"]):
        if len(g) < 2 or g.account.nunique() < 2 or not ((g.amount_sek > 0).any() and (g.amount_sek < 0).any()):
            continue
        acc = (g.account_type.astype(str) + " " + g.account_name.astype(str)).str.lower()
        card = acc.str.contains("credit|card|kredit|kort")
        if card.any():  # card bill: checking side = credit.loan, card side = transfer.own
            for i in g.index:
                out[i] = "transfer.own" if card[i] else "credit.loan"
        else:
            sav = acc.str.contains("sav|spar|buffert").any()
            for i in g.index:
                out[i] = "transfer.savings" if sav else "transfer.own"
    return out


MORTGAGE = r"\bSBAB\b|HYPOTEK|BOL[ÅA]N|STADSHYPOTEK|LANDSHYPOTEK"


def fixed_rules(df):
    """High-precision rules from the labelling guide (applied after own-transfer pairing)."""
    out = {}
    up = df.text.astype(str).str.upper()
    card = df.account_type.astype(str).str.contains("credit")
    # Card bill arriving on the card account (dates often differ from the checking side, so pairing misses it)
    for i in df.index[card & (df.amount_sek > 0) & up.str.contains(r"INBETALNING|PAYMENT|BETALNING")]:
        out[i] = "transfer.own"
    for i in df.index[(df.amount_sek < 0) & up.str.contains(MORTGAGE)]:
        out[i] = "credit.loan"
    return out


train, test = load(os.environ.get("TRAIN_CSV", os.path.join(HERE, "transactions.csv"))), load(sys.argv[1])
ALL_CATS = {"credit.collections", "credit.loan", "income.capital", "income.employment", "income.public",
            "income.sales", "spend.dining", "spend.family", "spend.gambling", "spend.groceries", "spend.health",
            "spend.housing", "spend.leisure", "spend.shopping", "spend.transport", "transfer.cash",
            "transfer.own", "transfer.person", "transfer.savings"}
CATS = sorted(ALL_CATS | set(train.category.unique()))  # full list even if training lacks some categories

# --- traditional ML + confidence threshold tuned on leave-one-customer-out (mimics "new customers")
model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                      LogisticRegression(C=10, max_iter=3000))
Xtr, ytr = feat(train), train.category.values
proba = cross_val_predict(model, Xtr, ytr, groups=train.customer, cv=GroupKFold(train.customer.nunique()),
                          method="predict_proba")
cv_pred, cv_conf = np.array(CATS)[proba.argmax(1)], proba.max(1)


def score(t):
    return ((cv_pred == ytr) & (cv_conf >= t)).sum() - ((cv_pred != ytr) & (cv_conf >= t)).sum()


TH = max(np.arange(0.3, 0.97, 0.05), key=score)
print(f"ML threshold {TH:.2f}; leave-one-customer-out score/100 (rest unknown): "
      f"{100 * score(TH) / len(ytr):.0f}", file=sys.stderr)
# With Haiku available, uncertain rows go to Haiku (not unknown), so ML should only keep very confident rows
if os.environ.get("ANTHROPIC_API_KEY"):
    TH = max(TH, float(os.environ.get("MIN_TH", "0.8")))
    print(f"Haiku available: ML threshold raised to {TH:.2f}", file=sys.stderr)
model.fit(Xtr, ytr)
lap("setup: load data, tune threshold (CV), train ML")

# --- Haiku for what ML isn't sure about
examples = "\n".join(f"{r.text} | {'OUT' if r.amount_sek < 0 else 'IN'} | {r.account_type} | {r.account_name} -> {r.category}"
                     for r in train.groupby("category").head(3).itertuples())
CAT_HELP = """income.employment: salary and wages from an employer
income.public: benefits/allowances from public bodies (e.g. child benefit from Försäkringskassan)
income.sales: selling your own things on a marketplace (Tradera, Blocket)
income.capital: interest and dividends
spend.housing: rent, electricity, internet and phone, moving costs
spend.groceries: supermarkets and food shops
spend.dining: restaurants, cafés, fast food, food delivery
spend.transport: public transport, taxis/ride-hailing, scooters, fuel, parking, bridge tolls
spend.shopping: clothes, beauty, home/furniture, electronics, DIY, online stores
spend.leisure: streaming, games, gym, cinema, events, flights, hotels, theme parks
spend.health: pharmacies, doctors, dentists
spend.family: childcare and school fees
spend.gambling: betting, casinos, lotteries (deposits and winnings alike)
credit.loan: repaying a loan, a credit card bill or a pay-later invoice
credit.collections: paying a debt collector or Kronofogden
transfer.own: between the customer's own everyday accounts (credit card account included)
transfer.savings: into or out of the customer's own savings
transfer.person: to or from a private person, no purpose stated
transfer.cash: cash withdrawals"""

SYSTEM = f"""Label Swedish bank transactions with exactly one category.
Categories:
{CAT_HELP}
Note: Swedish letters å/ä/ö may appear as '?'.
Rules: label by the type of business, not the occasion. Refunds/payouts take the category of what they reverse.
Paying a Klarna/Qliro/Walley invoice, a credit card bill, or a mortgage (bolån, hypotek) = credit.loan.
Card payments via KLARNA*, ZETTLE_*, SUMUP *, IZ *, PAYPAL * = category of the shop behind them.
Money to/from a person: use the message (e.g. hyra = spend.housing); no message = transfer.person.
Money into or out of a savings-purpose account = transfer.savings.
Set sure=false if you do not recognise the merchant. A wrong label costs more than an unsure one.
Answer every line index i.
Examples (text | direction | account type | account name -> category):
{examples}"""


class Item(BaseModel):
    i: int
    c: str
    sure: bool


class Out(BaseModel):
    items: list[Item]


USAGE = []  # (input_tokens, output_tokens) per Haiku call


def ask(batch):
    try:
        # Haiku 5.5 thinks by default; a lookup-style classification doesn't need it (faster, fewer output tokens)
        r = client.messages.parse(model="claude-haiku-5-5", max_tokens=4000, system=SYSTEM, output_format=Out,
                                  thinking={"type": "disabled"},
                                  messages=[{"role": "user", "content": "\n".join(f"{i}: {l}" for i, l in batch)}])
        USAGE.append((r.usage.input_tokens, r.usage.output_tokens))
        return r.parsed_output.items
    except Exception as e:
        print("batch failed:", e, file=sys.stderr)
        return []


# --- assemble: rules > confident ML > Haiku > unknown
pred = pd.Series("unknown", index=test.index, dtype=object)
stage = pd.Series("unknown", index=test.index, dtype=object)
rules = {**fixed_rules(test), **own_transfers(test)}
pred.loc[list(rules)] = list(rules.values())
stage.loc[list(rules)] = "rules"
lap("1 rules")
rest = test.index.difference(list(rules))
p = model.predict_proba(feat(test.loc[rest]))
conf = p.max(1)
# Rule 5 (payment-provider prefixes) and rule 6 (Swish messages) need reading the text, not pattern-matching:
# never let ML decide these, always send them to Haiku
needs_llm = test.loc[rest].text.astype(str).str.upper().str.contains(
    r"KLARNA\*|ZETTLE_|SUMUP |IZ \*|PAYPAL \*|SWISH").values
# Keep ML's opinion on every row: used later as a second opinion on Haiku's answers
ml_lab = pd.Series(model.classes_[p.argmax(1)], index=rest)
ml_p = pd.Series(p.max(1), index=rest)
llm_only = pd.Series(needs_llm, index=rest)
conf = np.where(needs_llm, 0.0, conf)
pred.loc[rest[conf >= TH]] = model.classes_[p.argmax(1)][conf >= TH]
stage.loc[rest[conf >= TH]] = "ml"

todo = test.loc[rest[conf < TH]]
todo = todo.assign(key=feat(todo).values,
                   line=(todo.text.astype(str) + " | " + sign(todo) + " | " + todo.account_type.astype(str)
                         + " | " + todo.account_name.astype(str)).values)
first = todo.drop_duplicates("key")  # dedupe: pay per merchant, not per row
lap("2 ML")
if os.environ.get("ANTHROPIC_API_KEY") and len(first):
    import anthropic
    client = anthropic.Anthropic(max_retries=6)  # 16 parallel calls may hit rate limits on a fresh key
    items = list(enumerate(first.line))
    with ThreadPoolExecutor(16) as ex:
        res = [x for b in ex.map(ask, [items[k:k + 50] for k in range(0, len(items), 50)]) for x in b]
    lap("3 Haiku")
    # Scoring: right +1, wrong -1, unknown 0 (a wrong label can move a credit decision),
    # so unsure answers become unknown. GUESS=1 keeps them (only if wrong labels stop costing points).
    abstain = os.environ.get("GUESS") != "1"
    xcheck = os.environ.get("XCHECK", "1") == "1"
    VETO = float(os.environ.get("VETO", "0.9"))  # ML must be this sure to overrule a "sure" Haiku answer
    got = {first.key.iloc[x.i]: x for x in res if x.c in CATS and 0 <= x.i < len(first)}

    def decide(idx, key, cross):
        """Cross-check Haiku with ML: agree -> accept; strong disagreement -> unknown."""
        x = got.get(key)
        if x is None:
            return "unknown"
        if not abstain:
            return x.c
        if not cross or llm_only[idx]:  # Swish/payment-provider rows: ML can't read messages, trust Haiku only
            return x.c if x.sure else "unknown"
        agree = ml_lab[idx] == x.c
        if x.sure:
            return x.c if agree or ml_p[idx] < VETO else "unknown"
        return x.c if agree else "unknown"

    hit = pd.Series([decide(i, k, xcheck) for i, k in zip(todo.index, todo.key)], index=todo.index)
    base = pd.Series([decide(i, k, False) for i, k in zip(todo.index, todo.key)], index=todo.index)
    pred.loc[todo.index] = hit.values
    stage.loc[todo.index[(hit != "unknown").values]] = "haiku"
    lap("4 cross-check")
    if "category" in test.columns:  # same Haiku answers, with vs without the cross-check
        def sc(s):
            k = s != "unknown"
            r = (s[k] == test.category[s.index][k]).sum()
            return f"right {r}, wrong {k.sum() - r}, unknown {(~k).sum()}"
        print(f"  Haiku rows sure-only : {sc(base)}", file=sys.stderr)
        print(f"  Haiku rows cross-check: {sc(hit)}", file=sys.stderr)
else:
    print("No ANTHROPIC_API_KEY: skipping Haiku", file=sys.stderr)

# GUESS=1 only: fill remaining unknowns with the ML model's best guess
if os.environ.get("GUESS") == "1":
    left = pred.index[pred == "unknown"]
    if len(left):
        pred.loc[left] = model.predict(feat(test.loc[left]))
        stage.loc[left] = "ml_guess"

print(f"rules {len(rules)}, ML {int((conf >= TH).sum())}, to Haiku {len(first)} merchants/{len(todo)} rows, "
      f"unknown {(pred == 'unknown').sum()}", file=sys.stderr)

print("Timing:", file=sys.stderr)
for k, v in TIMES.items():
    print(f"  {k:<50} {v:6.2f} s", file=sys.stderr)

if USAGE:  # Haiku 5.5: $0.10 / MTok in, $0.50 / MTok out
    tin, tout = sum(u[0] for u in USAGE), sum(u[1] for u in USAGE)
    cost = tin * 0.10 / 1e6 + tout * 0.50 / 1e6
    print(f"Haiku: {len(USAGE)} calls, {tin} in / {tout} out tokens, ${cost:.4f} "
          f"(${cost / len(test) * 1e6:.2f} per 1M rows at this mix)", file=sys.stderr)

if "category" in test.columns:  # labelled input: report score per stage (ML stage is optimistic on training data)
    print(f"  accuracy: {(pred == test.category).mean():.1%}", file=sys.stderr)
    for s in ["rules", "ml", "haiku", "ml_guess"]:
        m = stage == s
        if m.any():
            ok = (pred[m] == test.category[m]).sum()
            print(f"  {s:<6} rows {m.sum():>4}  right {ok:>4}  wrong {m.sum() - ok:>3}", file=sys.stderr)
    known = pred != "unknown"
    right = (pred[known] == test.category[known]).sum()
    print(f"  score/100: {100 * (right - (known.sum() - right)) / len(test):.0f}", file=sys.stderr)

pd.DataFrame({"id": test.id, "category": pred}).to_csv("predictions.csv", index=False)
