# Anyfin Ledger Labeller

Anyfin × KTH case night, Track A: label bank transactions with one of 19 categories, or `unknown`.

- **Pipeline diagram:** https://claude.ai/artifact/9sQfuRUzzUz71Pij7gHMpA

- **Run:** `pip install -r requirements.txt` then `python classify.py validation.csv` (reads `ANTHROPIC_API_KEY` from the environment; writes `predictions.csv`).

- **Approach:** deterministic rules (own-account transfer pairing, card bills, mortgages) → TF-IDF + logistic regression when ≥0.8 confident → Claude Haiku 5.5 on deduplicated merchants (batched, parallel, thinking off) → cross-check Haiku against the ML model; disagreement or doubt → `unknown`.

- **Measured:** 1,183 validation rows in ~11 s for $0.006; on held-out training customers 97.9% precision, score 92/100, all rule and ML labels correct.

- **Cost at 10M rows/month:** ~$50 as built (only deduplicated merchants reach the LLM); with a persistent merchant cache only new merchants are paid for, a few dollars a month.

- **With another day:** a merchant dictionary and periodic retraining of the ML model on cross-checked Haiku labels (distillation), plus normalisation for more bank text formats.
