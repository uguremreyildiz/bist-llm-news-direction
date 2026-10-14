# A3 — Dependence-robust confidence intervals

The eight stocks of a week share macro news and common shocks, so the 1,152 stock-week
predictions are not independent. `inference.py` replaces the independence assumption of the
exact binomial test with a moving-block bootstrap over whole weeks (Künsch, 1989): blocks of
8 consecutive weeks are resampled with all 8 stocks of each week kept together, which treats a
week as a cluster and preserves serial dependence across adjacent weeks.

**Input:** `outputs/oos_predictions.csv`. **Runtime:** a few seconds.
**Outputs:** `revision/outputs/tables/T_A3_inference.csv`, `A3_summary.json`.

| | Estimate | 95% CI | one-sided p (H0: accuracy ≤ 0.5) |
|---|---|---|---|
| Accuracy — exact binomial, independence | 0.538 | — | 0.005 |
| Accuracy — block bootstrap (L = 8, B = 5,000) | 0.538 | 0.487 – 0.588 | 0.075 |
| Macro-F1 — block bootstrap | 0.536 | 0.486 – 0.584 | — |

Block lengths of 4 and 12 weeks give p = 0.067 and 0.085. The intra-week correlation of the
hit indicator is 0.21.
