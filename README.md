# Weekly Direction Forecasting for BIST Equities with Multi-LLM News Enrichment

Replication material for the paper *(under review)*.

This repository contains everything needed to (i) reproduce the out-of-sample results, economic
backtest and coefficient attribution of the selected model (paper, Sections 5.1, 5.3 and 5.4), and (ii) inspect the full data-construction protocol — news acquisition, LLM
prompts, cross-model consensus, embedding, and weekly aggregation.

**Full news article texts are not redistributed.** They are third-party copyrighted material
obtained from publishers via the GDELT Document 2.0 API, and their licences do not permit
redistribution. What is shared instead is the numerical panel derived from them, which is
sufficient to reproduce these results.

---

## 1. Quick start

```bash
pip install -r requirements.txt
python reproduce.py
```

Runtime: a few seconds on a laptop. Expected output:

```
Panel: 3,144 rows x 2,117 columns | 393 weeks | 12 out-of-sample folds

Out-of-sample classification
  observations        1152
  accuracy            0.5382
  macro-F1            0.5364
  ROC-AUC             0.5467
  binomial p (1-sided)  0.005169

Annualised Sharpe ratio
  model long-only (net)   1.5958
  buy-and-hold            1.2640
  model long-short (net)  1.0501
  price momentum          0.8181

Coefficient share by feature family (%)
  emb_reduced_macro     59.6
  macro_price           16.6
  sentrel_other         12.9
  stock_price            7.2
  emb_reduced_stock      3.0
  emb_scalar             0.7
```

---

## 2. Contents

```
data/
  panel_bge_m3.parquet        the shared analysis panel (28 MB, zstd)
  panel_bge_m3.sha256         integrity checksum
reproduce.py                  standalone reproduction of the selected model
prompts/
  stock_news_prompt_tr.txt    verbatim stock-level LLM prompt
  macro_news_prompt_tr.txt    verbatim macro/sector LLM prompt
pipeline/
  01_gdelt_retrieval.py       news acquisition from the GDELT Document 2.0 API
  02_llm_enrichment.py        three-model enrichment (OpenRouter)
  03_consensus_and_weekly_features.py   consensus, embedding, weekly aggregation
tools/
  export_panel.py             how data/panel_bge_m3.parquet was produced (authors only)
outputs/                      written by reproduce.py
revision/                     supplementary analyses; run after reproduce.py
  A3_inference/               moving-block bootstrap intervals and p-values (Section 5.1)
  A9_fold_calendar/           dated fold calendar (Appendix Table A1)
  common/, config.py          shared fold, fit and bootstrap routines
RESULTS.md                    mapping from paper tables/figures to files
SEARCH_SPACE.md               model-selection protocol and hyperparameter search space
requirements.txt              exact package versions used for the reported results
LICENSE                       MIT (code)
LICENSE-DATA                  CC BY 4.0 (data/)
```

---

## 3. The shared panel

`data/panel_bge_m3.parquet` — 3,144 rows (8 tickers x 393 weeks, 2018-01 to 2025-08),
2,117 columns:

| Block | Columns | Description |
|---|---|---|
| Identifiers | `ticker`, `week_start` | week beginning Monday |
| Target | `target_log_return` | log return of week *t+1*; the label is `1[target_log_return > 0]` |
| Price | 9 | own past closing-price features only (lags, moving averages, volatility, momentum) |
| Macro numeric | 12 | USDTRY, EURTRY, Brent, gold and VIX log changes and derived indicators |
| News scalars | 45 | consensus sentiment moments, cross-model disagreement, category shares, news counts |
| Stock news text | `vec_bge_m3_0..1023` | BGE-M3 embedding of the three-model consensus summary |
| Macro news text | `macro_vec_0..1023` | BGE-M3 embedding of the three-model consensus macro summary |

Weeks with no qualifying news carry zero vectors, flagged by `has_news`.

**No article text, headline, URL, publisher name or publication date is present.** The two
embedding blocks encode the *models' own summaries*, not the source articles, and the
transformation is not invertible.

The panel is deliberately supplied **before** dimensionality reduction. PCA is refitted inside
every walk-forward fold on that fold's training rows only; supplying already-reduced features
would both hide this mechanism and prevent an independent check of it. `reproduce.py` performs
the fold-wise reduction, so the leakage control is verifiable rather than asserted.

---

## 4. What `reproduce.py` does

The selected configuration is **M3_Full / BGE-M3 / PCA(32 per stream) / L1-penalised logistic
regression** (`C = 0.02005544632966435`), chosen by inner-fold macro-F1 before any
out-of-sample week was touched. `SEARCH_SPACE.md` documents the 152-configuration grid, the
TPE budget and the per-classifier parameter ranges from which this configuration emerged.

1. Builds the expanding-window folds: first training block 156 weeks, test block 12 weeks,
   step 12 weeks; the first 60% of the sample is the tuning region, and only folds beginning
   after that boundary are evaluated — 12 out-of-sample folds, 1,152 predictions.
2. Per fold, fits PCA (q = 32) separately on the stock and macro embedding blocks, using **only
   the training rows that carry news**; news-free rows keep their zero vectors.
3. Per fold, fits `StandardScaler` on training rows only, then the classifier
   (`class_weight="balanced"`, `solver="liblinear"`, `random_state=42`).
4. Reports out-of-sample accuracy, macro-F1, ROC-AUC, a one-sided binomial test against a
   coin flip, per-ticker accuracy, the confusion matrix, coefficient importance by feature
   family, and the annualised Sharpe ratios of the four strategies.

Every news document enters the Monday–Sunday week of its date (the İş Bankası weekly bulletin
enters the week it reviews), so weekend news belongs to week *t*; the features of week *t* predict the direction of week *t+1* (paper, Sections 4.5 and 6.2).

### Backtest conventions

Positions are taken on the model's directional signal and held for one week: long-only takes a
unit long position on an "up" prediction and stays flat otherwise; long-short takes +1 on "up"
and −1 on "down". Transaction costs of 10 bps per side are charged on realised turnover,
computed per ticker as the absolute change in position (the first week of a position is charged
as an entry). Weekly portfolio returns are the equal-weighted cross-sectional mean across the
eight stocks. **The Sharpe ratio is computed on excess-of-zero returns — no risk-free rate is
subtracted** — and annualised as `sqrt(52) * mean / std` with a population standard deviation
(`ddof = 0`). Benchmarks are buy-and-hold (always long) and a price-momentum rule (long when
last week's return was positive, flat otherwise), evaluated on exactly the same weeks.

---

## 5. Data-construction pipeline

`pipeline/` documents how the panel was built from raw news. These scripts require access to
the raw article archive and paid LLM API credit; they are provided for inspection and are not
required to reproduce the results.

| Step | Script | Output |
|---|---|---|
| 1 | `01_gdelt_retrieval.py` | weekly JSONL files of articles (one query per month, per ticker and per macro source) |
| 2 | `02_llm_enrichment.py` | per-model enrichment: category, sentiment, relevance, summary |
| 3 | `03_consensus_and_weekly_features.py` | consensus records, BGE-M3 embeddings, weekly panel |

Set `OPENROUTER_API_KEY` in the environment before running step 2; no credential is stored in
this repository.

**Look-ahead control.** Each article is sent to each model as a single, independent `user`
message, with no system message and no conversation history, so no other article — least of
all a later one — can enter the model's context. The macro prompt states the constraint
explicitly as well. The scoring rubrics, the category definitions and the 5,000 / 15,000
character truncation limits are reproduced verbatim in `prompts/`.

**Consensus rule.** Sentiment and relevance are the plain mean across the three models;
`std_sentiment` records their disagreement; the category is a strict-majority vote that returns
`"Mixed"` when no majority exists. An article enters the weekly aggregates when its
*three-model mean* relevance exceeds 0.05 — a filter on the averaged score, so unanimity is not
required and no single model can veto an article. Within a week, surviving articles are averaged
with equal weights; relevance serves as the filter and is never used as an averaging weight.

---

## 6. Sample

Eight liquid BIST-30 constituents with news coverage over the whole window:
`AKBNK, ASELS, EREGL, KRDMD, SAHOL, TCELL, THYAO, YKBNK`.
Weekly frequency, 2018-01 to 2025-08 (393 weeks).

News sources:

* GDELT Project Document 2.0 API — per-ticker queries, plus general-economy and
  political-ambiguity macro queries (https://api.gdeltproject.org/api/v2/doc/doc)
* Central Bank of the Republic of Türkiye (CBRT) press releases
  (https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/duyurular/basin)
* Weekly economic bulletin of Türkiye İş Bankası
  (https://ekonomi.isbank.com.tr/haftalik-bulten)

Price series were obtained via Yahoo Finance (`yfinance`); macroeconomic series
(USDTRY, EURTRY, Brent, gold, VIX) are from the same public source. All are included in the
shared panel in transformed form (log returns and derived indicators).

---

## 7. Licence

* **Code** (`reproduce.py`, `pipeline/`, `tools/`, `prompts/`) — MIT, see `LICENSE`.
* **Data** (`data/`) — Creative Commons Attribution 4.0 International, see `LICENSE-DATA`.

If you use this material, please cite the paper.

---

## 8. Contact

Questions about the replication material may be addressed to the corresponding author.
