"""
Step 3 — Cross-model consensus, embedding, and weekly aggregation.
==================================================================
Reference implementation of the feature-construction stage (Sections 3.2–3.4 and 4.5 of the paper).
It turns the three per-model enrichment streams into the weekly news aggregates (mean embedding,
sentiment moments, cross-model disagreement, article counts, category shares) from which the
panel used by `../reproduce.py` is built.
No price data is touched here; prices are merged afterwards on (ticker, week_start).

Article-level consensus
-----------------------
The three enrichment streams are matched on the article URL within each source file (an article
returned by two adjacent monthly queries, or listed under two tickers, stays a separate record),
then reduced as follows:

    sentiment       = mean over the models that returned a value
    relevance       = mean over the models that returned a value
    std_sentiment   = standard deviation across models  (a disagreement feature)
    category        = MAJORITY vote; "Mixed" when no strict majority exists
    summary text    = the models' `core_signal` (stock) / `macro_summary` (macro) strings
                      concatenated with a blank line between them

Relevance filter
----------------
An article enters the weekly aggregates when its CONSENSUS (three-model mean) relevance exceeds
0.05. This is a filter on the averaged score, not a per-model vote: unanimity is not required,
and a single model cannot veto an article. The threshold only removes items the models jointly
marked as incidental mentions (the prompts instruct a score of 0.05 for the "Dolaylı" category).

Embedding
---------
The concatenated summary is encoded with BAAI/bge-m3 (1024-d) and L2-normalised. Because the
text encoded is the models' own summary rather than the article, no publisher text is retained.
The alternative sentence-transformer encoders benchmarked in the paper are listed in ENCODERS;
FastText (300-d, Equation (10) of the paper) is listed with the other five in SEARCH_SPACE.md.

Weekly aggregation
------------------
Within each (ticker, week) cell the surviving articles are averaged with EQUAL weights. Relevance
serves as the filter and is never used as an averaging weight: the embeddings are already L2-normalised, so
re-weighting them by relevance would drag the cell towards the direction of high-relevance
vectors and would apply the same model-derived score twice.

Usage
-----
    python 03_consensus_and_weekly_features.py --stock ./enriched_stock --macro ./enriched_macro \
                                               --out ./weekly
"""
import argparse                                                # CLI
import json                                                    # JSONL
import statistics                                              # mean / stdev
from collections import Counter                                # majority vote
from pathlib import Path                                       # paths

import numpy as np                                             # numerics
import pandas as pd                                            # tables

MODELS = ["gemma", "gpt", "llama"]                             # the three enrichment streams
REL_THRESHOLD = 0.05                                           # consensus-relevance filter
ENCODERS = {                                                   # benchmarked text encoders
    "bge_m3": "BAAI/bge-m3",                                              # selected, 1024-d
    "bert_tr": "dbmdz/bert-base-turkish-cased",                           # 768-d
    "standard_tr": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",    # 768-d
    "finturkbert": "ff112/FinTurkBERT",                                   # 768-d
    "mpnet_multi": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 768-d
}
SECTORS = {"Bank_Holding": "BH", "Industry": "IND", "Consumer_Telecom": "CT"}   # Aviation dropped
CATEGORIES = ["Finansal", "Kurumsal", "Operasyonel", "Dolaylı", "Mixed"]        # one-hot targets


# ── Article-level consensus ───────────────────────────────────────────────
def mean_of(vals):
    """Mean over the models that returned a value; None when all three failed."""
    v = [x for x in vals if x is not None]                     # drop missing
    return float(statistics.fmean(v)) if v else None           # consensus scalar


def std_of(vals):
    """Cross-model dispersion; 0.0 when fewer than two models responded."""
    v = [x for x in vals if x is not None]                     # available scores
    return float(statistics.stdev(v)) if len(v) > 1 else 0.0   # disagreement feature


def majority_category(vals):
    """Modal category across the models; 'Mixed' when there is no strict majority."""
    v = [x for x in vals if x]                                 # available labels
    if not v:
        return "Mixed"                                         # nothing to vote on
    top, n = Counter(v).most_common(1)[0]                      # modal label and its count
    return top if n > len(v) / 2 else "Mixed"                  # strict majority required


def concat_summaries(per_model):
    """Join the models' summaries in a fixed order; this string is what gets embedded."""
    parts = [per_model[m] for m in MODELS if per_model.get(m)]  # fixed gemma → gpt → llama order
    return "\n\n".join(parts)                                   # blank line between summaries


def merge_streams(dirs, kind):
    """Match the three enrichment streams on (source file, URL) and reduce them to consensus records."""
    by_key = {}                                                 # (file, url) -> {model: enrichment}
    meta = {}                                                   # (file, url) -> identity fields
    for model, d in dirs.items():                               # each model's directory
        for path in sorted(Path(d).rglob("*.jsonl")):           # each monthly file
            rel = path.relative_to(d).as_posix()                # same relative path in every model dir
            for line in path.open(encoding="utf-8"):            # each article
                if not line.strip():
                    continue
                r = json.loads(line)                            # enriched record (no body text)
                u = r.get("url")
                if not u:
                    continue
                by_key.setdefault((rel, u), {})[model] = r.get("enrichment") or {}
                meta.setdefault((rel, u), {k: r.get(k) for k in ("ticker", "date", "week_start")})

    rows = []                                                   # consensus article rows
    for (rel, u), per_model in by_key.items():                  # each article record
        row = dict(meta[(rel, u)], url=u, n_models=len(per_model))   # identity + coverage
        if kind == "stock":
            row["sentiment"] = mean_of([per_model.get(m, {}).get("sentiment") for m in MODELS])
            row["relevance"] = mean_of([per_model.get(m, {}).get("relevance") for m in MODELS])
            row["std_sentiment"] = std_of([per_model.get(m, {}).get("sentiment") for m in MODELS])
            row["category"] = majority_category(
                [per_model.get(m, {}).get("category") for m in MODELS])
            row["summary"] = concat_summaries(
                {m: per_model.get(m, {}).get("core_signal") for m in MODELS})
        else:                                                   # macro: one score block per sector
            for sec, tag in SECTORS.items():                    # BH / IND / CT
                sa = {m: (per_model.get(m, {}).get("sector_analysis") or {}).get(sec, {})
                      for m in MODELS}                          # per-model sector block
                row[f"{tag}_sentiment"] = mean_of([sa[m].get("sentiment") for m in MODELS])
                row[f"{tag}_relevance"] = mean_of([sa[m].get("relevance") for m in MODELS])
                row[f"{tag}_std_sentiment"] = std_of([sa[m].get("sentiment") for m in MODELS])
            row["summary"] = concat_summaries(
                {m: per_model.get(m, {}).get("macro_summary") for m in MODELS})
        rows.append(row)
    return pd.DataFrame(rows)                                   # one row per article


# ── Embedding ─────────────────────────────────────────────────────────────
def embed(texts, model_key="bge_m3", batch=32):
    """Encode the consensus summaries and L2-normalise; returns an (n, d) float32 array."""
    from sentence_transformers import SentenceTransformer       # imported lazily
    enc = SentenceTransformer(ENCODERS[model_key])              # selected encoder
    V = enc.encode(list(texts), batch_size=batch, convert_to_numpy=True,
                   show_progress_bar=True)                      # dense vectors
    n = np.linalg.norm(V, axis=1, keepdims=True)                # row norms
    return (V / np.where(n == 0, 1.0, n)).astype("float32")     # unit-length rows


# ── Weekly aggregation ────────────────────────────────────────────────────
def weekly_stock(df, V, prefix="vec_bge_m3"):
    """Collapse stock articles to one row per (ticker, week): mean vector + sentiment moments."""
    d = df[df["relevance"] > REL_THRESHOLD].copy()              # consensus-relevance filter
    vec_cols = [f"{prefix}_{i}" for i in range(V.shape[1])]     # embedding column names
    d[vec_cols] = V[d.index.values]                             # attach the vectors
    g = d.groupby(["ticker", "week_start"])                     # weekly cells
    out = g[vec_cols].mean().reset_index()                      # EQUAL-weight mean embedding
    sent = g.agg(weighted_sentiment=("sentiment", "mean"),      # cell-mean sentiment
                 min_sent=("sentiment", "min"),                 # weekly minimum
                 max_sent=("sentiment", "max"),                 # weekly maximum
                 std_sent=("sentiment", "std"),                 # within-week dispersion
                 avg_std_sent=("std_sentiment", "mean"),        # mean cross-model disagreement
                 max_std_sent=("std_sentiment", "max"),         # peak cross-model disagreement
                 news_count=("url", "size")).reset_index()      # article count
    cat = pd.get_dummies(d["category"]).reindex(columns=CATEGORIES, fill_value=0).astype(float)
    cat = pd.concat([d[["ticker", "week_start"]], cat.add_prefix("cat_")], axis=1)
    cat = cat.groupby(["ticker", "week_start"]).mean().reset_index()   # category shares
    return out.merge(sent, on=["ticker", "week_start"]).merge(cat, on=["ticker", "week_start"]) \
              .fillna(0.0)


def weekly_macro(df, V, prefix="macro_vec"):
    """Collapse macro documents to one row per week, per sector block (BH / IND / CT)."""
    vec_cols = [f"{prefix}_{i}" for i in range(V.shape[1])]     # embedding column names
    frames = []                                                 # per-sector results
    for tag in SECTORS.values():                                # each retained sector
        d = df[df[f"{tag}_relevance"] > REL_THRESHOLD].copy()   # sector-specific filter
        d[vec_cols] = V[d.index.values]                         # attach the vectors
        g = d.groupby("week_start")                             # weekly cells (no ticker)
        f = g[vec_cols].mean().reset_index()                    # EQUAL-weight mean embedding
        s = g.agg(**{f"{tag}_weighted_sentiment": (f"{tag}_sentiment", "mean"),
                     f"{tag}_min_sent": (f"{tag}_sentiment", "min"),
                     f"{tag}_max_sent": (f"{tag}_sentiment", "max"),
                     f"{tag}_std_sent": (f"{tag}_sentiment", "std"),
                     f"{tag}_avg_std_sent": (f"{tag}_std_sentiment", "mean"),
                     f"{tag}_news_count": ("url", "size")}).reset_index()
        frames.append(f.merge(s, on="week_start").assign(sector=tag))
    return pd.concat(frames, ignore_index=True).fillna(0.0)     # long format: week × sector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True, help="parent dir holding gemma/ gpt/ llama/")
    ap.add_argument("--macro", required=True, help="parent dir holding gemma/ gpt/ llama/")
    ap.add_argument("--out", required=True)                     # output directory
    ap.add_argument("--encoder", default="bge_m3", choices=list(ENCODERS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for kind, root, agg, pre in [("stock", a.stock, weekly_stock, f"vec_{a.encoder}"),
                                 ("macro", a.macro, weekly_macro, "macro_vec")]:
        dirs = {m: Path(root) / m for m in MODELS}              # one directory per model
        df = merge_streams(dirs, kind).reset_index(drop=True)   # consensus article table
        print(f"{kind}: {len(df):,} articles after 3-model merge")
        V = embed(df["summary"], a.encoder)                     # encode consensus summaries
        wk = agg(df, V, pre)                                    # weekly aggregation
        wk.to_parquet(out / f"weekly_{kind}_{a.encoder}.parquet", index=False)
        print(f"{kind}: {len(wk):,} weekly rows -> weekly_{kind}_{a.encoder}.parquet")


if __name__ == "__main__":
    main()
