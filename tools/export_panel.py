"""
export_panel.py -- Run ONCE by the authors; builds the shared panel from the private archive.
=============================================================================================
This script is included for provenance and is NOT run by end users: it requires access to the
raw enriched JSONL archive. Its output, `panel_bge_m3.parquet`, is the shared data file, and
`reproduce.py` uses nothing else.

Panel contents (NO news text -- numerical derivatives only):
  identifiers : ticker, week_start
  target      : target_log_return (log return of week t+1), log_return (week t, for the
                momentum benchmark)
  features    : 66 tabular variables (price, macro, LLM news scalars, category shares)
  embeddings  : vec_bge_m3_0..1023 (stock news), macro_vec_0..1023 (sector/macro news)

Dimensionality reduction is deliberately NOT applied here: PCA must be fitted inside each
walk-forward fold, so `reproduce.py` performs it.
"""
from pathlib import Path                                      # filesystem paths
import hashlib                                                # SHA-256 integrity digest
import pandas as pd                                           # tabular data

import bist_llm.mainOptuna as mo                              # dataset assembly (private archive)

OUT = Path(__file__).resolve().parents[1] / "data"            # repository data directory
OUT.mkdir(parents=True, exist_ok=True)                        # create if absent


def main():
    df, vec, mvec = mo.get_dataset("bge_m3")                  # encoder used by the selected model
    cols = mo.feature_cols("M3_Full", df)                     # the 66 M3_Full tabular features
    keep = ["ticker", "week_start", "target_log_return",      # identifiers and target
            "log_return", "has_news"] + cols + vec + mvec     # plus features and embeddings
    keep = list(dict.fromkeys(keep))                          # preserve order, drop duplicates
    panel = df[keep].copy()                                   # retain only what is shared
    panel["week_start"] = pd.to_datetime(panel["week_start"])  # ensure datetime dtype

    path = OUT / "panel_bge_m3.parquet"                       # output file
    panel.to_parquet(path, index=False, compression="zstd")   # compressed columnar format
    sha = hashlib.sha256(path.read_bytes()).hexdigest()       # integrity digest
    (OUT / "panel_bge_m3.sha256").write_text(f"{sha}  panel_bge_m3.parquet\n")   # record it

    print(f"Written : {path}")                                # report
    print(f"Rows    : {len(panel):,} | Columns: {panel.shape[1]:,}")
    print(f"Size    : {path.stat().st_size / 1e6:.1f} MB")
    print(f"SHA-256 : {sha}")


if __name__ == "__main__":
    main()
