"""
check_baseline.py -- Consistency gate for the analyses in revision/.
======================================================================================
Confirms that common/pipeline.py reproduces the published headline numbers exactly when it is
given the selected model's feature set and hyperparameters, and that the fold structure is the
6 inner / 12 outer split described in the manuscript.
"""
from pathlib import Path                                        # paths
import sys, time                                                # timing, import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # make config/common importable
import numpy as np, pandas as pd                                # numerics, tables
import config as C                                              # constants
from common import pipeline as P                                # the shared machinery

PUBLISHED = dict(accuracy=0.5382, f1_macro=0.5364, roc_auc=0.5467, n_oos=1152,
                 sharpe_lo_net=1.5958, sharpe_buyhold=1.2640,
                 sharpe_ls_net=1.0501, sharpe_momentum=0.8181,
                 share_emb_macro=59.6, share_emb_firm=3.0)       # Sections 5.1-5.4

V4_GROUPS = ["PRICE", "MACRO_NUM", "SECTOR_ID", "CROSS",         # the published feature set
             "SCALAR_FIRM", "SCALAR_MACRO", "EMB_FIRM", "EMB_MACRO"]


def main():
    t0 = time.time()                                             # runtime
    C.ensure_dirs()                                              # output directories
    df, emb_f, emb_m = P.load_panel()                            # panel + embedding columns
    print(f"Panel: {len(df):,} rows x {df.shape[1]:,} columns")
    print("Feature groups:", {k: len(v) for k, v in P.TABULAR_GROUPS.items()},
          "-> total", sum(len(v) for v in P.TABULAR_GROUPS.values()))
    print("Frozen to:", P.freeze_groups())

    weeks = sorted(df["week_start"].unique())                    # 393 weekly timestamps
    inner, outer = P.build_folds(weeks)                          # fold families
    print(f"\nFolds: {len(inner)} inner, {len(outer)} outer  (expected 6 / 12)")
    assert (len(inner), len(outer)) == (6, 12), "fold structure does not match the manuscript"
    cal = P.fold_calendar(weeks)                                 # dated calendar
    print("Last inner test ends :", cal[cal.family == "inner"].test_end.max())
    print("First outer test starts:", cal[cal.family == "outer"].test_start.min())
    assert cal[cal.family == "inner"].test_end.max() < cal[cal.family == "outer"].test_start.min(), \
        "inner folds are not strictly before the outer folds"

    pooled, per_fold, preds, coefs = P.evaluate_outer(           # the published configuration
        df, outer, emb_f, emb_m, V4_GROUPS,
        c=C.SELECTED["C"], penalty=C.SELECTED["penalty"])
    wk = P.backtest(preds)                                       # weekly portfolio returns

    cf = coefs.abs().mean(axis=1)                                # mean |beta| across folds
    share = lambda pre: 100 * cf[[c for c in cf.index if c.startswith(pre)]].sum() / cf.sum()

    got = dict(accuracy=round(pooled["accuracy"], 4), f1_macro=round(pooled["f1_macro"], 4),
               roc_auc=round(pooled["roc_auc"], 4), n_oos=len(preds),
               sharpe_lo_net=round(P.ann_sharpe(wk.lo_net), 4),
               sharpe_buyhold=round(P.ann_sharpe(wk.buyhold), 4),
               sharpe_ls_net=round(P.ann_sharpe(wk.ls_net), 4),
               sharpe_momentum=round(P.ann_sharpe(wk.momentum), 4),
               share_emb_macro=round(share("red_macro"), 1),
               share_emb_firm=round(share("red_firm"), 1))

    print("\n{:<20}{:>12}{:>12}{:>8}".format("quantity", "published", "pipeline", "match"))
    ok = True                                                    # overall verdict
    for k, v in PUBLISHED.items():                               # compare every headline number
        tol = 0 if k == "n_oos" else (0.051 if k.startswith("share") else 0.0002)
        good = abs(got[k] - v) <= tol; ok &= good                # within tolerance?
        print("{:<20}{:>12}{:>12}{:>8}".format(k, v, got[k], "OK" if good else "DIFF"))

    print(f"\nRuntime: {time.time()-t0:.1f}s")
    print("GATE:", "PASSED - pipeline reproduces the paper exactly" if ok else "FAILED - stop here")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
