"""
A3 -- Dependence-robust confidence intervals and significance for the selected model.
=====================================================================================
The eight stocks of a week share macro news and common shocks, so the 1,152 stock-week
predictions are not independent and the exact binomial test reported in the paper overstates
the effective sample size. Inference is therefore based on a moving-block bootstrap over whole
weeks (Kunsch, 1989): blocks of 8 consecutive weeks are resampled with all 8 stocks of each week
kept together, which treats a week as a cluster and preserves serial dependence across adjacent
weeks. The binomial test is kept only as the naive reference.

Input : outputs/oos_predictions.csv (written by reproduce.py)
Outputs: revision/outputs/tables/T_A3_inference.csv, A3_summary.json
Runtime: a few seconds.
"""
from pathlib import Path                                        # paths
import sys, json                                                # json, import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # make config/common importable
import numpy as np                                              # numerics
import pandas as pd                                             # tables
from scipy.stats import binomtest                               # exact binomial test
import config as C                                              # constants and paths
from common import pipeline as P                                # shared machinery


def accuracy(y, p):
    """Share of correct directional predictions."""
    return float(np.mean(y == p))


def macro_f1(y, p):
    """Macro-F1 over the two classes, from confusion counts (fast enough for the bootstrap)."""
    tp = np.sum((y == 1) & (p == 1)); tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    f_up = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f_dn = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return float((f_up + f_dn) / 2)


def main():
    C.ensure_dirs()                                             # output directory
    pr = pd.read_csv(C.PRED_CSV, parse_dates=["week_start"])    # 1,152 out-of-sample predictions
    pr = pr.sort_values(["week_start", "ticker"]).reset_index(drop=True)   # week-major order
    W = pr.week_start.nunique()                                 # 144 weeks
    assert len(pr) == 8 * W == 1152, "expected 144 weeks x 8 stocks"
    Y = pr.y_true.values.reshape(W, 8)                          # outcomes, weeks x stocks
    Pd = pr.y_pred.values.reshape(W, 8)                         # predictions, weeks x stocks
    y, p = Y.ravel(), Pd.ravel()                                # flattened
    est_acc, est_f1 = accuracy(y, p), macro_f1(y, p)            # point estimates
    assert abs(est_acc - 0.5382) < 1e-4 and abs(est_f1 - 0.5364) < 1e-4, "not the published model"

    # Naive reference: exact binomial test (paper, Section 3.8)
    k = int((y == p).sum())                                     # correct predictions
    p_binom = binomtest(k, len(y), 0.5, alternative="greater").pvalue

    # Dependence: correlation of the hit indicator among the 8 stocks of a week
    hits = (Y == Pd).astype(float); a_t = hits.mean(axis=1)     # weekly accuracy
    msb = 8 * np.var(a_t, ddof=1)                               # between-week mean square
    msw = np.sum((hits - a_t[:, None]) ** 2) / (W * 7)          # within-week mean square
    icc = (msb - msw) / (msb + 7 * msw)                         # intra-week correlation

    # Moving-block bootstrap over whole weeks
    def draws(L):
        """Bootstrap accuracy and macro-F1 for block length L."""
        idx = P.week_block_indices(W, L, C.N_BOOT, C.BOOT_SEED)  # resampled week positions
        return (np.array([accuracy(Y[i].ravel(), Pd[i].ravel()) for i in idx]),
                np.array([macro_f1(Y[i].ravel(), Pd[i].ravel()) for i in idx]))

    acc_b, f1_b = draws(C.BLOCK_LEN)                            # main specification, L = 8
    acc_lo, acc_hi = P.boot_ci(acc_b); f1_lo, f1_hi = P.boot_ci(f1_b)
    p_boot = P.boot_p(acc_b, est_acc, 0.5)                      # H0: accuracy <= 0.5
    p_sens = {L: round(P.boot_p(draws(L)[0], est_acc, 0.5), 4) for L in (4, 12)}   # block-length check

    table = pd.DataFrame([
        dict(statistic="accuracy", method="exact binomial (independence, naive reference)",
             estimate=round(est_acc, 4), ci_low=None, ci_high=None, p_one_sided=round(p_binom, 4)),
        dict(statistic="accuracy", method=f"moving-block bootstrap over weeks (L={C.BLOCK_LEN}, B={C.N_BOOT})",
             estimate=round(est_acc, 4), ci_low=round(acc_lo, 4), ci_high=round(acc_hi, 4),
             p_one_sided=round(p_boot, 4)),
        dict(statistic="macro_f1", method=f"moving-block bootstrap over weeks (L={C.BLOCK_LEN}, B={C.N_BOOT})",
             estimate=round(est_f1, 4), ci_low=round(f1_lo, 4), ci_high=round(f1_hi, 4), p_one_sided=None)])
    table.to_csv(C.OUT_TABLES / "T_A3_inference.csv", index=False)

    summary = dict(n_stock_weeks=len(y), n_weeks=W, correct=k, intra_week_correlation=round(icc, 3),
                   accuracy=round(est_acc, 4), accuracy_ci95=[round(acc_lo, 4), round(acc_hi, 4)],
                   macro_f1=round(est_f1, 4), macro_f1_ci95=[round(f1_lo, 4), round(f1_hi, 4)],
                   p_binomial=round(p_binom, 4), p_block_bootstrap=round(p_boot, 4),
                   p_block_bootstrap_by_block_length=p_sens, block_length=C.BLOCK_LEN, n_boot=C.N_BOOT)
    (C.OUT_TABLES / "A3_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
