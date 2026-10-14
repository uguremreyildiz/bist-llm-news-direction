"""
common/pipeline.py -- Shared machinery for the supplementary analyses.
===========================================================================
One fold builder, one fit/predict routine, one backtest, one bootstrap, shared by the scripts in
revision/ so that they follow the protocol of the paper (Section 3.7).

Protocol (identical to reproduce.py):
  * expanding window: first training block 156 weeks, test block 12, step 12
  * the first 60% of weeks is the tuning region -> 6 inner folds; folds whose test block
    begins after that boundary -> 12 outer folds; the single fold straddling the boundary is
    discarded (its test block crosses the boundary)
  * PCA(32) per embedding stream, fitted on news-bearing TRAINING rows only
  * StandardScaler fitted on TRAINING rows only
  * LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)

Feature groups. The 66 tabular features are split into six disjoint groups. Three of them need
care and are NOT assigned by name alone:
  SECTOR_ID  -- static sector dummies; not news at all. Giving them to the firm-news variant
                only would hand it free identity information, so they belong to the baseline.
  CROSS      -- stock_macro_alignment needs BOTH streams to exist; it cannot be attributed to
                either one and enters only the full model.
  SCALAR_FIRM additionally contains the two sentiment x price interactions, because price is
                already in the baseline, so their incremental content is the firm-news part.
"""
from pathlib import Path                                        # paths
import json                                                     # group freeze
import numpy as np                                              # numerics
import pandas as pd                                             # tables
from sklearn.decomposition import PCA                           # dimensionality reduction
from sklearn.preprocessing import StandardScaler                # scaling
from sklearn.linear_model import LogisticRegression             # classifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,   # metrics
                             recall_score, confusion_matrix)

import sys                                                      # import path juggling
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # make `config` importable
import config as C                                              # protocol constants and paths

# -- Explicit feature-group definitions ------------------------------------
PRICE = ["log_return", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
         "ret_ma3", "ret_ma5", "ret_std5", "ret_momentum"]                      # 9, own price
MACRO_NUM = ["USDTRY_log_return", "EURTRY_log_return", "BRENT_log_return",      # 12, market data
             "GOLD_log_return", "VIX_log_change", "USDTRY_log_return_lag1",
             "EURTRY_log_return_lag1", "BRENT_log_return_lag1",
             "GOLD_log_return_lag1", "VIX_log_change_lag1", "USDTRY_ma3", "VIX_ma3"]
SECTOR_ID = ["stock_sector_BH", "stock_sector_IND", "stock_sector_CT"]          # 3, static identity
CROSS = ["stock_macro_alignment"]                                               # 1, needs both streams
SCALAR_MACRO = ["macro_weighted_sentiment", "macro_min_sent", "macro_max_sent", # 11, macro-news derived
                "macro_std_sent", "macro_rel_mean", "macro_rel_max",
                "macro_sent_ma3", "macro_sent_momentum", "macro_emb_norm",
                "macro_cos_lag1", "macro_cos_lag3"]
SCALAR_FIRM = ["has_news", "weighted_sentiment", "min_sent", "max_sent",        # 30, firm-news derived
               "std_sent", "avg_std_sent", "max_std_sent", "sent_lag1",
               "sent_lag3", "sent_lag5", "sent_ma3", "sent_ma10", "sent_momentum",
               "sent_volatility", "news_count", "days_since_news", "news_burst",
               "sent_price_divergence", "sent_price_agree", "rel_mean", "rel_max",
               "cat_Dolaylı", "cat_Finansal", "cat_Kurumsal", "cat_Mixed",
               "cat_Operasyonel", "stock_emb_norm", "stock_emb_norm_z",
               "stock_cos_lag1", "stock_cos_lag3"]
EMB_FIRM_PREFIX, EMB_MACRO_PREFIX = "vec_bge_m3_", "macro_vec_"                 # 1024 each

TABULAR_GROUPS = dict(PRICE=PRICE, MACRO_NUM=MACRO_NUM, SECTOR_ID=SECTOR_ID,
                      CROSS=CROSS, SCALAR_FIRM=SCALAR_FIRM, SCALAR_MACRO=SCALAR_MACRO)


# ==========================================================================
# Data
# ==========================================================================
def load_panel():
    """Load the shared panel and return it with its embedding column lists."""
    df = pd.read_parquet(C.PANEL)                                # the single shared data file
    df["week_start"] = pd.to_datetime(df["week_start"])          # datetime dtype
    df = df.sort_values(["ticker", "week_start"]).reset_index(drop=True)   # deterministic order
    emb_firm = [c for c in df.columns if c.startswith(EMB_FIRM_PREFIX)]    # 1024 firm-news dims
    emb_macro = [c for c in df.columns if c.startswith(EMB_MACRO_PREFIX)]  # 1024 macro-news dims
    declared = sum(TABULAR_GROUPS.values(), [])                  # every tabular feature we named
    actual = [c for c in df.columns                              # every tabular feature present
              if c not in ["ticker", "week_start", "target_log_return"] + emb_firm + emb_macro]
    missing, extra = sorted(set(actual) - set(declared)), sorted(set(declared) - set(actual))
    assert not missing and not extra, f"group mismatch; unassigned={missing} unknown={extra}"
    return df, emb_firm, emb_macro


def freeze_groups(path=None):
    """Write the group assignment to disk so the split used in the paper is auditable."""
    path = Path(path or C.OUT_TABLES / "FEATURE_GROUPS.json")    # default location
    path.parent.mkdir(parents=True, exist_ok=True)               # ensure directory
    payload = {k: v for k, v in TABULAR_GROUPS.items()}          # tabular groups by name
    payload["EMB_FIRM"] = f"{EMB_FIRM_PREFIX}0..1023"            # embedding blocks by prefix
    payload["EMB_MACRO"] = f"{EMB_MACRO_PREFIX}0..1023"
    payload["_counts"] = {k: len(v) for k, v in TABULAR_GROUPS.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


# ==========================================================================
# Folds
# ==========================================================================
def build_folds(weeks):
    """Return (inner, outer) fold lists; the fold straddling the boundary is discarded."""
    n = len(weeks); dev_end = int(C.DEV_FRACTION * n)            # tuning / out-of-sample boundary
    inner, outer, t = [], [], C.TAU0                             # fold lists, first training end
    while t + C.H <= n:                                          # while a test block still fits
        tr, te = weeks[:t], weeks[t:t + C.H]                     # expanding train, fixed test
        if t + C.H <= dev_end:                                   # test block entirely before boundary
            inner.append((tr, te))                               # -> inner (tuning) fold
        elif t >= dev_end:                                       # test block begins after the boundary
            outer.append((tr, te))                               # -> outer (evaluation) fold
        # a fold whose test block crosses the boundary is discarded
        t += C.STEP                                              # next fold
    return inner, outer


def fold_calendar(weeks):
    """Dated calendar of both fold families, for the appendix table that proves the chronology."""
    inner, outer = build_folds(weeks)                            # both families
    rows = []                                                    # one row per fold
    for kind, folds in [("inner", inner), ("outer", outer)]:     # label each family
        for i, (tr, te) in enumerate(folds, 1):                  # 1-based fold number
            rows.append(dict(family=kind, fold=i, n_train_weeks=len(tr), n_test_weeks=len(te),
                             train_start=pd.Timestamp(tr[0]).date(),
                             train_end=pd.Timestamp(tr[-1]).date(),
                             test_start=pd.Timestamp(te[0]).date(),
                             test_end=pd.Timestamp(te[-1]).date()))
    return pd.DataFrame(rows)


def rows_for(df, weeks):
    """Row indices for the given weeks that carry a valid next-week label."""
    return df.index[df["week_start"].isin(set(weeks)) & df["target_log_return"].notna()]


# ==========================================================================
# Feature construction
# ==========================================================================
def _reduce_stream(df, tr_idx, te_idx, emb_cols, prefix, q=None):
    """PCA for one embedding stream, fitted on this fold's news-bearing training rows only."""
    q = q or C.Q_PCA                                             # components per stream
    out_cols = [f"{prefix}_{i}" for i in range(q)]               # output column names
    idx = tr_idx.union(te_idx)                                   # rows in this fold
    red = pd.DataFrame(0.0, index=idx, columns=out_cols)         # news-free rows stay at zero
    has = df[emb_cols].abs().sum(axis=1) > 0                     # rows carrying a news vector
    fit_idx = tr_idx[has.loc[tr_idx].values]                     # fit set: training + news
    if len(fit_idx) < q + 5:                                     # not enough rows to fit
        return red
    pca = PCA(n_components=q, random_state=C.SEED)               # fold-local PCA
    pca.fit(df.loc[fit_idx, emb_cols].values.astype("float32"))  # fitted on training rows only
    app = idx[has.loc[idx].values]                               # transform train + test, news rows
    if len(app):
        red.loc[app, out_cols] = pca.transform(
            df.loc[app, emb_cols].values.astype("float32")).astype(float)
    return red


def fold_matrices(df, tr_idx, te_idx, emb_firm, emb_macro, groups):
    """Build the (train, test) design matrices for one fold given the groups in use.

    The PCA blocks are computed once here and reused across the whole hyperparameter grid,
    which is what keeps the tuning loop cheap.
    """
    parts_tr, parts_te, names = [], [], []                       # column blocks and their names
    tab = [c for g in groups if g in TABULAR_GROUPS for c in TABULAR_GROUPS[g]]   # tabular columns
    if tab:
        parts_tr.append(df.loc[tr_idx, tab]); parts_te.append(df.loc[te_idx, tab]); names += tab
    for g, cols, pre in [("EMB_FIRM", emb_firm, "red_firm"),     # firm-news embedding stream
                         ("EMB_MACRO", emb_macro, "red_macro")]: # macro-news embedding stream
        if g in groups:                                          # only reduce streams actually used
            red = _reduce_stream(df, tr_idx, te_idx, cols, pre)  # fold-local PCA
            parts_tr.append(red.loc[tr_idx]); parts_te.append(red.loc[te_idx])
            names += list(red.columns)
    X_tr = pd.concat(parts_tr, axis=1).values.astype(float)      # training design matrix
    X_te = pd.concat(parts_te, axis=1).values.astype(float)      # test design matrix
    clean = lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)   # same cleaning as paper
    return clean(X_tr), clean(X_te), names


def labels(df, idx):
    """Binary direction label: 1 when next week's log return is positive."""
    return (df.loc[idx, "target_log_return"] > 0).astype(int).values


# ==========================================================================
# Model
# ==========================================================================
def fit_predict(X_tr, y_tr, X_te, c, penalty, seed=None):
    """Scale on training rows, fit the classifier, return (y_pred, proba, coef)."""
    sc = StandardScaler().fit(X_tr)                              # scaler sees training rows only
    m = LogisticRegression(C=c, penalty=penalty, solver="liblinear",
                           class_weight="balanced", max_iter=2000,
                           random_state=seed or C.SEED).fit(sc.transform(X_tr), y_tr)
    Xs = sc.transform(X_te)                                      # scale the test block
    return m.predict(Xs), m.predict_proba(Xs)[:, 1], m.coef_[0]


def metrics(y_true, y_pred, proba=None):
    """The five headline classification metrics."""
    out = dict(accuracy=accuracy_score(y_true, y_pred),
               f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
               recall_down=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
               recall_up=recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    out["roc_auc"] = roc_auc_score(y_true, proba) if proba is not None else np.nan
    return out




def evaluate_outer(df, outer_folds, emb_firm, emb_macro, groups, c, penalty, seed=None):
    """Fit per outer fold and return (pooled metrics, per-fold metrics, prediction table, coefs)."""
    preds, per_fold, coefs, names = [], [], [], None             # accumulators
    for j, (tr_w, te_w) in enumerate(outer_folds):               # each held-out fold
        tr_i, te_i = rows_for(df, tr_w), rows_for(df, te_w)      # row indices
        X_tr, X_te, names = fold_matrices(df, tr_i, te_i, emb_firm, emb_macro, groups)
        y_tr, y_te = labels(df, tr_i), labels(df, te_i)          # labels
        yp, pr, cf = fit_predict(X_tr, y_tr, X_te, c, penalty, seed)   # fold-local fit
        fr = df.loc[te_i, ["ticker", "week_start", "target_log_return", "log_return"]].copy()
        fr = fr.rename(columns={"target_log_return": "y_true_ret", "log_return": "ret_t"})
        fr["y_true"], fr["y_pred"], fr["proba"], fr["fold"] = y_te, yp, pr, j
        preds.append(fr); coefs.append(pd.Series(cf, index=names))
        per_fold.append(dict(fold=j, **metrics(y_te, yp, pr)))   # fold-level metrics
    pred_table = pd.concat(preds, ignore_index=True)             # pooled predictions
    pooled = metrics(pred_table.y_true, pred_table.y_pred, pred_table.proba)
    return pooled, pd.DataFrame(per_fold), pred_table, pd.concat(coefs, axis=1)


# ==========================================================================
# Economic backtest (Section 3.9 conventions)
# ==========================================================================
def backtest(pred_table, cost_bps=None, borrow_fee_pa=0.0):
    """Weekly equal-weighted portfolio returns for the four strategies."""
    cost = (C.COST_BPS if cost_bps is None else cost_bps) / 10000.0   # one-way cost
    p = pred_table.sort_values(["ticker", "week_start"]).copy()       # order for turnover
    p["pos_lo"] = (p.y_pred > 0).astype(float)                        # long-only position
    p["pos_ls"] = np.where(p.y_pred > 0, 1.0, -1.0)                   # long-short position
    g = p.groupby("ticker")                                            # turnover is per ticker
    t_lo = g.pos_lo.diff().abs().fillna(p.pos_lo.abs())                # realised LO turnover
    t_ls = g.pos_ls.diff().abs().fillna(p.pos_ls.abs())                # realised LS turnover
    borrow = (borrow_fee_pa / C.PERIODS) * (p.pos_ls < 0)              # weekly borrow cost on shorts
    p["lo_gross"] = p.pos_lo * p.y_true_ret                            # before costs
    p["ls_gross"] = p.pos_ls * p.y_true_ret
    p["lo_net"] = p.lo_gross - cost * t_lo                             # after costs
    p["ls_net"] = p.ls_gross - cost * t_ls - borrow
    p["buyhold"] = p.y_true_ret                                        # always long
    p["momentum"] = np.where(p.ret_t > 0, p.y_true_ret, 0.0)           # long after an up week
    cols = ["lo_gross", "lo_net", "ls_gross", "ls_net", "buyhold", "momentum"]
    return p.groupby("week_start")[cols].mean().sort_index()           # equal-weighted weekly series


def ann_sharpe(r):
    """Annualised Sharpe: sqrt(52) * mean / sd, population sd, no risk-free rate."""
    r = np.asarray(r, float)                                     # to array
    s = r.std(ddof=0)                                            # population sd, as in the paper
    return float(np.sqrt(C.PERIODS) * r.mean() / s) if s > 0 else 0.0


def cum_return(r):
    """Cumulative simple return implied by the weekly log returns."""
    return float(np.expm1(np.asarray(r, float).sum()))


def max_drawdown(r):
    """Largest peak-to-trough decline of the wealth path."""
    w = np.exp(np.cumsum(np.asarray(r, float)))                  # wealth path
    return float((w / np.maximum.accumulate(w) - 1.0).min())     # worst drawdown


# ==========================================================================
# Moving-block bootstrap over weeks
# ==========================================================================
def week_block_indices(n_weeks, block_len=None, n_boot=None, seed=None):
    """Moving-block bootstrap index matrix over WEEKS, shape (n_boot, n_weeks).

    Each row is a resample of week positions built from overlapping blocks of consecutive weeks
    (Kunsch 1989). Resampling whole weeks keeps the 8 stocks of a week together, which preserves
    cross-sectional dependence; using blocks preserves serial dependence up to the block length.
    """
    L = block_len or C.BLOCK_LEN; B = n_boot or C.N_BOOT          # block length, draws
    rng = np.random.default_rng(seed or C.BOOT_SEED)              # fixed stream
    n_blocks = int(np.ceil(n_weeks / L))                          # blocks per resample
    starts = rng.integers(0, n_weeks - L + 1, size=(B, n_blocks)) # block start positions
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)
    return idx[:, :n_weeks]                                       # trim to original length


def block_bootstrap(series, stat_fn, block_len=None, n_boot=None, seed=None):
    """Moving-block bootstrap of a statistic of a WEEK-indexed series or (weeks x k) array."""
    arr = np.asarray(series.values if hasattr(series, "values") else series)
    idx = week_block_indices(len(arr), block_len, n_boot, seed)   # resampled week positions
    return np.array([stat_fn(arr[i]) for i in idx])               # statistic per draw


def boot_ci(draws, alpha=0.05):
    """Percentile confidence interval from bootstrap draws."""
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def boot_p(draws, est, null):
    """Null-imposed one-sided bootstrap p-value for H0: theta <= null (Hall & Wilson, 1991)."""
    return float(np.mean(np.asarray(draws) - est >= est - null))
