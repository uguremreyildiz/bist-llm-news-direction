"""
reproduce.py -- Reproduces the results of the selected model from scratch.
=========================================================================
Selected (validation-chosen) configuration: M3_Full / BGE-M3 / PCA(32 per stream) /
L1-penalised logistic regression (C = 0.02005544632966435).

This script depends only on data/panel_bge_m3.parquet and the standard scientific stack.
It contains no news text and imports nothing from the authors' private codebase.

The leakage control is preserved here: within every evaluation fold, PCA and StandardScaler
are fitted on that fold's training rows ONLY.

Run:  python reproduce.py
"""
from pathlib import Path                                       # filesystem paths
import json                                                    # summary output
import numpy as np                                             # numerics
import pandas as pd                                            # tabular data
from scipy.stats import binomtest                              # test against a coin flip
from sklearn.decomposition import PCA                          # dimensionality reduction
from sklearn.preprocessing import StandardScaler               # feature scaling
from sklearn.linear_model import LogisticRegression            # classifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# -- Constants: identical to the protocol described in the paper -----------
SEED = 42                                                      # reproducibility
Q = 32                                                         # PCA components per embedding stream
TAU0, H, STEP = 156, 12, 12                                    # expanding window: initial train, test, step
DEV_FRACTION = 0.60                                            # first 60% is the tuning region
BEST_PARAMS = dict(C=0.02005544632966435, penalty="l1")        # hyperparameters chosen on inner folds
COST = 10 / 10000.0                                            # one-way transaction cost (10 bps)
PERIODS = 52                                                   # weekly annualisation factor

ROOT = Path(__file__).resolve().parent                         # repository root
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)               # output directory


def load_panel():
    """Load the shared panel and split its columns into blocks."""
    panel = pd.read_parquet(ROOT / "data" / "panel_bge_m3.parquet")     # the single data file
    panel["week_start"] = pd.to_datetime(panel["week_start"])            # ensure datetime dtype
    panel = panel.sort_values(["ticker", "week_start"]).reset_index(drop=True)   # deterministic order
    vec = [c for c in panel.columns if c.startswith("vec_bge_m3_")]      # stock-news embedding (1024)
    mvec = [c for c in panel.columns if c.startswith("macro_vec_")]      # macro-news embedding (1024)
    ids = ["ticker", "week_start", "target_log_return"]                  # identifiers and label source
    feats = [c for c in panel.columns if c not in ids + vec + mvec]      # 66 tabular features
    return panel, feats, vec, mvec


def expanding_eval_folds(weeks):
    """Expanding-window folds; returns only the out-of-sample folds used for evaluation."""
    n = len(weeks); dev_end = int(DEV_FRACTION * n)             # boundary between tuning and OOS
    folds, t = [], TAU0                                         # fold list, end of first training block
    while t + H <= n:                                           # while a test block still fits
        if t >= dev_end:                                        # keep only folds starting after the boundary
            folds.append((weeks[:t], weeks[t:t + H]))           # (expanding train, 12-week test)
        t += STEP                                               # advance to the next fold
    return folds                                                # 12 out-of-sample folds


def fold_reduce(panel, tr_idx, te_idx, emb_cols, prefix):
    """Fit PCA on this fold's news-bearing TRAINING rows only; news-free rows stay at zero."""
    out_cols = [f"{prefix}_{i}" for i in range(Q)]              # red_stock_* / red_macro_*
    idx = tr_idx.union(te_idx)                                  # all rows in this fold
    red = pd.DataFrame(0.0, index=idx, columns=out_cols)        # default: zero (no news that week)
    has = panel[emb_cols].abs().sum(axis=1) > 0                 # rows that carry a news vector
    fit_idx = tr_idx[has.loc[tr_idx].values]                    # fit set: training rows with news
    if len(fit_idx) < Q + 5:                                    # too few samples to fit
        return red                                              # leave the block at zero
    pca = PCA(n_components=Q, random_state=SEED)                # fold-local PCA
    pca.fit(panel.loc[fit_idx, emb_cols].values.astype("float32"))       # fitted on training rows only
    app_idx = idx[has.loc[idx].values]                          # transform: train + test, news-bearing
    if len(app_idx):                                            # if any qualify
        red.loc[app_idx, out_cols] = pca.transform(             # reduced representation
            panel.loc[app_idx, emb_cols].values.astype("float32")).astype(float)
    return red                                                  # this fold's reduced features


def run():
    panel, feats, vec, mvec = load_panel()                      # data and column blocks
    weeks = sorted(panel["week_start"].unique())                # 393 weekly timestamps
    folds = expanding_eval_folds(weeks)                         # 12 out-of-sample folds
    print(f"Panel: {len(panel):,} rows x {panel.shape[1]:,} columns | "
          f"{len(weeks)} weeks | {len(folds)} out-of-sample folds\n")

    preds, coefs = [], []                                       # predictions and per-fold coefficients
    for j, (trw, tew) in enumerate(folds):                      # each out-of-sample fold
        m_tr = panel["week_start"].isin(trw) & panel["target_log_return"].notna()   # training mask
        m_te = panel["week_start"].isin(tew) & panel["target_log_return"].notna()   # test mask
        tr_idx, te_idx = panel.index[m_tr], panel.index[m_te]   # row indices

        rs = fold_reduce(panel, tr_idx, te_idx, vec, "red_stock")        # stock news -> 32 components
        rm = fold_reduce(panel, tr_idx, te_idx, mvec, "red_macro")       # macro news -> 32 components
        red = pd.concat([rs, rm], axis=1)                                # 64 reduced features

        X_tr = pd.concat([panel.loc[tr_idx, feats], red.loc[tr_idx]], axis=1)   # training matrix
        X_te = pd.concat([panel.loc[te_idx, feats], red.loc[te_idx]], axis=1)   # test matrix
        names = list(X_tr.columns)                                              # feature names
        X_tr = np.nan_to_num(X_tr.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        y_tr = (panel.loc[tr_idx, "target_log_return"] > 0).astype(int).values   # training label
        y_te = (panel.loc[te_idx, "target_log_return"] > 0).astype(int).values   # test label

        sc = StandardScaler().fit(X_tr)                          # scaler fitted on training rows only
        model = LogisticRegression(**BEST_PARAMS, solver="liblinear",    # the selected classifier
                                   class_weight="balanced", max_iter=2000,
                                   random_state=SEED).fit(sc.transform(X_tr), y_tr)
        Xs_te = sc.transform(X_te)                               # apply the same scaler to the test rows
        fr = panel.loc[te_idx, ["ticker", "week_start", "target_log_return", "log_return"]].copy()
        fr = fr.rename(columns={"target_log_return": "y_true_ret", "log_return": "ret_t"})
        fr["y_true"] = y_te                                      # realised direction
        fr["y_pred"] = model.predict(Xs_te)                      # predicted direction
        fr["proba"] = model.predict_proba(Xs_te)[:, 1]           # P(up)
        fr["fold"] = j                                           # fold identifier
        preds.append(fr); coefs.append(pd.Series(model.coef_[0], index=names))

    preds = pd.concat(preds, ignore_index=True)                  # all out-of-sample predictions
    preds.to_csv(OUT / "oos_predictions.csv", index=False)       # save

    # -- 1) Out-of-sample classification metrics --
    correct = int((preds.y_true == preds.y_pred).sum()); n = len(preds)      # hits and total
    res = dict(n_oos=n,
               accuracy=round(accuracy_score(preds.y_true, preds.y_pred), 4),
               f1_macro=round(f1_score(preds.y_true, preds.y_pred, average="macro"), 4),
               roc_auc=round(roc_auc_score(preds.y_true, preds.proba), 4),
               p_vs_random=float(binomtest(correct, n, 0.5, alternative="greater").pvalue))

    # -- 2) Coefficient importance aggregated by feature family --
    cf = pd.concat(coefs, axis=1).abs().mean(axis=1)             # mean |beta| across folds
    fam = pd.Series({f: cf[[c for c in cf.index if rule(c)]].sum() for f, rule in {
        "emb_reduced_macro": lambda c: c.startswith("red_macro_"),
        "emb_reduced_stock": lambda c: c.startswith("red_stock_"),
        "macro_price": lambda c: any(c.startswith(p) for p in
                                     ("USDTRY", "EURTRY", "BRENT", "GOLD", "VIX")),
        "stock_price": lambda c: c in ("log_return", "ret_lag1", "ret_lag2", "ret_lag3",
                                       "ret_lag5", "ret_ma3", "ret_ma5", "ret_std5", "ret_momentum"),
        "emb_scalar": lambda c: c in ("stock_emb_norm", "macro_emb_norm", "stock_emb_norm_z",
                                      "stock_macro_alignment", "stock_cos_lag1", "stock_cos_lag3",
                                      "macro_cos_lag1", "macro_cos_lag3"),
    }.items()})
    fam["sentrel_other"] = cf.sum() - fam.sum()                  # remainder: sentiment/category scalars
    fam_pct = (100 * fam / cf.sum()).round(1)                    # percentage share
    fam_pct.to_csv(OUT / "coefficient_family_shares.csv", header=["share_pct"])

    # -- 3) Economic backtest --
    p = preds.sort_values(["ticker", "week_start"]).copy()       # ordered for turnover accounting
    p["pos_lo"] = (p.y_pred > 0).astype(float)                   # long-only position
    p["pos_ls"] = np.where(p.y_pred > 0, 1.0, -1.0)              # long-short position
    g = p.groupby("ticker")                                       # turnover is computed per ticker
    p["lo_net"] = p.pos_lo * p.y_true_ret - COST * g.pos_lo.diff().abs().fillna(p.pos_lo.abs())
    p["ls_net"] = p.pos_ls * p.y_true_ret - COST * g.pos_ls.diff().abs().fillna(p.pos_ls.abs())
    p["buyhold"] = p.y_true_ret                                   # always long
    p["momentum"] = np.where(p.ret_t > 0, p.y_true_ret, 0.0)      # long after an up week, else flat
    wk = p.groupby("week_start")[["lo_net", "ls_net", "buyhold", "momentum"]].mean().sort_index()
    # Annualised Sharpe ratio; NO risk-free rate is subtracted, population sd (ddof = 0)
    sharpe = {s: round(float(np.sqrt(PERIODS) * wk[s].mean() / wk[s].std(ddof=0)), 4)
              for s in wk.columns}
    pd.DataFrame([dict(strategy=s, sharpe=v,
                       cum_return=round(float(np.expm1(wk[s].sum())), 4)) for s, v in sharpe.items()]
                 ).to_csv(OUT / "sharpe.csv", index=False)

    # -- 4) Per-ticker breakdown and confusion matrix --
    pt = preds.groupby("ticker").apply(lambda x: pd.Series({
        "n": len(x), "accuracy": round(accuracy_score(x.y_true, x.y_pred), 4),
        "always_long_acc": round(float((x.y_true > 0).mean()), 4)}), include_groups=False).reset_index()
    pt.to_csv(OUT / "per_ticker.csv", index=False)
    cm = confusion_matrix(preds.y_true, preds.y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["true_DOWN", "true_UP"], columns=["pred_DOWN", "pred_UP"]).to_csv(
        OUT / "confusion_matrix.csv")

    # -- 5) Report --
    summary = dict(res, **{f"sharpe_{s}": v for s, v in sharpe.items()})     # combined summary
    print("Out-of-sample classification")
    print(f"  observations        {res['n_oos']}")
    print(f"  accuracy            {res['accuracy']:.4f}")
    print(f"  macro-F1            {res['f1_macro']:.4f}")
    print(f"  ROC-AUC             {res['roc_auc']:.4f}")
    print(f"  binomial p (1-sided){res['p_vs_random']:>10.6f}")
    print("\nAnnualised Sharpe ratio")
    for s, lab in [("lo_net", "model long-only (net) "), ("buyhold", "buy-and-hold          "),
                   ("ls_net", "model long-short (net)"), ("momentum", "price momentum        ")]:
        print(f"  {lab}  {sharpe[s]:.4f}")
    print("\nCoefficient share by feature family (%)")
    for k, v in fam_pct.sort_values(ascending=False).items():
        print(f"  {k:<20}{v:>6.1f}")
    json.dump(summary, open(OUT / "reproduced_summary.json", "w"), indent=2)
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    run()
