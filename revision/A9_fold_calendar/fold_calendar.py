"""
A9 -- Fold calendar: the dated chronology of model selection.
=============================================================
Documents when model selection happened relative to the evaluated weeks. It prints every inner
and outer fold with its training and test window, and verifies mechanically that the last inner test week
ends strictly before the first outer test week begins. Selection therefore used only weeks that
precede every evaluated week.

Outputs: TA1_fold_calendar.csv, F_A9_timeline.png/pdf, A9_summary.json
Runtime: a few seconds.
"""
from pathlib import Path                                        # paths
import sys, json                                                # json summary, import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # make config/common importable
import pandas as pd                                             # tables
import matplotlib                                               # plotting
matplotlib.use("Agg")                                           # headless backend
import matplotlib.pyplot as plt                                 # pyplot
import matplotlib.dates as mdates                               # date axis
import config as C                                              # constants and paths
from common import pipeline as P                                # shared machinery


def main():
    C.ensure_dirs()                                             # output directories
    df, _, _ = P.load_panel()                                   # the shared panel
    weeks = sorted(df["week_start"].unique())                   # 393 weekly timestamps
    inner, outer = P.build_folds(weeks)                         # 6 inner, 12 outer
    cal = P.fold_calendar(weeks)                                # dated calendar

    # Row counts per fold, so the table also documents the sample behind each estimate
    counts = []                                                 # training/test row counts
    for fam, folds in [("inner", inner), ("outer", outer)]:     # both families
        for i, (tr, te) in enumerate(folds, 1):                 # 1-based fold number
            counts.append(dict(family=fam, fold=i,
                               n_train_rows=len(P.rows_for(df, tr)),
                               n_test_rows=len(P.rows_for(df, te))))
    cal = cal.merge(pd.DataFrame(counts), on=["family", "fold"])      # join counts
    cal.to_csv(C.OUT_TABLES / "TA1_fold_calendar.csv", index=False)   # appendix table

    # The mechanical check that makes the chronology claim verifiable
    last_inner_test = pd.Timestamp(cal.loc[cal.family == "inner", "test_end"].max())
    last_inner_train = pd.Timestamp(cal.loc[cal.family == "inner", "train_end"].max())
    first_outer_test = pd.Timestamp(cal.loc[cal.family == "outer", "test_start"].min())
    first_outer_train_end = pd.Timestamp(cal.loc[cal.family == "outer", "train_end"].min())
    assert last_inner_test < first_outer_test, "inner folds are not strictly earlier"

    # The buffer: panel weeks that are a test block for neither family (the discarded fold's test block).
    # Counted in panel weeks, not calendar weeks -- the panel omits six holiday-shortened weeks
    # (fewer than three sessions), so a calendar-difference count would overstate the gap.
    wk = pd.Series(pd.to_datetime(weeks))                             # panel week index
    i_last_inner = int(wk[wk == last_inner_test].index[0])            # position of last inner test
    i_first_outer = int(wk[wk == first_outer_test].index[0])          # position of first outer test
    buffer_weeks = i_first_outer - i_last_inner - 1                    # untested weeks in between
    buffer_span = (str(wk.iloc[i_last_inner + 1].date()), str(wk.iloc[i_first_outer - 1].date()))
    cal_weeks = round((first_outer_test - last_inner_test).days / 7)  # calendar-week difference

    # The discarded straddling fold: report it explicitly rather than leaving a silent gap
    n = len(weeks); dev_end = int(C.DEV_FRACTION * n)                 # boundary index
    candidates = len(range(C.TAU0, n - C.H + 1, C.STEP))              # all admissible folds
    discarded = candidates - len(inner) - len(outer)                  # straddling folds

    summary = dict(
        n_weeks=n, boundary_index=dev_end,
        boundary_week=str(pd.Timestamp(weeks[dev_end]).date()),
        candidate_folds=candidates, inner_folds=len(inner), outer_folds=len(outer),
        discarded_straddling_folds=discarded,
        inner_train_start=str(pd.Timestamp(cal.loc[cal.family == "inner", "train_start"].min())
                              .date()),
        last_inner_train_end=str(last_inner_train.date()),
        last_inner_test_end=str(last_inner_test.date()),
        first_outer_train_end=str(first_outer_train_end.date()),
        first_outer_test_start=str(first_outer_test.date()),
        last_outer_test_end=str(pd.Timestamp(cal.loc[cal.family == "outer", "test_end"].max())
                                .date()),
        buffer_panel_weeks=buffer_weeks, buffer_span=buffer_span,
        buffer_calendar_weeks=cal_weeks,
        panel_weeks_missing_vs_monday_grid=int(
            len(pd.date_range(wk.iloc[0], wk.iloc[-1], freq="W-MON")) - len(wk)),
        test_coverage=dict(
            never_tested_initial_window=f"{str(wk.iloc[0].date())}..{str(wk.iloc[C.TAU0-1].date())}"
                                        f" ({C.TAU0} weeks)",
            inner_test_block_weeks=len(inner) * C.H, outer_test_block_weeks=len(outer) * C.H,
            unused_tail_weeks=len(wk) - (C.TAU0 + (len(inner) + 1 + len(outer)) * C.H)),
        total_outer_test_rows=int(cal.loc[cal.family == "outer", "n_test_rows"].sum()),
        selection_scheme=("configuration and hyperparameters chosen once on the 6 inner folds, "
                          "then frozen; the 12 outer folds were never used for any selection "
                          "decision"))
    (C.OUT_TABLES / "A9_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure: a Gantt-style timeline that makes the separation visible at a glance
    fig, ax = plt.subplots(figsize=(7.0, 4.2))                       # single panel
    for _, r in cal.iterrows():                                      # one bar pair per fold
        y = (r.fold if r.family == "inner" else r.fold + 7)          # inner on top, outer below
        tr0, tr1 = pd.Timestamp(r.train_start), pd.Timestamp(r.train_end)
        te0, te1 = pd.Timestamp(r.test_start), pd.Timestamp(r.test_end)
        ax.barh(y, tr1 - tr0, left=tr0, height=0.6, color="#CCCCCC", edgecolor="none")
        ax.barh(y, te1 - te0, left=te0, height=0.6,
                color="#0072B2" if r.family == "inner" else "#D55E00", edgecolor="none")
    ax.axvline(pd.Timestamp(weeks[dev_end]), color="0.3", ls="--", lw=1.0)   # boundary
    ax.annotate("tuning / out-of-sample boundary", xy=(pd.Timestamp(weeks[dev_end]), 20.2),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8, color="0.3")
    ax.set_yticks(list(range(1, 7)) + list(range(8, 20)))            # fold labels
    ax.set_yticklabels([f"inner {i}" for i in range(1, 7)] + [f"outer {i}" for i in range(1, 13)],
                       fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("Week")                          # chronological top-to-bottom
    ax.xaxis.set_major_locator(mdates.YearLocator())                  # yearly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["top", "right"]].set_visible(False)                    # clean frame
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#CCCCCC", "#0072B2", "#D55E00")]
    ax.legend(handles, ["training window", "inner test block (tuning)",
                        "outer test block (evaluation)"], frameon=False, fontsize=8,
              loc="lower right")
    for ext, dpi in [("png", 300), ("pdf", None)]:                    # Word + vector
        fig.savefig(C.OUT_FIGURES / f"F_A9_timeline.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(cal.to_string(index=False))                                 # the appendix table
    print("\n--- A9 summary ---")
    for k, v in summary.items():
        print(f"{k:<32}{v}")
    return summary


if __name__ == "__main__":
    main()
