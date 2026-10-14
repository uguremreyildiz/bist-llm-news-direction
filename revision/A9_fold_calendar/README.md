# A9 — Fold calendar and selection chronology

Dated chronology of the expanding-window design and of model selection.

**Input:** `data/panel_bge_m3.parquet` only. **Runtime:** ~3 s.

**Outputs:**
- `outputs/tables/TA1_fold_calendar.csv` — every inner and outer fold with dated training and
  test windows and row counts (appendix table).
- `outputs/figures/F_A9_timeline.png` / `.pdf` — Gantt-style timeline of the 18 folds.
- `outputs/tables/A9_summary.json` — the numbers quoted in the text.

## What it establishes

| Quantity | Value |
|---|---|
| Weekly timestamps in the panel | 393 (2018-01-01 … 2025-08-18) |
| Admissible expanding-window folds | 19 |
| Inner (tuning) folds | 6 |
| Outer (evaluation) folds | 12 |
| Discarded straddling folds | 1 |
| Tuning / out-of-sample boundary | week of 2022-07-25 (index 235 = ⌊0.60 × 393⌋) |
| Last inner **test** week | 2022-05-23 |
| First outer **test** week | 2022-08-29 |
| Buffer between the two families | **12 panel weeks** (2022-05-30 … 2022-08-22) |
| Last outer test week | 2025-06-16 |
| Outer test observations | 1,152 |

The fold whose test block crosses the boundary is dropped, which leaves a 12-week buffer
(2022-05-30 … 2022-08-22) that is a test block for neither family; every outer fold still uses
these weeks for training. The last week used for tuning therefore precedes the first evaluated
week by a full test horizon. In calendar terms the two dates are 14 weeks apart, i.e. 13 intermediate
calendar weeks; the panel omits one of them (the holiday-shortened week of 11 July 2022), so the
count in panel weeks is 12.

## Test-block coverage (why the outer blocks span more weeks than the inner blocks)

The 60/40 split places the *boundary*, not the test coverage. The first 156 weeks are consumed
entirely by the initial training window and are never a test block for anyone, so the tuning
region yields far fewer test weeks than the out-of-sample region:

| Segment | Weeks | Role |
|---|---|---|
| 2018-01-01 … 2020-12-21 | 156 | initial training window, never tested |
| 2020-12-28 … 2022-05-23 | 72 | inner test blocks (6 x 12) |
| 2022-05-30 … 2022-08-22 | 12 | test block of the discarded straddling fold (buffer) |
| 2022-08-29 … 2025-06-16 | 144 | outer test blocks (12 x 12) |
| 2025-06-23 … 2025-08-18 | 9 | tail too short for another 12-week block, unused |
| **Total** | **393** | |

## Missing weeks

The panel contains 393 weeks where a complete Monday grid over the same span would contain 399.
The six absent weeks are 2021-07-19, 2022-05-02, 2022-07-11, 2023-06-26, 2024-04-08 and
2024-06-17. Borsa İstanbul DID trade in each of them, but only for one or two sessions
(holiday-shortened weeks). The weekly price builder excludes any week with fewer than three trading
sessions (paper, Section 4.5). Consequences, verified on AKBNK:

* the week is not an observation and has no features or label;
* for the week that precedes it, the "next-week" label is the return of the next RETAINED week
  (e.g. the row for 2021-07-12 is labelled with the return of the week of 2021-07-26);
* that next retained week's return is measured from the close of the short week's last session
  (e.g. log(close 30 Jul / close 19 Jul 2021)), so the one or two short-week sessions enter no
  return. This is a horizon irregularity affecting 6 of 393 weeks.

The script asserts `max(inner test_end) < min(outer test_start)` and fails if it does not hold,
so the chronology claim is machine-checked rather than asserted.

## Note on the end of the sample

The panel runs to 2025-08-18, but the last outer test block ends on 2025-06-16: the remaining
9 weeks cannot fill another 12-week block and are therefore never evaluated. The out-of-sample
calendar window is **29 August 2022 – 16 June 2025**, not "September 2022 – August 2025".
