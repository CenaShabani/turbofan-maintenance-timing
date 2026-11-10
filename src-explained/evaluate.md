# `src/evaluate.py`

## What this module is for

Two jobs. It scores the model — discrimination, calibration, accuracy of the probability —
and it measures what ignoring censoring costs, which is the project's headline finding.

Called by `run_analysis.py` and the capstone. Consumes `panel.py`'s fleet table and
`survival.py`'s fitted models.

## The public surface

| Function | In | Out |
|---|---|---|
| `landmark_scores(result, risk_set, features, edges, fit_risk_set, landmarks, horizon)` | | per-landmark Harrell's C and Uno's C |
| `time_dependent_auc(..., landmark, n_points=8)` | | AUC vs cycles ahead |
| `brier(..., landmark, horizons)` | | dict: IBS, Kaplan-Meier baseline IBS, improvement |
| `calibration(result, risk_set, features, edges, landmark, horizon, n_bins=10)` | | predicted vs Kaplan-Meier observed, by decile |
| `censoring_bias(fleet)` | fleet table | dict: truth / complete-case / naive / KM errors |
| `bias_by_censoring_rate(trajectories, rates)` | | the sweep behind figure 03 |
| `compare_feature_sets(fit, holdout, drift_cols, level_cols, edges)` | | the drift-vs-level test |
| `bootstrap_metric(values, groups, statistic, n_boot)` | | `(mean, lo, hi)` |

## The walkthrough

**`landmark_scores`** builds the landmark population, computes horizon risk, and reports both
concordance indices. Uno's C needs a censoring distribution estimated on the **fit** data —
using the evaluation set's own censoring pattern to weight the evaluation set is circular.
It also needs a `tau` inside the range both sets cover, computed as the min of the two
maximum event times.

**`time_dependent_auc`** evaluates on a grid of quantiles of observed event times. One AUC
number hides the thing a planner cares about most: models are usually good at spotting
imminent failure and much worse at spotting it early, and early is when action is cheap.

**`brier`** computes the integrated Brier score over a horizon grid, and — the part that
makes it meaningful — the same score for a model that predicts the fleet-wide Kaplan-Meier
curve for everyone. That baseline uses no sensor information at all, so beating it is the
minimum evidence that the sensors are being read to any effect.

**`calibration`** bins by predicted risk and computes the observed rate **with
Kaplan-Meier inside each bin**.

**`censoring_bias`** produces the three-way comparison plus the Kaplan-Meier median error.

**`bias_by_censoring_rate`** re-censors the same lifetimes at several target rates and
recomputes, producing figure 03.

**`compare_feature_sets`** fits drift-only, level-only and both, and reports in-sample
pseudo-R² against holdout concordance.

**`bootstrap_metric`** resamples groups, not rows.

## The non-obvious bits

**Kaplan-Meier inside each calibration bin, not a raw proportion.** A raw proportion would
count a unit censored 10 cycles into a 30-cycle horizon as a survivor. That understates the
observed rate and would make the model look systematically *over*-cautious — the exact error
the project exists to avoid making. Using a raw proportion here would have been an
embarrassing own goal, and it is the kind that is very easy to commit.

**`duplicates="drop"` on the qcut.** Predicted risks pile up near zero, so several decile
edges can be identical. Without it, `qcut` raises; with it, the bins collapse gracefully and
the returned frame simply has fewer rows.

**Both concordance indices, always.** The gap between them measures how much the censoring is
distorting the familiar metric. Here it is 1–2 points at 40% censoring — small, which is
itself the useful finding. Reporting only Harrell's would have been quietly self-serving in a
project about censoring.

**`compare_feature_sets` reports in-sample fit *and* holdout discrimination side by side**,
because the finding is precisely that they disagree: levels score pseudo-R² 0.59 vs drift's
0.40 while scoring Uno's C 0.83 vs 0.88. Reporting either alone would support the opposite
conclusion.

**`bootstrap_metric` resamples the unit.** Resampling rows would treat one engine's hundreds
of correlated cycle observations as independent draws and produce an interval several times
too narrow. Pinned by a test that uses a single group and asserts the interval collapses to a
point.

**The bare `except (ValueError, ZeroDivisionError)` around Uno's C.** scikit-survival raises
when the IPCW weights degenerate — a small landmark population with an awkward censoring
pattern. Returning NaN for that landmark and carrying on is better than aborting the run, and
the NaN is visible in the output table rather than hidden.

**Why `censoring_bias` compares *means* but reports the Kaplan-Meier *median*.** Kaplan-Meier
does not give a mean without integrating the curve, and that integral is undefined when the
curve does not reach zero — which it does not under heavy censoring. The median is the
quantity that survives censoring cleanly, so it is the one reported, compared against the
true median rather than the true mean.

## How it can break

- **A landmark with fewer than 5 events** is skipped in `landmark_scores`, silently by
  design — an index computed on three pairs is noise.
- **A landmark beyond every unit** raises from `survival.landmark_frame`; `landmark_scores`
  catches it and continues, the other functions do not.
- **Degenerate IPCW weights** produce NaN for Uno's C, visibly.
- **`brier` with all horizons beyond the last event** returns NaN rather than raising.

`tests/test_evaluate.py` (7 tests) pins the direction of the complete-case bias, that the
naive estimate is worse still, that Kaplan-Meier is nearly unbiased, that the bias grows with
the censoring rate, that the Kaplan-Meier series stays flatter than the complete-case series,
that the bootstrap resamples groups, and that calibration bins are complete and ordered.
