# `src/survival.py`

## What this module is for

Holds the four models — Kaplan-Meier, Cox with time-varying covariates, the discrete-time
logistic hazard, and a Random Survival Forest — plus the machinery that turns a fitted hazard
into the forward-looking risk the decision layer consumes.

Called by `run_analysis.py`, `evaluate.py` and `policy.py`. Consumes `panel.py`'s risk set.

## The public surface

| Function | In | Out |
|---|---|---|
| `kaplan_meier(fleet, group_column)` | | curves with CI bands per group |
| `logrank(fleet, group_column)` | | dict with statistic, p-value, df |
| `risk_window_start(risk_set)` | | `int` — earliest observed failure cycle |
| `restrict_to_risk_window(risk_set, start)` | | filtered risk set |
| `derive_band_edges(risk_set, n_bands=10)` | | `ndarray` of cut points |
| `assign_band(cycles, edges)` | | `ndarray[int]` |
| `build_design(risk_set, features, edges)` | | design matrix with constant |
| `fit_discrete_hazard(risk_set, features, edges)` | | statsmodels `BinaryResults` |
| `fit_complete_case(...)` | | same, fitted only on units that failed |
| `fit_cox_time_varying(risk_set, features, penalizer=0.1)` | | `CoxTimeVaryingFitter` |
| `schoenfeld_residuals(fitter, risk_set, features)` | | residuals + `.attrs` counts |
| `test_proportional_hazards(...)` | | per-covariate table with Holm-adjusted p |
| `predict_hazard(...)` | | per-cycle failure probability |
| `horizon_risk(result, risk_set, features, edges, horizon=30)` | | P(fail within horizon) |
| `landmark_frame(risk_set, _, landmark)` | | units at risk at that cycle |
| `fit_random_survival_forest(landmark_data, features)` | | `RandomSurvivalForest` |
| `tidy(result)` | | coefficients as odds ratios with CIs |

## The walkthrough

**`risk_window_start` + `restrict_to_risk_window` implement left truncation.** No engine in
this fleet fails before cycle 128, so those ~40,000 rows have a constant outcome and drive the
intercept to minus infinity. Restricting means every quantity the model produces is
conditional on having survived to that cycle, and 106 engines censored earlier drop out.

**`derive_band_edges` cuts at quantiles of observed failure cycles.** Equal-width bands leave
zero-event bands in the sparse tail; equal-event bands cannot. Derived from the fit set only.

**`build_design` assembles features + band dummies + constant**, dropping band 1 as reference.

**`fit_discrete_hazard`** is a `statsmodels.Logit` with `cov_type="cluster"` grouped by
`unit_id`.

**`fit_cox_time_varying`** uses lifelines' counting-process fitter with a ridge penalty.

**`schoenfeld_residuals`** computes, at each failure time, the failing unit's covariates minus
the risk-set's `exp(beta'x)`-weighted mean.

**`test_proportional_hazards`** rank-correlates those residuals with failure time, then
applies Holm-Bonferroni across the 17 covariates.

**`horizon_risk`** compounds `(1 - hazard)` forward over the horizon, advancing the cycle band
each step while holding the covariates fixed, and returns `1 - survival`.

**`landmark_frame`** takes the rows at exactly one cycle and redefines time relative to it.

## The non-obvious bits

**`BAND_REFERENCE = 1`, and why dropping it is not housekeeping.** After the risk-window
restriction, band 0 is empty. If a full set of dummies for bands 1..10 were included, they
would sum to 1 in every row and be **exactly collinear with the intercept**. The fit would
then return whatever the pseudo-inverse produced rather than an estimate. An earlier version
of this file had that bug: it appeared to converge, which is the worst possible symptom.
`tests/test_survival.py::test_design_matrix_drops_the_reference_band` asserts full rank.

**The naive PH test is wrong, and this file used to use it.** Correlating raw covariate values
at failures with failure time reported **16 of 17** violations — because the drift features
rise with age *by construction*, so the test measured the feature definition rather than the
model fit. Subtracting the model's expectation gives **10 of 17** with a non-uniform pattern.
The docstring keeps the explanation because the shortcut is genuinely tempting.

**`linear[at_risk] - linear[at_risk].max()` before `np.exp`.** Standard log-sum-exp
stabilisation. Without it a large linear predictor overflows and the weights come back as NaN,
silently corrupting the residual rather than raising.

**Singleton risk sets are skipped and counted.** When only one unit is at risk, it is
necessarily the one that failed, so its residual is identically zero — including it would
dilute the correlation *toward* "no violation", i.e. bias the test in the direction that
flatters the model. The counts land in `.attrs` and a test asserts used + skipped = total.

**The ridge penalty on Cox is reported, not hidden.** Sensor pairs 8/13 and 9/14 correlate at
0.97 (corrected and uncorrected fan and core speed), so the partial likelihood is nearly flat
along those directions. The penalty biases individual coefficients toward zero, which is
exactly why the coefficient table is read as "which sensors matter" and never as "this one is
worth 0.31".

**`horizon_risk` freezes covariates but advances the band.** Freezing is an operational
necessity — at decision time today's readings are all you have. But it makes the model
under-confident about fast-degrading units, which is the calibration problem in
`evaluate.calibration`, and it is why `policy.py` sweeps the threshold instead of computing
it. The band still advances, so estimated ageing is carried forward.

**`_standard_normal_cdf` uses `math.erf`.** One normal CDF call did not justify importing
scipy into this module.

## How it can break

- **A risk set with no failures** — `risk_window_start` and `derive_band_edges` both raise.
- **A landmark beyond every unit** — `landmark_frame` raises.
- **Quasi-separation** with too many features for the event count: statsmodels warns and
  `mle_retvals["converged"]` goes False. `run_analysis.py` prints that status rather than
  hiding it. The current 17-feature specification converges with ~300 events.
- **Bands absent from the fitted model** are reindexed to 0.0, assigning the reference
  baseline — the conservative choice, since it does not extrapolate a rising hazard into
  cycles never seen.

`tests/test_survival.py` (11 tests) pins band-edge balance, band monotonicity, design-matrix
rank, the risk window, that horizon risk increases with horizon and is zero at horizon 0,
Kaplan-Meier monotonicity, landmark construction, and Schoenfeld centring.
