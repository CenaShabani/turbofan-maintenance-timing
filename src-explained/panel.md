# `src/panel.py`

## What this module is for

Turns complete run-to-failure trajectories into the partial, still-running fleet a real
operator would see, and reshapes what is left into the counting-process format survival
models need. It is where the project's central methodological claim lives.

Called by `run_analysis.py` and the capstone, immediately after `acquisition`. Calls nothing
else in `src/`.

## The public surface

| Function | In | Out |
|---|---|---|
| `prepare(trajectories, target_rate=0.40, seed)` | raw frame | dict: `risk_set`, `evaluation_panel`, `fleet`, `feature_columns`, `level_columns`, `sensors_kept/dropped`, `censoring` |
| `assign_regime(frame)` | frame with `setting_1..3` | `Series[str]`, 6 distinct values |
| `screen_sensors(frame)` | frame with `regime` | `(kept, dropped)` lists |
| `normalise_within_regime(frame, sensors)` | | frame of `*_z` columns |
| `add_dynamics(frame, sensors, window=5)` | | frame of `*_roll` and `*_drift` columns |
| `unit_lifetimes(trajectories)` | | `DataFrame[subset, unit, true_lifetime]` |
| `calibrate_horizon(lifetimes, target_rate)` | `ndarray, float` | `int` — the ceiling `U` |
| `induce_censoring(lifetimes, target_rate, seed)` | | adds `horizon`, `observed_time`, `event`, `censored` |
| `split_units(lifetimes, holdout_fraction, seed)` | | adds `split` |
| `build_risk_set(trajectories, fleet)` | | truncated unit-cycle panel |
| `build_evaluation_panel(trajectories, fleet)` | | untruncated unit-cycle panel |

## The walkthrough

**`assign_regime` rounds; it does not cluster.** The three operational settings fall into six
tight, well-separated groups. Rounding recovers them exactly, is deterministic, needs no seed,
and can be read off in one line — all things k-means is not.

**`screen_sensors` finds constant columns from the data.** Four sensors (1, 5, 18, 19) never
vary within a regime. A hard-coded list would be wrong the moment the subset selection
changed; a variance rule stays correct, and the dropped set is *returned* so it can be
reported rather than silently removed.

**`normalise_within_regime` z-scores each sensor against its own operating point.** Without
this, a model pooling the four subsets learns the flight profile and calls it wear — sensor
7's raw standard deviation is 0.9 in FD001 and 146 in FD002, and that difference is altitude
and throttle, not degradation.

**`add_dynamics` produces two features per sensor.** `_roll` is the trailing 5-cycle mean —
the state estimate a single noisy reading is not. `_drift` is that minus the unit's own
first-5-cycle average. The argument for drift is that engines enter service with different
unstated initial wear, so a sensor's *level* confounds "born worn" with "worn out".

**`calibrate_horizon` solves for `U` by bisection.** With `C ~ Uniform{1..U}` drawn
independently of `T`, the censored fraction is `mean(min(T-1, U) / U)`, which decreases
monotonically in `U`. Bisection therefore finds the unique `U` hitting a target rate.

**`induce_censoring` draws the horizons and computes `observed_time` and `event`.**
`true_lifetime` deliberately stays in the frame — it is the ground truth used to score the
three estimates against each other, and every model-fitting function takes the censored
columns only.

**`build_risk_set`** keeps cycles 1..`observed_time`, sets `failed` on the last row only when
the event was observed, and builds `unit_id`.

**`build_evaluation_panel`** keeps everything.

## The non-obvious bits

**`+ 0.0` in `assign_regime`.** FD001 stores its setting as `-0.0` and FD002 as `0.0`. These
are the same physical condition and the same float value, but they format to different
strings — so without the normalisation the fleet appears to fly **seven** regimes, and every
within-regime z-score is computed on a split population. This cost real debugging time and is
pinned by a test.

**Two panels, and the separation is load-bearing.** `risk_set` is the only thing any model
touches. `evaluation_panel` exists solely to simulate what a policy *would have* cost, which
needs the cycles after the snapshot. Merging them would be the exact leak the project is
designed to avoid, and the split is enforced by convention rather than by the type system —
so the docstrings say so loudly.

**`split_units` splits engines, never rows.** A unit contributes hundreds of correlated cycle
observations; splitting rows would put an engine's early cycles in the fit set and its late
cycles in the holdout, letting the model memorise the engine it is about to be scored on.
Stratified by subset so both halves contain all four fleet designs.

**`seed + 1` in `split_units`.** The split must not be correlated with the censoring draw.
Reusing the same seed for two `default_rng` streams consumed in the same order would not
actually produce identical draws here, but the offset makes the independence explicit rather
than accidental.

**`.replace(0.0, np.nan)` then `.fillna(0.0)` in `normalise_within_regime`.** A regime where
a sensor has zero variance would divide by zero. Mapping the result to 0.0 says "this reading
is exactly average for its regime", which is true, rather than propagating NaN through the
whole design matrix.

**No baseline-hazard banding here.** It lives in `survival.py` because its cut points must be
derived from the fit set alone. Computing them in `prepare`, which sees everything, would leak
the holdout's failure times into the design.

## How it can break

- **A new operating condition** would appear as a seventh regime and normalise against very
  few rows. Not currently detected — an assertion on the regime count would be a reasonable
  addition.
- **A target rate outside (0,1)** raises `ValueError`.
- **A target rate too low to reach** raises after the ceiling search exceeds 10,000x the
  longest lifetime.
- **Rolling window longer than a unit's life** — `min_periods=1` handles it; the baseline
  becomes the mean of what exists.

`tests/test_panel.py` (16 tests) pins the negative-zero regime bug, the sensor screen,
horizon calibration hitting its target and being monotone, censoring reproducibility and its
independence from lifetime, the `observed_time`/`event` definitions, unit-level splitting,
risk-set truncation, the failure flag, evaluation-panel completeness, and — importantly —
that `add_dynamics` features never change when later cycles are removed.
