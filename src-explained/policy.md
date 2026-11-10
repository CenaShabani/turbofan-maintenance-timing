# `src/policy.py`

## What this module is for

Turns a per-cycle failure probability into a maintenance decision, and measures what that
decision is worth against the policies operators actually run. Everything before this module
produces a number; this module spends it.

Called by `run_analysis.py` and the capstone, last. Consumes `panel.py`'s **evaluation
panel** (untruncated, holdout only) and a fitted model from `survival.py`.

## The public surface

| Function | In | Out |
|---|---|---|
| `score_paths(result, evaluation_panel, features, edges, risk_window_start, horizon)` | | panel + `predicted_risk` per row |
| `simulate_threshold(paths, threshold, cost_ratio)` | | dict: cost, pulled, failed, service, lead time |
| `sweep_thresholds(paths, thresholds, cost_ratio)` | | one row per threshold |
| `simulate_fixed_interval` / `sweep_fixed_interval` | | the calendar baseline |
| `run_to_failure(paths, cost_ratio)` | | the never-pull baseline |
| `compare_policies(paths, cost_ratio)` | | best of each, plus both sweeps and the savings |
| `analytic_threshold_penalty(paths, cost_ratio)` | | cost of trusting the textbook formula |
| `cost_ratio_sensitivity(paths, ratios)` | | optimum as a function of the cost ratio |
| `lead_time_distribution(paths, threshold)` | | per-engine warning time |
| `unit_outcome_matrices(paths, thresholds, intervals, cost_ratio)` | | precomputed engines × thresholds grids |
| `bootstrap_optimum(paths, cost_ratio, n_boot=400)` | | CIs on threshold, cost and saving |

## The walkthrough

**`score_paths`** computes predicted risk at every cycle of every engine's *real* life,
once. The risk path does not depend on the threshold — only the decision does — so
recomputing it per threshold would multiply the work by several hundred for nothing.

**`simulate_threshold`** finds each engine's first alarm cycle, decides whether that counts
as a save, and accumulates cost and service.

**`sweep_fixed_interval`** does the same for a calendar policy across a grid of intervals, so
the model competes against the *best* calendar rather than an arbitrary one.

**`unit_outcome_matrices`** precomputes, for every engine and every threshold, the service
delivered and the cost incurred.

**`bootstrap_optimum`** then resamples engine indices and takes two column sums per replicate.

## The non-obvious bits

**The denominator is the analysis.** Cost per 1,000 cycles *of service delivered*, not cost
per engine. Score per engine and the optimum is degenerate: pull everything on cycle 1,
nothing fails, every removal is cheap, and the fleet does no work. Getting this wrong would
not have crashed anything — it would have quietly reported a policy that looked brilliant and
was useless.

**`pulled = pull_cycle.notna() & (pull_cycle < units)`.** An alarm at *or after* the failure
cycle is not a save; the engine had already failed. Dropping the strict inequality is the
single easiest way to accidentally report a policy that beats reality, and it is pinned by
`test_an_alarm_at_or_after_failure_is_not_a_save`.

**Cycles before the risk window get zero risk, not a model prediction.** The model is
explicitly conditional on surviving to cycle 128 and has nothing to say before it. No engine
fails there either, so the alarm has nothing to catch.

**The `np.maximum.accumulate` trick in `unit_outcome_matrices`.** An engine alarms at
threshold `t` exactly when its **running maximum** risk first reaches `t`. That running
maximum is monotone by construction, so `np.searchsorted` finds the first qualifying cycle
for *every* threshold in one vectorised call instead of a loop per threshold. This is what
takes the bootstrap from about ten minutes to about a tenth of a second.

More important than the speed: it makes each bootstrap replicate the *same* computation as
the point estimate rather than a parallel re-implementation that could silently drift.
`test_vectorised_matrices_reproduce_the_sweep_exactly` asserts they agree to `rtol=1e-12`,
and in the real run the difference is exactly 0.

**`THRESHOLD_GRID` is log-spaced below 0.5 and linear above.** Predicted 30-cycle risks are
mostly small, so a purely linear grid would spend most of its points where no engine ever
sits and resolve the interesting region badly.

**`DEFAULT_COST_RATIO = 5.0` is a placeholder, deliberately.** Nothing rests on it —
`cost_ratio_sensitivity` sweeps 2× to 20× and the recommendation is reported as a function of
the ratio. The constant exists so the module has a runnable default, not because 5 is right.

**`analytic_threshold_penalty` exists to convert a caveat into a number.** "The model is not
well calibrated" is a sentence anyone can write. "Using the calibrated-model formula costs
21.9% more and takes failures from 0.9% to 8.0%" is a measurement.

**`bootstrap_optimum` resamples engines, then re-argmins.** Re-finding the optimum inside
each replicate rather than fixing the point-estimate threshold is what makes the interval an
interval on the *optimum* rather than on the cost at one arbitrary threshold.

## How it can break

- **A threshold nothing reaches** degenerates to run-to-failure. Correct, and tested.
- **An engine with no rows in the risk window** never alarms and always fails. Correct.
- **`cost_ratio` of 1.0** makes failure and planned removal equally expensive, so
  run-to-failure wins. Not guarded — it is a legitimate input with a sensible answer.
- **Ties at the optimum**: `idxmin` takes the first, i.e. the lowest threshold among equals.
  Slightly conservative (pulls earlier), which is the right direction to err.

`tests/test_policy.py` (9 tests) checks the cost metric against hand-computed arithmetic, the
at-or-after-failure guard, threshold degeneration to run-to-failure, the vectorised/loop
agreement, lead times, monotonicity of the pull cycle in the threshold, bootstrap bracketing,
and — the one the plan called for — a **fixed-interval optimum verified against a
calculation**: with all engines failing at cycle 200 the best interval must be the largest
grid value below 200 at cost exactly `1000 / I`, with a known jump to `1000 × 5 / 200` on the
other side of the cliff.
