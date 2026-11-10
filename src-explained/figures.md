# `src/figures.py`

## What this module is for

Draws every published figure. It contains no analysis: each function takes
already-computed results and renders them, so if a number in a chart disagrees with a number
in the README, the cause is in the module that produced it and never here.

Called by `run_analysis.py` and the capstone. Depends only on matplotlib, numpy and pandas.

## The public surface

Every function saves a PNG to `outputs/figures/` and **returns its `Path`**, so a caller can
display it without hardcoding a location.

| Function | Figure | What it argues |
|---|---|---|
| `apply_style()` | — | house style, set once |
| `degradation_trajectories(risk_set, sensor, n_units)` | 01 | every engine wears differently, so read the unit not the calendar |
| `survival_curves(curves)` | 02 | the four fleet designs really do differ |
| `censoring_bias(bias_sweep)` | 03 | **the headline** — complete-case error grows, Kaplan-Meier's does not |
| `baseline_hazard(coefficients, edges)` | 04 | ageing was estimated, not assumed |
| `calibration(frame, landmark)` | 05 | the model under-states risk |
| `threshold_sweep(sweep, optimum, bootstrap, interval_best, rtf_cost)` | 06 | where to set the alarm |
| `policy_frontier(model_sweep, interval_sweep, optimum)` | 07 | same safety, more service |
| `lead_times(distribution, threshold)` | 08 | is it operationally usable |
| `cost_ratio_sensitivity(frame)` | 09 | the recommendation does not hinge on the cost ratio |
| `time_dependent_auc(frame)` | 10 | how far ahead the model can see |

## The walkthrough

`apply_style` sets rcParams once — no per-chart colour choices, so the figures read as one
set and as one set with the other projects in the portfolio. `_save` creates the directory,
tight-layouts, writes, closes the figure, and returns the path.

Each figure function is then a few lines of plotting against a fixed four-colour palette:
`PRIMARY` (the thing being argued for), `MUTED` (what it is argued against), `ACCENT` (the
contrast), `POSITIVE` (the decision).

## The non-obvious bits

**`matplotlib.use("Agg")` before importing pyplot.** `run_analysis.py` runs headless. The
order matters — setting the backend after pyplot is imported has no effect, and the failure
is an obscure Tk error rather than anything that names the cause.

**`plt.close(fig)` in `_save`.** Ten figures per run is nothing, but the `cost_ratio_sensitivity`
path draws inside a loop during development, and matplotlib warns and then leaks after twenty
open figures.

**Returning the `Path`.** This is what lets the capstone write
`Image(filename=str(figures.censoring_bias(sweep)))` — one call, no duplicated path string.
It also makes the notebook work on Kaggle, where the repo-relative `../outputs/figures/...`
does not exist but `FIGURE_DIR` still resolves correctly from the inlined module's
`__file__`.

**Figure 03 plots two series with different units on one axis** — complete-case error in
cycles of *mean* life, and Kaplan-Meier error in cycles of *median* life. Strictly these are
not the same quantity. They share an axis anyway because the argument is about *magnitude and
trend*, not about comparing the two point estimates, and both are "error against the known
truth, in cycles". The figure caption and the README both frame it that way. An alternative
would be two panels, which would weaken the visual contrast that is the entire point.

**Figure 05 sizes the scatter points by unit count.** With deciles the counts are nearly
equal, so this reads as decoration — but it stops being decoration the moment `duplicates="drop"`
collapses some bins, which happens on smaller fleets.

**The log scale on figures 04 and 06.** Odds ratios and predicted risks both span orders of
magnitude; on a linear scale the interesting region is a smear against the axis.

**Figure 01's right panel deliberately shows only failed units.** Aligning on remaining life
requires knowing the failure cycle, which censored units do not have. Plotting them at their
snapshot would draw a line that stops early for no visible reason and imply the engine was
fine when it was simply not watched any longer.

## How it can break

- **A results frame missing an expected column** raises `KeyError` at plot time. Acceptable:
  these functions are called immediately after the code that builds those frames.
- **An empty `pulled` set in `lead_times`** would make `pulled.max()` fail. Only reachable
  with a threshold no engine ever reaches, which `compare_policies` never selects.
- **`baseline_hazard` indexes `edges` by band level** with a `min(level - 1, len(edges) - 1)`
  clamp, so a coefficient table containing a band beyond the edge array plots at the last
  edge instead of raising.

**These functions have no dedicated tests**, which is deliberate: asserting on pixel output is
brittle and asserting that a function returns a `Path` proves nothing. They are covered
transitively — `run_analysis.py` calls all ten on every run, so a broken figure function fails
the analysis immediately and visibly. The figures themselves are checked by looking at them.
