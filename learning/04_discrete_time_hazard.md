# 04 — Discrete-time hazard = logistic regression

## The anchor

Here is the trick, and it is genuinely a trick — the kind that makes a hard-sounding topic
collapse into something you already know.

You have 700 engines with different lifetimes. Instead of one row per engine, write **one
row per engine per cycle it was running**. Each row asks a yes/no question: *did it fail
this cycle?*

Almost every row says no. Occasionally one says yes.

That is a binary outcome with predictors. That is logistic regression. **You already know
how to do survival analysis.**

## The precise version

Build the **risk set**: one row per unit per cycle at risk. On each row record

- the outcome `failed` — 1 only on the final cycle of a unit whose failure was *observed*
- the covariates at that cycle
- the age of the unit at that cycle

Then fit `logit(hazard) = alpha(t) + beta' x`. The fitted probability *is* the hazard, and
`alpha(t)` is the baseline: how risky this age is before covariates.

Two consequences worth noticing:

- **Censoring is handled automatically.** A censored unit contributes rows with outcome 0
  for every cycle it was watched, then stops. Nothing deleted, nothing imputed.
- **Time-varying covariates are free.** Each row carries that cycle's readings, so the model
  naturally uses the fact that an engine's sensors change as it degrades.

## The four design choices this project made

**Age as bands, not a linear term.** A linear term would impose that each extra cycle
multiplies the odds of failing by a constant. Whether the hazard accelerates is exactly the
question a maintenance policy turns on, so it is estimated: one dummy per band of cycles,
and the hazard is free to rise, fall or wander.

**Equal-*event* bands, not equal-width.** With 20-cycle bands, several bands in the sparse
tail contain zero failures. A band with no events wants a coefficient of minus infinity —
"perfect separation" — and the fit will not converge; the Hessian is singular. Cutting at
the quantiles of observed failure cycles puts about thirty events in each band, and none is
empty.

**Clustered standard errors.** One engine contributes hundreds of rows and they are
anything but independent — an engine degrading fast does so on all of them. Treating them
as independent draws would shrink the standard errors by roughly the square root of the
average unit length, making everything look far more precise than it is.

**Left truncation at cycle 128.** No engine in this fleet fails before cycle 128, so the
~40,000 rows below that have a *constant* outcome. A logistic model asked to explain them
drives its intercept to minus infinity — separation again, from the other direction. The
fix is to condition the whole analysis on having survived to 128, which is *delayed entry*.
It costs something real: 106 engines censored before that window drop out entirely. That is
stated rather than buried, because everything the model says is now conditional on it.

Notice that three of those four are responses to the same underlying pathology. Separation
is *the* recurring failure mode of hazard models on sparse events, and recognising it is
most of the skill.

## Why not a fancier model?

The project fits a Random Survival Forest as a comparator, precisely so this has a measured
answer rather than an opinion. The general lesson — the same one the breast-cancer project
reached with nine classifiers inside one standard deviation of each other — is that the
modelling is usually not the lever. Lesson 09 is where the lever actually is.

## Where this is in the code

- `src/panel.build_risk_set` — the one-row-per-cycle expansion.
- `src/survival.fit_discrete_hazard` — the logistic fit with clustered errors.
- `src/survival.derive_band_edges`, `assign_band`, `build_design` — the baseline hazard.
- `src/survival.restrict_to_risk_window` — the left truncation, argued at length.

## Next

[05 — Cox, and why this project did not use it](05_cox_and_proportional_hazards.md)
