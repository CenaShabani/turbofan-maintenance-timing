# 08 — Concordance, and why there are two of them

## The anchor

Take every pair of engines. For each pair ask: **did the one the model thought was sicker
actually fail first?**

The fraction of pairs where the answer is yes is the concordance index. 0.5 is a coin flip;
1.0 is perfect ordering. It is the survival-analysis version of AUC, and it means almost
exactly the same thing.

## The complication censoring introduces

Not every pair can be judged. If engine A was censored at cycle 150 and engine B failed at
cycle 300, you know A did not fail before 150 — but you have no idea whether it would have
beaten B. That pair is **not comparable** and gets dropped.

Here is the problem: **the dropped pairs are not a random sample.** Which pairs survive the
comparability filter depends on the censoring distribution, so the estimate depends on it
too. Harrell's C — the familiar one — therefore has a bias that grows with the censoring
rate, and it is not even guaranteed to be in a consistent direction across datasets.

For a project whose entire subject is censoring, quoting only the censoring-sensitive
metric would be self-defeating.

## Uno's C

Uno's concordance fixes it with **inverse probability of censoring weighting** (IPCW). The
idea is the reusable one:

> If certain observations are systematically less likely to survive into your sample,
> up-weight the ones that did, in proportion to how unlikely they were to make it.

Concretely: estimate the censoring distribution (a Kaplan-Meier curve on the *censoring*
event rather than the failure event), then weight each comparable pair by the inverse of its
probability of being comparable. Pairs that were unlikely to be observable count for more,
which restores the balance the filter destroyed.

One practical detail that matters: the censoring distribution must be estimated on the
**fit** data, not the evaluation data. Using the evaluation set's own censoring pattern to
weight the evaluation set is circular.

## What this project found

| Landmark cycle | Harrell's C | Uno's C |
|--:|--:|--:|
| 150 | 0.883 | 0.864 |
| 175 | 0.899 | 0.889 |
| 200 | 0.904 | 0.892 |

The gap is small — 1 to 2 points — which is itself the useful finding: at 40% censoring the
optimistic bias is real but modest. Had the gap been 10 points, the headline number would
have needed rewriting. Reporting both is what lets you say that rather than hope it.

## Landmarking: which C-index, exactly?

A model with time-varying covariates does not have *one* concordance. Its predictions change
every cycle as the sensors move, so "the model's C-index" is ambiguous until you say which
covariate values you used.

**Landmarking** resolves it. Fix a cycle. Keep the units still at risk then. Use their
readings at exactly that instant. Score how well the predicted risk orders what happens
next. That is both well defined and exactly the operational question: *standing at cycle
150, which of these engines should I worry about?*

Units already gone by the landmark are excluded, not counted as failures — an operator
standing at cycle 150 is deciding about engines running at cycle 150.

## Where this is in the code

- `src/survival.landmark_frame` — builds the landmark population.
- `src/evaluate.landmark_scores` — both indices at each landmark.
- `src/evaluate.time_dependent_auc` — AUC as a function of how far ahead you ask, which is
  the honest statement of how much warning the model gives (`outputs/figures/10_*.png`).
- `src/evaluate.bootstrap_metric` — confidence intervals that resample **units**, not rows.
  Resampling rows would treat one engine's hundreds of correlated observations as
  independent draws and give an interval several times too narrow.

## Next

[09 — Turning a probability into a decision](09_threshold_economics.md)
