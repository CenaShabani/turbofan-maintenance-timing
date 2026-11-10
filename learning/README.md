# The concept series

Ten short lessons on the ideas behind this project, written for someone who starts knowing
nothing about survival analysis — including the version of me that has forgotten. Read in
order; each builds on the last. Every lesson points at the exact file in `src/` where the
idea becomes code.

| # | Lesson | The one idea |
|---|---|---|
| 00 | [What "time-to-event" means](00_what_is_time_to_event.md) | The interesting question is usually *when*, not *whether* — and answering it needs a different toolkit from classification. |
| 01 | [Censoring: the thing that makes it hard](01_censoring.md) | A machine that has not failed yet is not a missing data point; it is a real observation that says "at least this long". |
| 02 | [The hazard function](02_the_hazard_function.md) | Instead of modelling the whole lifetime, model the chance of failing *right now given you have survived this far* — and the rest follows. |
| 03 | [Kaplan-Meier](03_kaplan_meier.md) | You can estimate a survival curve without any model at all, by chaining together "what fraction survived each moment". |
| 04 | [Discrete-time hazard = logistic regression](04_discrete_time_hazard.md) | If you inspect once per cycle, a survival model is just a logistic regression on one row per cycle at risk. |
| 05 | [Cox, and why this project did not use it](05_cox_and_proportional_hazards.md) | Cox buys elegance by refusing to estimate the baseline — which is exactly what a decision needs. |
| 06 | [Testing proportional hazards](06_testing_ph_schoenfeld.md) | An assumption you can test and did not is a weakness; the naive version of this test is wrong in a way worth understanding. |
| 07 | [Discrimination vs calibration](07_discrimination_vs_calibration.md) | Ranking correctly and being right about the number are different skills, and a decision needs both — or an honest workaround. |
| 08 | [Concordance, and why there are two of them](08_concordance_harrell_uno.md) | The familiar C-index is itself biased by censoring; Uno's fixes it, and the gap between them is informative. |
| 09 | [Turning a probability into a decision](09_threshold_economics.md) | The model is not the lever. The threshold is — and the metric you score it with decides whether the answer is useful or degenerate. |

## The spine

If you only read three, read **01 (censoring)**, **04 (discrete-time hazard)** and
**09 (threshold economics)**. Those three are the project: what makes the problem hard, the
model that handles it, and what the model is *for*. Lesson 06 is the one that most changed
what the code does.
