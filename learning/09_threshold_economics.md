# 09 — Turning a probability into a decision

The last lesson, and the one the other eight exist to reach.

## The anchor

Your model says this engine has a 7% chance of failing in the next 30 cycles.

**So what?** Do you pull it or not?

Nothing in the model answers that. The answer depends on what a planned removal costs, what
a failure costs, and how much service you give up by acting early. The model produces a
number; the decision spends it.

## The trap in the obvious metric

Score a maintenance policy on **cost per unit** and the optimum is degenerate: pull every
engine on cycle 1. Nothing ever fails. Every removal is the cheap kind. Cost per unit is
minimised — and the fleet does no work at all.

The metric has to charge for the service you gave up:

```
cost per 1,000 cycles = 1000 x (total cost) / (total cycles of service delivered)
```

Now pulling early buys a smaller numerator at the price of a smaller denominator. That is
the real trade, and the metric cannot be gamed by being conservative.

This is the single most important line in `policy.py`, and getting it wrong would not have
crashed anything — it would have quietly reported a policy that looked brilliant and was
useless. **Choosing the denominator is the analysis.**

## Sweep the threshold, do not calculate it

For a calibrated model the optimum is analytic: act when expected cost of waiting exceeds
cost of acting, i.e. at `c_pull / c_fail`.

Lesson 07 explained why that formula does not apply here — the probabilities are not
calibrated. So the threshold is **swept**: try every value, simulate the fleet, take the
cheapest. That makes it a rank cut, which only needs discrimination.

Measured penalty for using the formula anyway: **21.9% more cost, and an 8.0% failure rate
instead of 0.9%.**

## Baselines that are worth beating

A saving is meaningless without a comparison, and the comparison must be the *best* version
of the incumbent, not a strawman:

- **Run to failure** — never pull. The right policy when failure is cheap.
- **Fixed interval** — pull at a set age, **swept across all ages** so the model competes
  against the best possible calendar, not a badly chosen one.

A condition-based policy that cannot beat a well-tuned calendar is not worth the sensors.

## The result, and the shape of the argument

| Policy | Cost / 1,000 cycles | Failures | Service per unit |
|---|--:|--:|--:|
| Run to failure | 21.87 | 100% | 229 |
| Best fixed interval (140 cycles) | 7.42 | 0.9% | 140 |
| Model policy (threshold 0.077) | **5.09** | 0.9% | **204** |

The clearest way to say it: **both policies accept the same 0.9% failure rate.** The
calendar gets there by pulling everything at 140 cycles. The model gets there by pulling
each engine when *that engine* looks unwell — at 204 cycles on average. Same safety, 46%
more working life.

## Three things that turn a result into a recommendation

**1. An interval.** A threshold read off a single sweep is an argmin with no uncertainty.
Bootstrapping over *engines* gives 95% CI 0.077-0.102 on the threshold and 28.0-34.3% on the
saving.

**2. Sensitivity to what you do not know.** The failure-to-removal cost ratio is the one
input the maintenance organisation knows better than the analyst. Reporting one optimum for
one assumed ratio hands over a number that secretly depends on a guess. Reporting the curve
across 2x-20x hands over something usable — and here the saving stays 28-31% throughout, so
the recommendation does not hinge on it.

**3. Operational feasibility.** A policy with a perfect cost curve and a two-cycle lead time
cannot be executed — you cannot get a shop slot, a part and a crew inside two flights. The
lead-time *distribution* matters more than its mean, because the left tail is the real risk.
Here: median 25 cycles, 10th percentile 12.

## The general lesson

Across this portfolio the same pattern keeps appearing. In the breast-cancer project, nine
classifiers sat inside one standard deviation of each other and the lever turned out to be
the decision threshold. Here, the Random Survival Forest barely beats a logistic regression
and the lever is again the threshold — plus the choice of denominator in the cost metric.

**The modelling is rarely the lever. The decision layer usually is, and it is usually the
part that gets skipped.**

## Where this is in the code

All of `src/policy.py`:

- `score_paths` — risk at every cycle of every engine's real life, computed once.
- `simulate_threshold` — the cost model, with the guard that an alarm at or after failure is
  not a save.
- `sweep_thresholds`, `sweep_fixed_interval`, `run_to_failure` — the policies.
- `unit_outcome_matrices` — precomputation that makes each bootstrap replicate the *same*
  computation as the point estimate rather than a re-implementation that could drift.
- `bootstrap_optimum`, `cost_ratio_sensitivity`, `lead_time_distribution`,
  `analytic_threshold_penalty` — the three things above.

`tests/test_policy.py` checks the arithmetic against hand-computed values, including a
fixture whose optimum is a calculation rather than a re-run of the code being tested.

## Back to the index

[README](README.md)
