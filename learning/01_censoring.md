# 01 — Censoring: the thing that makes it hard

## The anchor

You want to know how long a light bulb lasts. You switch on 100 bulbs and go on holiday for
six months. When you come back, 60 have burnt out — you know exactly when, because each has
a little timer. The other 40 are still glowing.

What is the average bulb lifetime?

The tempting answers are both wrong:

- **"Average the 60 that died."** But the 40 still glowing are the *long-lived* ones. You
  have thrown away exactly the good bulbs, so your answer is too low.
- **"Count the survivors as lasting six months."** They will last longer than that. Also
  too low, and worse.

The 40 survivors are not missing data. They carry real information: **each one lasted at
least six months.** Censoring is the name for that kind of partial information, and
survival analysis is the toolkit built to use it rather than discard it.

## The precise version

An observation is **right-censored** when you know the event has not happened by some time,
but not when it will. You observe two things per unit:

- `Y = min(T, C)` — the time you actually watched, where `T` is the true event time and `C`
  is when observation stopped
- `δ = 1{T ≤ C}` — whether the event was actually seen

If `δ = 1` you know `T` exactly. If `δ = 0` you know only `T > Y`.

There are other kinds (left-censoring: you know it happened before you looked;
interval-censoring: somewhere between two checks; left-*truncation*: the unit only entered
your view after already surviving a while). This project meets right-censoring throughout,
and left-truncation once — see the end of lesson 04.

## The bias, and its direction

This is the part worth being able to state out loud, because it is the project's finding.

Dropping censored units is called **complete-case analysis**. The intuition that it is
"just less data" is wrong. Look at when a unit gets censored: exactly when its lifetime
`T` exceeds its observation window `C`. So being censored is *caused by* being long-lived.
Filtering to the units that failed is filtering **toward short lives**.

The consequence: your fleet looks more fragile than it is. In this project, on 709 engines
at 40% censored:

| | Mean life |
|---|--:|
| Truth | 226.2 cycles |
| Complete-case | 211.9 (**−6.3%**) |
| Counting the snapshot as the lifetime | 179.0 (**−20.9%**) |

And it gets worse the more realistic the fleet is — at 68% still running the complete-case
error reaches −33 cycles. Real fleets are mostly survivors.

## The good news

Kaplan-Meier's median was off by **+2 cycles out of 207**. That is the other half of the
argument, and it is why the project has a recommendation rather than only a complaint: the
information is not destroyed by censoring, only by the wrong estimator.

## The assumption everything rests on

Survival methods need censoring to be **independent** of the event process — knowing a unit
was censored at cycle 200 tells you nothing about whether it was about to fail.

That is often false in real maintenance data, and dangerously so: units get pulled
*because* a technician thought they looked unwell. That is **informative censoring**, and
it breaks both Kaplan-Meier and Cox. This project sidesteps it by constructing censoring
that is independent by design, and names it as a limitation rather than a solved problem.

## Where this is in the code

`src/panel.py`:

- `calibrate_horizon` — solves for the horizon ceiling `U` that produces a target censored
  fraction, by bisection. So the censoring rate is *chosen*, not accidental.
- `induce_censoring` — draws `C ~ Uniform{1..U}` independently of `T`, then computes `Y`
  and `δ`.
- `build_risk_set` — drops the cycles after the snapshot, because they were never observed.

`src/evaluate.py`:

- `censoring_bias` — the three-way comparison in the table above.
- `bias_by_censoring_rate` — the sweep showing the bias grows.

The tests in `tests/test_panel.py::test_censoring_is_independent_of_lifetime` and
`tests/test_evaluate.py::test_complete_case_understates_life` pin both the mechanism and
the direction of the bias.

## Next

[02 — The hazard function](02_the_hazard_function.md)
