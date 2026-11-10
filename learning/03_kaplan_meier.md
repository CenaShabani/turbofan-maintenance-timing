# 03 — Kaplan-Meier

## The anchor

You want the survival curve of your bus fleet, and you want to assume nothing at all — no
bell curve, no exponential decay, no covariates. Just: what does the data say?

Kaplan-Meier is the answer, and it is almost embarrassingly simple once you see it.

## The precise version

Walk forward through time. At each moment where *something failed*:

1. Count how many units were still at risk just before (`n`).
2. Count how many failed at that moment (`d`).
3. The fraction surviving this instant is `(n - d) / n`.

Multiply those fractions together as you go. That running product is the survival curve —
the empirical version of the chaining identity from lesson 02.

## Where censoring enters — the elegant part

A censored unit stays in the denominator `n` for every moment it was under observation, and
then **quietly leaves the risk set**. It never appears in `d`.

So it contributes exactly what it knows ("I made it this far") and nothing it does not ("I
have no idea when I would have failed"). No unit is deleted, and none is counted as a
failure. That is why the Kaplan-Meier median in this project is off by 2 cycles out of 207,
while complete-case analysis is off by 14.

## The log-rank test

Kaplan-Meier describes; the log-rank test compares. It asks whether two or more survival
curves differ by more than chance, by comparing observed failures in each group against how
many you would expect if the groups were really the same.

This project runs it across the four fleet designs **before** any model conditions on
subset: chi-squared 44.4 on 3 degrees of freedom, p = 1.3e-09. They genuinely differ, so
conditioning on them is legitimate. Had the test come back null, adding subset indicators
would have been fitting noise.

The ordering is the habit worth keeping: establish the difference is real, *then* build the
model that uses it.

## What it cannot do

Kaplan-Meier handles one grouping variable at a time. It cannot tell you whether sensor 11
matters *holding operating regime and age fixed*. For that you need a model with
covariates, which is lesson 04.

## Where this is in the code

- `src/survival.kaplan_meier` — curves with confidence bands, per group.
- `src/survival.logrank` — the multi-group test.
- `src/evaluate.calibration` — uses Kaplan-Meier *inside each risk decile* to compute the
  observed failure rate. Not a flourish: a raw proportion would treat a unit censored 10
  cycles into a 30-cycle horizon as a survivor, understating the observed rate and making
  the model look over-cautious — the exact error this project exists to avoid making.
- `tests/test_survival.py::test_kaplan_meier_survival_is_non_increasing`.

## Next

[04 — Discrete-time hazard = logistic regression](04_discrete_time_hazard.md)
