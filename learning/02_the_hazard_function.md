# 02 — The hazard function

## The anchor

You are 40 years old. What is the chance you die this year?

Notice you did **not** ask "what is the chance a newborn dies at age 40". You asked *given
that I have already made it to 40*. That conditioning is the whole idea.

The hazard is that quantity: **the chance the event happens right now, given it has not
happened yet.**

## The precise version

In discrete time — which is what this project has, since an engine is inspected once per
flight cycle — the hazard at time `t` is

```
lambda(t) = P(event at t | survived past t-1)
```

Survival is what you get by chaining hazards together. You survive to `t` only by surviving
every step along the way:

```
S(t) = (1 - lambda(1)) x (1 - lambda(2)) x ... x (1 - lambda(t))
```

That product is the single most useful identity in the subject. It turns "model the whole
lifetime", which is hard, into "model one step at a time", which is a problem you already
know how to solve.

## Why modelling the hazard is the right move

**Censoring becomes free.** A unit censored at cycle 200 contributes to the hazard at
cycles 1 through 200 — it was at risk and did not fail — and simply stops contributing
after. No special case, no imputation, no dropped rows. Compare that to modelling the
lifetime directly, where a censored unit has no target at all.

**The shape is informative.** A hazard that *rises* with age means wear-out — the thing
maintenance schedules exist for. A *flat* hazard means failures are random and no amount of
scheduling helps. A *falling* hazard means infant mortality: the longer it has run, the
safer it is. Those three demand completely different policies, and you cannot tell them
apart from average lifetime alone.

**It is what the decision needs.** "Pull the unit when its risk over the next 30 cycles
exceeds 7.7%" is a statement about compounded hazard, which the identity above gives
directly.

## The shape in this project

Estimated, not assumed — cycle enters the model as a set of band dummies, so the hazard is
free to take any shape. See `outputs/figures/04_baseline_hazard.png`. It is flat at zero
until cycle 128 (no engine in this fleet fails earlier at all), then rises.

## Where this is in the code

- `src/survival.fit_discrete_hazard` — estimates the hazard directly.
- `src/survival.horizon_risk` — the compounding identity above, as a loop over the horizon:
  multiply `(1 - lambda)` forward, return `1 - S`.
- `src/survival.derive_band_edges` — the bands that let the shape be estimated.
- `tests/test_survival.py::test_horizon_risk_increases_with_the_horizon` pins the identity's
  most basic consequence: waiting longer cannot make you safer.

## Next

[03 — Kaplan-Meier](03_kaplan_meier.md)
