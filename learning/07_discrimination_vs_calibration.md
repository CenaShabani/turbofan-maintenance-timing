# 07 — Discrimination vs calibration

## The anchor

Two weather forecasters.

**Alice** says "70% chance of rain" on days it rains and "20%" on days it does not. Perfect
ordering — she never once ranks a dry day above a wet one. But it rains on 95% of her "70%"
days and 5% of her "20%" days. Her *ordering* is flawless; her *numbers* are wrong.

**Bob** says "40%" every single day, and it rains on exactly 40% of days. His numbers are
perfect. His ordering is worthless — he cannot tell you which day to cancel the picnic.

Alice has **discrimination** without calibration. Bob has **calibration** without
discrimination. They are different skills, and which one you need depends entirely on what
you are going to do with the forecast.

## The precise version

- **Discrimination** — does the model *rank* correctly? Measured by concordance (lesson 08)
  and AUC. Invariant to any monotone transformation of the scores: multiply every
  prediction by 3 and discrimination is unchanged.
- **Calibration** — do the predicted probabilities *mean* what they say? Of the units given
  a 10% risk, do about 10% fail? Measured by plotting predicted against observed in bins,
  and by proper scoring rules like the Brier score.

The Brier score is squared error on a probability, so unlike concordance it punishes a model
that ranks correctly while stating the wrong number. That is why this project reports both.

## Why this project needs both — and only has one

The decision is "act when risk crosses a threshold". If the model is calibrated, the
threshold is *analytic*: act when the cost of waiting exceeds the cost of acting, which puts
it at `c_pull / c_fail`.

This model is **not** calibrated. In the top decile it predicts 62% where the observed rate
is 100%. It is systematically under-confident about the sickest engines.

The mechanism is known and is a direct consequence of a deliberate design choice.
`horizon_risk` compounds the hazard forward over 30 cycles while **freezing the sensor
readings at their current value** — so an engine that is degrading unusually fast has its
future understated. That freezing is not laziness: at the moment of the decision, today's
readings are all the operator has. A model that needed tomorrow's would be unusable.

## What to do about miscalibration

Two options, and the choice is worth understanding because the tempting one is worse.

**Recalibrate** — fit an isotonic or Platt correction mapping predicted to observed. The
standard answer, and it is a trap here: the outcome you need to fit the correction ("did
this unit fail within 30 cycles?") is *itself censored* for any unit whose observation ends
inside the window. Restricting to units where the outcome is determined reintroduces exactly
the complete-case selection bias the whole project is about. That repairs one bias by
importing another.

**Stop relying on the numbers being right.** Sweep the threshold and pick the one that
actually costs least, measured. This makes the threshold a **rank cut**, and a rank cut only
needs discrimination — which this model has in abundance (Uno's C = 0.88).

The project takes the second route, and then measures what the first-order shortcut would
have cost: setting the threshold analytically costs **21.9% more** and takes the failure
rate from 0.9% to 8.0%.

That last step is the difference between naming a limitation and measuring one.

## Where this is in the code

- `src/evaluate.calibration` — predicted vs observed by decile, with Kaplan-Meier inside
  each bin (see lesson 03 for why a raw proportion would be wrong).
- `src/evaluate.brier` — integrated Brier score against a Kaplan-Meier baseline.
- `src/policy.analytic_threshold_penalty` — the measurement of what miscalibration costs.
- `outputs/figures/05_calibration.png` — deliberately given a prominent place in the README
  rather than buried, because it is the project's main weakness and it changed what the
  project does next.

## Next

[08 — Concordance, and why there are two of them](08_concordance_harrell_uno.md)
