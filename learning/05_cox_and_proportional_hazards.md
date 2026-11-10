# 05 — Cox, and why this project did not use it

## The anchor

Cox proportional hazards is the most cited statistical model in the world outside linear
regression. If you say "survival analysis" to someone who half-remembers a stats course,
this is what they picture. So a project that does not use it owes an explanation.

## The precise version

Cox splits the hazard into two pieces:

```
hazard(t, x) = baseline(t) x exp(beta' x)
```

The clever move — the one that made Cox famous — is that you can estimate `beta` **without
ever estimating `baseline(t)`**. At each failure time you ask only: *of everyone at risk
right now, how surprising is it that this particular one failed?* The baseline cancels out
of that comparison, so the shape of ageing never has to be specified. That is the partial
likelihood.

The name comes from what it assumes: the ratio of two units' hazards is **proportional** —
constant over time. If engine A is twice as likely to fail as engine B today, it is twice
as likely at every age.

## Three reasons this project uses a discrete-time model instead

**1. The decision needs an absolute probability, not a ranking.**

This is the big one. The partial likelihood deliberately discards the baseline, so Cox
estimates *relative* risk and is silent about the absolute level unless you bolt a baseline
estimator back on afterwards. But the decision here is "pull the unit when its 30-cycle
failure risk exceeds *p*" — a statement about an absolute probability. A model that
estimates one directly is the honest fit to the question.

**2. Time is genuinely discrete.**

Cox assumes continuous time, in which two units failing at exactly the same instant is
impossible. Here time is integer flight cycles, so ties are not a nuisance to approximate
away with Efron or Breslow corrections — **ties are the data**. The discrete-time
formulation matches the data-generating process instead of correcting for a mismatch.

**3. The assumption fails.**

Tested, not assumed. **10 of 17 covariates violate proportional hazards** after correcting
for multiple testing, and the violations are concentrated in the *highest-effect* sensors —
their influence weakens as a unit ages. Lesson 06 covers how that was tested and why the
obvious version of the test is wrong.

## So why fit Cox at all?

Because "I used the simpler model" is only credible if the standard one was tried and its
assumption actually checked. Fitting Cox and reporting that it fails is a much stronger
position than never fitting it — and it costs about half a second of compute.

There is also a small honest caveat in how it was fitted: a ridge penalty of 0.1, because
several sensor pairs are near-duplicates by construction (corrected and uncorrected fan
speed correlate at 0.97) and the partial likelihood is nearly flat along those directions
without it. The penalty biases individual coefficients toward zero, which is why the
coefficient table is read as "which sensors matter" and never as "this one is worth exactly
0.31".

## Where this is in the code

- `src/survival.fit_cox_time_varying` — the counting-process fit, clustered by unit.
- `src/survival.test_proportional_hazards` — the check.
- `src/survival.fit_discrete_hazard` — what actually gets used.

## Next

[06 — Testing proportional hazards](06_testing_ph_schoenfeld.md)
