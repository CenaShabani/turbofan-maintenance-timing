# 06 — Testing proportional hazards (and getting the test wrong first)

This lesson exists because building this project produced a wrong answer before it produced
a right one, and the mistake is instructive enough to keep.

## The anchor

At each moment an engine fails, the model has an opinion about *which* of the engines still
running should have been the one to go — the sickest-looking one is most likely, the
healthiest least. Compare what the model expected against what actually happened.

If the model is right about the *shape* of risk, those comparisons are just noise. If its
expectations drift systematically as engines age — consistently surprised early, in the
other direction late — then the effect of the covariates is changing with time, and
proportional hazards is false.

## The precise version

The **Schoenfeld residual** for a covariate at failure time `t` is

```
residual = (covariate of the unit that failed) - (weighted mean over everyone at risk)
```

where the weights are `exp(beta' x)` — the model's own opinion about who was likely to go.

Under proportional hazards these residuals have no relationship with time. So: rank-correlate
each covariate's residuals against failure time. A significant correlation means that
covariate's effect moves with age.

## The mistake worth remembering

The first version of this test in the project did something that *sounds* equivalent and is
not: it correlated the **raw covariate values at failures** with failure time, skipping the
subtraction.

It reported **16 of 17 covariates violating PH** — a suspiciously uniform result, and that
uniformity is what gave it away.

Here is why it was wrong. The covariates in this project are *drift from the unit's own
baseline*, which **rises with age by construction** — that is the entire point of the
feature. So the raw values at failure times correlate with time automatically, whether or
not the model's assumption is broken. The test was measuring the feature's definition, not
the model's fit.

Subtracting the model's own expectation is exactly what removes the artefact. With the real
Schoenfeld residual the answer became **10 of 17**, the correlations were modest (-0.34 to
+0.17), and — crucially — they were *not* uniform. Seven covariates came back clean. A test
that flags everything is a test that has told you nothing.

**The transferable habit:** when a diagnostic returns a suspiciously clean sweep, suspect
the diagnostic before believing the conclusion.

## One more correction: seventeen tests, seventeen chances to be wrong

Testing 17 covariates at p < 0.05 gives roughly a 58% chance of at least one false positive
even if PH holds perfectly everywhere. The project applies **Holm-Bonferroni**, which
controls the family-wise error rate without assuming the tests are independent — and they
are not, because the sensors are correlated with each other.

Holm is the right choice over plain Bonferroni here because it is uniformly more powerful
at no cost in validity: sort the p-values, and compare the k-th smallest against
`alpha / (m - k + 1)` rather than `alpha / m` for all of them.

The adjustment moved the count from 11 to 10 — a small change, which is itself worth
knowing, because it means the finding was not resting on multiple-comparison luck.

## Where this is in the code

- `src/survival.schoenfeld_residuals` — the residual computation, with a docstring that
  spells out why the shortcut is wrong.
- `src/survival.test_proportional_hazards` — rank correlation plus the Holm adjustment.
- `tests/test_survival.py::test_schoenfeld_residuals_are_centred_by_construction` — pins the
  property that distinguishes a real residual from the naive version: it is centred, because
  it is an observation minus its expectation.

## Next

[07 — Discrimination vs calibration](07_discrimination_vs_calibration.md)
