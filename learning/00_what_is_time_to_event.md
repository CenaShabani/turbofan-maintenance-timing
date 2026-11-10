# 00 — What "time-to-event" means

## The anchor

Imagine you run a bus depot with fifty buses. Someone asks: *"which buses will break down?"*

That is a classification question, and it has a frustrating answer: **all of them,
eventually.** The question is badly posed. What you actually need to know is *when*, and
*how confident are you*, and *is it soon enough that I should do something today*.

That is a time-to-event question, and it is a different kind of question with a different
toolkit.

## The precise version

Time-to-event analysis (also called survival analysis, from its origins in medical trials,
and reliability analysis in engineering) models a **duration**: the time from some origin
until some event happens.

Three ingredients, always:

1. **An origin.** When does the clock start? Here it is the first cycle a unit runs.
2. **An event.** What ends the duration? Here, failure. (In a churn model it is
   cancellation; in a medical trial, death or relapse.)
3. **A time scale.** Here it is *flight cycles*, not calendar days. That choice matters: an
   engine that sits in a hangar for a year does not age in the way this model means.

The output is not a single number. It is a **survival curve** — the probability the unit is
still going at each moment — or, equivalently, a **hazard** (lesson 02).

## Why not just predict the lifetime with a regression?

This is the natural first idea, and it is what almost every published analysis of the
C-MAPSS benchmark does: regress "cycles remaining" on the sensors, report the RMSE.

Three things go wrong, in increasing order of seriousness.

**It gives you a point estimate when you needed a distribution.** "This engine has 40
cycles left" is not actionable. "There is a 7% chance this engine fails in the next 30
cycles" is, because you can compare 7% against what a failure costs you.

**It cannot express "not yet".** A regression needs a target value for every training row.
What is the target for an engine that is still running?

**And that third problem is fatal, not inconvenient.** Because the only way to get a target
is to wait for the failure, a regression can only be trained on machines that have already
failed — and lesson 01 shows why that sample is systematically biased.

## Where this is in the code

Nothing yet — this lesson is framing. The first code is in lesson 01.

The project's framing decision, though, is visible right at the top of `src/panel.py`: the
module's job is to turn complete run-to-failure trajectories into the kind of partial,
still-running fleet a real operator sees, precisely so that the model has to handle the
"not yet" case.

## Next

[01 — Censoring: the thing that makes it hard](01_censoring.md)
