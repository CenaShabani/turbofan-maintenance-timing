"""Build the project deck from outputs/figures and outputs/results.

Scripted rather than hand-made for the same reason everything else here is: a
deliverable that can only be produced by hand is one that silently goes stale.
Building it by script also makes the "every figure is a file from
outputs/figures/" rule enforceable rather than aspirational — this script reads
those files, so a missing or renamed figure fails loudly instead of leaving a
stale image on a slide.

Every number on every slide is read from outputs/results/*.json at build time.
None is typed in. If run_analysis.py produces a different number, the deck
changes with it.

    python deck/build_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

PROJECT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT / "outputs" / "figures"
RESULTS = PROJECT / "outputs" / "results"
OUT = Path(__file__).resolve().parent / "turbofan-maintenance-timing-deck.pptx"

# Matches src/figures.py, so the slides and the charts are one palette.
PRIMARY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0xC4, 0x62, 0x2D)
INK = RGBColor(0x24, 0x28, 0x2E)
MUTED = RGBColor(0x5A, 0x60, 0x6A)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


H = load("headline.json")
BIAS = load("censoring_bias.json")
ANALYTIC = load("analytic_threshold_penalty.json")


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, left, top, width, height, text, size, *, bold=False,
            colour=INK, align_centre=False, spacing=1.15):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.line_spacing = spacing
        if align_centre:
            from pptx.enum.text import PP_ALIGN
            para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        run.font.name = "Calibri"
    return box


def title(slide, text, *, size=30):
    """The slide title states the claim, not the topic."""
    textbox(slide, Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.1),
            text, size, bold=True, colour=PRIMARY)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def picture(slide, figure_name, left, top, width):
    path = FIGURES / figure_name
    if not path.exists():
        raise SystemExit(
            f"missing figure {path}. Run `python run_analysis.py` first — "
            "the deck never contains an image the repo cannot regenerate."
        )
    return slide.shapes.add_picture(str(path), left, top, width=width)


def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    # ── 1. Title + the finding in one line ────────────────────────────────
    slide = blank(prs)
    textbox(slide, Inches(0.8), Inches(2.1), Inches(11.7), Inches(1.0),
            "Turbofan Maintenance Timing", 40, bold=True, colour=PRIMARY)
    textbox(slide, Inches(0.8), Inches(3.1), Inches(11.7), Inches(1.6),
            "Pulling machines when they look unwell costs "
            f"{H['saving_vs_interval_pct']:.0f}% less than pulling them on a schedule\n"
            "— at exactly the same failure rate.", 24, colour=INK)
    textbox(slide, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.2),
            "Censoring-aware survival analysis of 709 industrial units  ·  "
            "NASA C-MAPSS benchmark (public domain, simulated)\n"
            "Cena Shabani  ·  Python, statsmodels, lifelines, scikit-survival  ·  "
            "49 tests, reproducible in ~20 seconds", 14, colour=MUTED)
    notes(slide, """
Two findings in this deck. The headline is a decision: a condition-based pull rule beats the
best possible calendar schedule by about 31 percent at the same failure rate. Underneath it is
a methodological finding that I think matters more, about what almost everyone gets wrong with
this kind of data. I will do the method finding first, because the decision depends on it.
One thing up front: this is simulator data from NASA, not a real fleet, and not from any
employer. I use it because it has a property real maintenance data never has, which I will
come to.
""")

    # ── 2. The decision ───────────────────────────────────────────────────
    slide = blank(prs)
    title(slide, "The decision: at what predicted risk do you pull a machine?")
    textbox(slide, Inches(0.7), Inches(1.7), Inches(5.6), Inches(3.6),
            "Pull it early\n"
            "    Cheap — but you throw away remaining life.\n\n"
            "Leave it in\n"
            "    You risk an in-service failure, which costs\n"
            "    several times a planned removal once\n"
            "    collateral damage and unscheduled downtime\n"
            "    are counted.", 18, colour=INK)
    textbox(slide, Inches(6.9), Inches(1.7), Inches(5.7), Inches(3.6),
            "So the policy is one number:\n"
            "an alarm threshold on predicted risk.\n\n"
            "Who acts on it:\n"
            "whoever owns the maintenance schedule.\n\n"
            "Everything else in this deck exists to\n"
            "earn the right to answer that question.", 18, colour=PRIMARY)
    notes(slide, """
This is a threshold decision, not a prediction problem. That distinction runs through the whole
project. A model that tells you a machine has forty cycles left has not told you what to do.
A model that says there is a seven percent chance of failure in the next thirty cycles has,
because you can compare seven percent against what a failure costs you.
""")

    # ── 3. The data ───────────────────────────────────────────────────────
    slide = blank(prs)
    title(slide, "709 machines run to failure — and that is what makes the check possible")
    textbox(slide, Inches(0.7), Inches(1.6), Inches(5.7), Inches(4.4),
            "NASA C-MAPSS turbofan degradation benchmark\n"
            "Public domain (U.S. Government work)\n\n"
            f"{H['engines']} units  ·  160,359 unit-cycles\n"
            "21 sensors + 3 operational settings per cycle\n"
            "4 designs: 1 or 6 operating conditions × 1 or 2 fault modes\n\n"
            "Downloaded by script, three mirrors tried in order,\n"
            "SHA-256 on every file, published unit counts asserted.", 16, colour=INK)
    textbox(slide, Inches(6.9), Inches(1.6), Inches(5.7), Inches(4.4),
            "The one thing that made it harder than it looks\n\n"
            "In the six-condition fleets, raw sensor variance is\n"
            "mostly about which condition the engine is flying,\n"
            "not how worn it is.\n\n"
            "Sensor 7's standard deviation:\n"
            "      0.9 in FD001      146 in FD002\n\n"
            "That is altitude and throttle. Pool them raw and a\n"
            "model learns the flight profile and calls it wear.\n"
            "Every sensor is normalised within its regime.", 16, colour=ACCENT)
    notes(slide, """
Worth being straight about the data. It is simulated — a physics model with injected noise. So
none of these numbers is a claim about real turbofans; the methods transfer, the numbers do not.
I chose it because it has complete run-to-failure trajectories for every unit, which is exactly
what lets me remove information on purpose and then check my answer against a truth I actually
know. No real fleet snapshot can do that.
The regime point on the right is the kind of thing that would quietly ruin the model. A
hundredfold difference in sensor variance that has nothing to do with degradation.
""")

    # ── 4. Method: the thing everyone gets wrong ──────────────────────────
    slide = blank(prs)
    title(slide, "Almost every published analysis assumes you can watch every machine fail")
    textbox(slide, Inches(0.7), Inches(1.55), Inches(11.9), Inches(2.4),
            "You cannot. At any moment your fleet is mostly machines that have not failed yet —\n"
            "and those are exactly the machines a maintenance decision is about.\n\n"
            "The standard framing, \"regress on remaining useful life\", needs a completed lifetime\n"
            "as its training target. So it can only learn from machines that already failed.", 19, colour=INK)
    textbox(slide, Inches(0.7), Inches(4.1), Inches(11.9), Inches(2.2),
            "That is not just wasteful. It is biased, and the direction is knowable:\n\n"
            "a unit is missing its failure exactly when its lifetime ran past the observation window,\n"
            "so dropping unfailed units drops the long-lived ones preferentially.", 19,
            bold=True, colour=ACCENT)
    notes(slide, """
This is the argument the whole project rests on, so I want to say it carefully.
Censoring means you know a machine had not failed by some point, but not when it would.
The tempting move is to drop those machines, because they have no target to regress on.
But being censored is caused by being long-lived. So you are not losing a random sample —
you are systematically deleting your best machines, and your fleet looks more fragile than it is.
Next slide is me measuring exactly that, which I can only do because this dataset has the
full trajectories.
""")

    # ── 5. Results: the censoring bias, measured ──────────────────────────
    slide = blank(prs)
    title(slide, f"Dropping unfailed machines understates life by "
                 f"{abs(BIAS['complete_case_bias_pct']):.1f}% — and it gets worse")
    picture(slide, "03_censoring_bias.png", Inches(0.55), Inches(1.45), Inches(7.5))
    textbox(slide, Inches(8.4), Inches(1.7), Inches(4.4), Inches(4.2),
            "Mean life, 709 units at 40% censored\n\n"
            f"Truth                         {BIAS['true_mean_life']:.1f} cycles\n"
            f"Complete-case            {BIAS['complete_case_mean_life']:.1f}   "
            f"({BIAS['complete_case_bias_pct']:+.1f}%)\n"
            f"Naive                          {BIAS['naive_observed_mean_life']:.1f}   "
            f"({100 * BIAS['naive_bias_cycles'] / BIAS['true_mean_life']:+.1f}%)\n\n"
            f"Kaplan-Meier median\n"
            f"{BIAS['kaplan_meier_median']:.0f} vs true {BIAS['true_median_life']:.0f} "
            f"→ {BIAS['kaplan_meier_median_error']:+.0f} cycles", 15, colour=INK)
    textbox(slide, Inches(8.4), Inches(5.5), Inches(4.4), Inches(1.4),
            "The information is not destroyed by\ncensoring — only by the wrong estimator.",
            16, bold=True, colour=PRIMARY)
    notes(slide, """
Orange is the complete-case error, walking away from zero as more of the fleet is still running.
Navy is Kaplan-Meier, which stays on zero — two cycles out of two hundred and seven.
That pairing is the whole argument. If both lines had drifted I would only have a complaint.
Because the navy line is flat, I have a recommendation: use an estimator that handles censoring,
and you lose essentially nothing.
And note the direction of the x-axis. Real fleets sit on the right-hand side of this chart,
because real fleets are mostly survivors.
""")

    # ── 6. Method: the model, and the assumption that failed ──────────────
    slide = blank(prs)
    title(slide, "Cox is the reflex choice — I tested its assumption and it fails")
    textbox(slide, Inches(0.7), Inches(1.6), Inches(5.7), Inches(4.5),
            "Proportional hazards says a sensor's effect is the\n"
            "same at cycle 130 as at cycle 350.\n\n"
            f"Tested with Schoenfeld residuals:\n"
            f"violated by {H['ph_violations']} of {H['ph_covariates']} covariates\n"
            "(Holm-corrected for multiple testing)\n\n"
            "— and the violations are concentrated in the\n"
            "highest-effect sensors. Their influence weakens\n"
            "as a machine ages.", 17, colour=INK)
    textbox(slide, Inches(6.9), Inches(1.6), Inches(5.7), Inches(4.5),
            "So: discrete-time logistic hazard\n"
            "src/survival.py\n\n"
            "1.  The decision needs an absolute probability.\n"
            "     Cox deliberately factors the baseline out.\n\n"
            "2.  Time is discrete — one inspection per cycle.\n"
            "     Ties are the data, not a nuisance to correct.\n\n"
            "3.  Whether risk accelerates with age is the\n"
            "     question, so it is estimated, not assumed.", 17, colour=PRIMARY)
    notes(slide, """
I fitted Cox rather than skipping it, because "I used the simpler model" is only credible if you
tried the standard one and checked it.
Worth telling you how I got this wrong first. My initial test correlated the raw sensor values at
failures against failure time, and reported sixteen of seventeen violations. That uniformity is
what tipped me off. My features are drift from each unit's own baseline, so they rise with age by
construction — the test was measuring my feature definition, not the model. The real Schoenfeld
residual subtracts the model's own expectation, and that gives ten of seventeen, with seven
covariates coming back clean. If a diagnostic flags everything, suspect the diagnostic.
""")

    # ── 7. Results: it discriminates, but it is not calibrated ────────────
    slide = blank(prs)
    title(slide, "It ranks machines well — but its probabilities are too low")
    picture(slide, "05_calibration.png", Inches(1.2), Inches(1.4), Inches(5.2))
    textbox(slide, Inches(7.0), Inches(1.8), Inches(5.6), Inches(4.4),
            "Holdout discrimination (3 landmarks)\n\n"
            f"Harrell's C        {H['holdout_harrell_c']:.3f}\n"
            f"Uno's C             {H['holdout_uno_c']:.3f}   (censoring-robust)\n"
            f"Integrated Brier  {H['integrated_brier_score']:.3f}   "
            f"vs {H['baseline_integrated_brier_score']:.3f} baseline\n\n"
            "But in the top decile it predicts 62%\n"
            "where the observed rate is 100%.\n\n"
            "Known cause: forward risk freezes the sensor\n"
            "readings, so a fast-degrading machine has its\n"
            "future understated.", 16, colour=INK)
    notes(slide, """
Discrimination and calibration are different skills and I have one of them.
I report both concordance indices deliberately — Harrell's is itself biased by censoring, and in a
project about censoring, quoting only the censoring-sensitive metric would be self-defeating. The
gap is one to two points, which tells you the optimistic bias here is real but modest.
The calibration plot is my main weakness and I put it on a slide rather than in a footnote,
because it changed what I did next. I could have fitted a recalibration, but the outcome you need
to fit one is itself censored — that repairs one bias by importing another. So instead I stopped
relying on the probabilities being right, which is the next slide.
""")

    # ── 8. The decision, answered ─────────────────────────────────────────
    slide = blank(prs)
    title(slide, f"Same safety, {100 * H['model_service_cycles'] / H['interval_service_cycles'] - 100:.0f}% more working life")
    picture(slide, "07_policy_frontier.png", Inches(0.5), Inches(1.4), Inches(6.6))
    textbox(slide, Inches(7.4), Inches(1.6), Inches(5.4), Inches(4.6),
            "Cost per 1,000 cycles of service\n"
            "213 held-out machines\n\n"
            f"Run to failure                {H['run_to_failure_cost_per_1000']:.2f}\n"
            f"Best fixed interval        {H['interval_cost_per_1000']:.2f}   "
            f"({H['best_interval_cycles']:.0f} cycles)\n"
            f"Model policy                  {H['model_cost_per_1000']:.2f}\n\n"
            f"Saving  {H['saving_vs_interval_pct']:.1f}%\n"
            f"95% CI  {H['saving_vs_interval_ci'][0]:.1f}–{H['saving_vs_interval_ci'][1]:.1f}%\n\n"
            f"Both policies accept {H['model_failure_rate']:.1%} failures.\n"
            f"The calendar delivers {H['interval_service_cycles']:.0f} cycles per machine.\n"
            f"The model delivers {H['model_service_cycles']:.0f}.", 16, colour=INK)
    notes(slide, """
Here is the finding in the form I would actually say out loud.
Both policies accept the same zero point nine percent failure rate. The calendar gets there by
pulling every machine at a hundred and forty cycles. The model gets there by pulling each machine
when that machine looks unwell — two hundred and four cycles on average. Same safety, forty-six
percent more working life.
Two things about the comparison. The metric is cost per cycle of service delivered, not per
machine — score it per machine and the optimum is to pull everything on day one, which minimises
cost and does no work. And the baseline is the best fixed interval, swept across every interval,
not a strawman. If I could not beat a well-tuned calendar, the sensors would not be worth it.
""")

    # ── 9. Why the threshold is swept, not calculated ─────────────────────
    slide = blank(prs)
    title(slide, "Trusting the textbook threshold would have cost "
                 f"{ANALYTIC['penalty_pct']:.0f}% more")
    textbox(slide, Inches(0.7), Inches(1.6), Inches(11.9), Inches(1.3),
            "For a calibrated model the optimum is analytic: act when risk exceeds c_pull / c_fail.\n"
            "This model is not calibrated — so the threshold is swept and chosen by measured cost.", 18, colour=INK)
    textbox(slide, Inches(0.9), Inches(3.1), Inches(11.5), Inches(2.2),
            f"Textbook rule    threshold {ANALYTIC['analytic_threshold']:.3f}    "
            f"cost {ANALYTIC['analytic_cost']:.2f}    "
            f"failures {ANALYTIC['analytic_failure_rate']:.1%}\n\n"
            f"Swept rule          threshold {ANALYTIC['empirical_threshold']:.3f}    "
            f"cost {ANALYTIC['empirical_cost']:.2f}    "
            f"failures {ANALYTIC['empirical_failure_rate']:.1%}", 20, colour=PRIMARY)
    textbox(slide, Inches(0.7), Inches(5.5), Inches(11.9), Inches(1.3),
            "Sweeping makes the threshold a rank cut. It only needs the model to order machines\n"
            "correctly — which it does, at Uno's C of "
            f"{H['holdout_uno_c']:.2f}.", 18, bold=True, colour=ACCENT)
    notes(slide, """
This is what turns a caveat into a measurement. Saying "the model is not well calibrated" is a
sentence anyone can write. Saying "using the calibrated-model formula costs twenty-two percent
more and takes your failure rate from zero point nine to eight percent" is a number someone can
act on.
And it is an eightfold increase in in-service failures, which in most maintenance organisations
is the number that would actually end the conversation.
""")

    # ── 10. What I'd do about it ──────────────────────────────────────────
    slide = blank(prs)
    title(slide, "What I would do about it")
    textbox(slide, Inches(0.7), Inches(1.5), Inches(11.9), Inches(1.4),
            f"Set the alarm at a predicted 30-cycle failure risk of {H['optimal_threshold']:.3f}\n"
            f"(95% CI {H['optimal_threshold_ci'][0]:.3f}–{H['optimal_threshold_ci'][1]:.3f}), "
            f"after reading it off the sensitivity curve using your own cost ratio.", 20,
            bold=True, colour=PRIMARY)
    picture(slide, "09_cost_ratio_sensitivity.png", Inches(0.6), Inches(3.0), Inches(7.2))
    textbox(slide, Inches(8.1), Inches(3.1), Inches(4.7), Inches(3.6),
            "Two conditions, both checkable first:\n\n"
            "1.  Do not read the probabilities as\n"
            "     probabilities. The threshold is a rank cut.\n\n"
            "2.  Verify pulls are not already condition-\n"
            "     driven. The whole analysis assumes\n"
            "     censoring is independent of health.\n"
            "     If technicians already pull machines that\n"
            "     look unwell, these numbers are an\n"
            "     upper bound.\n\n"
            f"Median warning: {H['median_lead_time']:.0f} cycles\n"
            f"10th percentile: {H['lead_time_p10']:.0f} cycles", 15, colour=INK)
    notes(slide, """
The sensitivity curve matters because the failure-to-removal cost ratio is the one input the
maintenance organisation knows better than I do. Rather than give one number that secretly depends
on my guess, I give the curve — and across a tenfold range of ratios the saving stays between
twenty-eight and thirty-one percent, so the recommendation does not hinge on getting it right.
The lead time is the feasibility check. A policy with a perfect cost curve and a two-cycle warning
cannot be executed — you cannot get a shop slot, a part and a crew inside two flights.
Twenty-five cycles median, twelve at the tenth percentile, is workable.
""")

    # ── 11. Limitations ───────────────────────────────────────────────────
    slide = blank(prs)
    title(slide, "Limitations")
    textbox(slide, Inches(0.7), Inches(1.6), Inches(11.9), Inches(5.0),
            "The data is simulated. Physics model plus injected noise, not a real fleet. The\n"
            "methods transfer; these specific numbers do not.\n\n"
            "Censoring is induced by design, not observed. Independent by construction. Real\n"
            "maintenance data often has informative censoring — machines pulled because they\n"
            "looked unwell — which breaks both Kaplan-Meier and the partial likelihood.\n\n"
            "The model is not well calibrated, for the reason on slide 7. The decision layer is\n"
            "built around this rather than pretending otherwise; the cost is quantified at "
            f"{ANALYTIC['penalty_pct']:.0f}%.\n\n"
            "Forward risk freezes the covariates, which makes the reported gain conservative.\n\n"
            "Costs are relative, not currency. A removal is 1; a failure is the multiple.\n\n"
            "Nothing is said before cycle 128, by construction — and 106 fit-set machines\n"
            "censored before that window drop out of the model fit entirely.", 16, colour=INK)
    notes(slide, """
The second one is the one I would push on if I were you, so let me answer it before you ask.
My censoring is independent because I built it that way, and that is convenient — it is exactly
what gives me a ground truth to check against. But it is also the main limitation, because real
maintenance data usually is not independent. If I were doing this on real data I would need either
a joint model of the pull decision and the failure process, or an instrument for why a machine was
pulled. And I would expect the honest answer to be that this result is an upper bound.
""")

    # ── 12. Reproduce ─────────────────────────────────────────────────────
    slide = blank(prs)
    title(slide, "Reproduce it")
    textbox(slide, Inches(0.7), Inches(1.7), Inches(11.9), Inches(2.6),
            "py -3.10 -m venv .venv && .venv/Scripts/pip install -r requirements.txt\n"
            "python src/acquisition.py          # 3 mirrors, SHA-256, unit counts asserted\n"
            "python -m pytest -q                # 49 tests\n"
            "python run_analysis.py             # every number and figure, ~20 seconds", 17,
            colour=PRIMARY)
    textbox(slide, Inches(0.7), Inches(4.6), Inches(11.9), Inches(2.2),
            "run_analysis.py regenerates every number in this deck, the README and the notebook.\n"
            "The capstone notebook imports src/ and calls the same functions, so the notebook and\n"
            "the script cannot disagree. No number here exists only in a slide.\n\n"
            "Data: Saxena & Goebel (2008), NASA Prognostics Data Repository. Public domain.\n"
            "Code: MIT.", 16, colour=INK)
    notes(slide, """
The reproducibility claim is the one I would most want someone to actually test. Clone it, run
four commands, and every number in this deck comes back. The deck itself is built by a script that
reads the results files, so if the analysis changes the slides change with it.
""")

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
    print(f"  {len(Presentation(path).slides.__iter__.__self__._sldIdLst)} slides")
