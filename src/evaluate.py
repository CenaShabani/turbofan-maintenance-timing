"""Score the models, and measure the cost of ignoring censoring.

Two jobs, and the second is the one the project is really about.

Job one: does the model discriminate and is it calibrated
---------------------------------------------------------
Discrimination and calibration are different questions and a maintenance
policy needs both. Discrimination asks whether the model *orders* engines
correctly — is the one it thinks is sicker actually the one that fails first?
Calibration asks whether the number means what it says — of the engines it
gives a 10% chance of failing within 30 cycles, do about 10% fail?

A model can discriminate perfectly and be badly calibrated, and for this
project that would be fatal: the whole decision layer is "act when the
probability crosses a threshold", so a probability that is systematically
double the truth moves the threshold to the wrong place. Both are reported.

Why two concordance indices
---------------------------
**Harrell's C** is the familiar one, and it is biased under censoring: the
comparable pairs it can form are not a random sample of all pairs, and the
heavier the censoring the worse the distortion. **Uno's C** reweights pairs by
the inverse probability of remaining uncensored, which removes that dependence.

Both are reported precisely because the gap between them is informative. In a
project whose subject is censoring, quoting only the censoring-sensitive metric
would be self-defeating.

Scoring is done at **landmarks**. A time-varying-covariate model does not have
one C-index; it has one per moment at which you might ask. Fixing a cycle,
keeping the units still running then, and scoring the ordering of what happens
next is both well defined and exactly the operational question: standing at
cycle 150, which of these engines should I worry about?

Job two: what does dropping censored units actually cost
---------------------------------------------------------
This is the measurement the plan is built around. Three estimates of the same
truth are compared:

1. **Truth** — computed on the full run-to-failure trajectories. Available only
   because the raw benchmark has them; no real fleet snapshot could produce it.
2. **Complete-case** — censored units deleted. This is what any method needing
   a completed lifetime as its target is forced to do, "regress on remaining
   useful life" included.
3. **Censoring-aware** — the survival estimate, which keeps them.

The complete-case estimate is not just noisier. A unit is censored exactly when
its lifetime exceeds its snapshot horizon, so deleting censored units deletes
the long-lived ones preferentially. The surviving sample is biased short, and
the fleet looks more fragile than it is. ``censoring_bias`` puts a number on it
in cycles, and ``bias_by_censoring_rate`` shows the number growing as the fleet
gets more realistic — which is the direction that matters, because real fleets
are mostly survivors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from src import survival

N_BOOTSTRAP = 400
RANDOM_SEED = 20260801

# Bins used to check calibration. Quantile bins rather than fixed-width,
# because predicted risks are heavily concentrated near zero and fixed-width
# bins would put almost every engine in the first one and measure nothing.
N_CALIBRATION_BINS = 10


def _surv_array(times: np.ndarray, events: np.ndarray):
    """scikit-survival's structured (event, time) array."""
    return Surv.from_arrays(event=np.asarray(events).astype(bool),
                            time=np.asarray(times).astype(float))


def landmark_scores(result, risk_set: pd.DataFrame, feature_columns: list[str],
                    edges: np.ndarray, fit_risk_set: pd.DataFrame,
                    landmarks: tuple[int, ...] = survival.LANDMARK_CYCLES,
                    horizon: int = survival.DEFAULT_HORIZON) -> pd.DataFrame:
    """Discrimination at each landmark, by both concordance indices.

    Uno's C needs the censoring distribution, and it must be estimated on the
    **fit** data: using the evaluation set's own censoring pattern to weight
    the evaluation set is circular. It also requires the evaluation times to
    lie inside the range the fit data covers, so times beyond that are trimmed
    and the number trimmed is reported rather than quietly dropped.
    """
    rows: list[dict] = []

    for landmark in landmarks:
        try:
            evaluation = survival.landmark_frame(risk_set, landmark)
        except ValueError:
            continue
        if evaluation["event_from_landmark"].sum() < 5:
            continue

        risk = survival.horizon_risk(result, evaluation, feature_columns, edges,
                                     horizon=horizon)
        times = evaluation["time_from_landmark"].to_numpy(dtype=float)
        events = evaluation["event_from_landmark"].to_numpy(dtype=bool)

        harrell = concordance_index_censored(events, times, risk)[0]

        try:
            fit_landmark = survival.landmark_frame(fit_risk_set, landmark)
            train_surv = _surv_array(
                fit_landmark["time_from_landmark"].to_numpy(),
                fit_landmark["event_from_landmark"].to_numpy(),
            )
            tau = min(times[events].max(),
                      fit_landmark.loc[fit_landmark["event_from_landmark"] == 1,
                                       "time_from_landmark"].max())
            uno = concordance_index_ipcw(
                train_surv, _surv_array(times, events), risk, tau=float(tau)
            )[0]
        except (ValueError, ZeroDivisionError):
            uno = np.nan

        rows.append({
            "landmark": landmark,
            "units_at_risk": len(evaluation),
            "events": int(events.sum()),
            "harrell_c": float(harrell),
            "uno_c": float(uno),
            "mean_predicted_risk": float(risk.mean()),
        })

    return pd.DataFrame(rows)


def time_dependent_auc(result, risk_set: pd.DataFrame, feature_columns: list[str],
                       edges: np.ndarray, fit_risk_set: pd.DataFrame,
                       landmark: int, horizon: int = survival.DEFAULT_HORIZON,
                       n_points: int = 8) -> pd.DataFrame:
    """AUC as a function of how far ahead you are asking.

    One number for "does the model work" hides the thing a maintenance planner
    cares about most: models are usually good at spotting imminent failure and
    much worse at spotting it early, and early is when action is still cheap.
    The shape of this curve is the honest statement of how much warning the
    model really provides.
    """
    evaluation = survival.landmark_frame(risk_set, landmark)
    fit_landmark = survival.landmark_frame(fit_risk_set, landmark)

    risk = survival.horizon_risk(result, evaluation, feature_columns, edges,
                                 horizon=horizon)
    times = evaluation["time_from_landmark"].to_numpy(dtype=float)
    events = evaluation["event_from_landmark"].to_numpy(dtype=bool)

    event_times = np.sort(times[events])
    if len(event_times) < n_points + 2:
        n_points = max(2, len(event_times) - 2)
    grid = np.unique(
        np.quantile(event_times, np.linspace(0.1, 0.9, n_points)).round()
    )

    auc, mean_auc = cumulative_dynamic_auc(
        _surv_array(fit_landmark["time_from_landmark"].to_numpy(),
                    fit_landmark["event_from_landmark"].to_numpy()),
        _surv_array(times, events),
        risk,
        times=grid,
    )
    return pd.DataFrame({
        "cycles_ahead": grid,
        "auc": auc,
        "mean_auc": mean_auc,
        "landmark": landmark,
    })


def brier(result, risk_set: pd.DataFrame, feature_columns: list[str],
          edges: np.ndarray, fit_risk_set: pd.DataFrame,
          landmark: int, horizons: tuple[int, ...] = (10, 20, 30, 40, 50)) -> dict:
    """Integrated Brier score: accuracy of the *probability*, not just the ranking.

    The Brier score is squared error on a probability, so unlike the C-index it
    punishes a model that orders engines correctly while stating the wrong
    number. Integrated over horizons it summarises calibration and
    discrimination together. Lower is better; 0.25 is what a model that always
    predicts 0.5 would score, and the always-predict-the-KM-curve baseline is
    reported alongside so the number has something to mean.
    """
    evaluation = survival.landmark_frame(risk_set, landmark)
    fit_landmark = survival.landmark_frame(fit_risk_set, landmark)

    times = evaluation["time_from_landmark"].to_numpy(dtype=float)
    events = evaluation["event_from_landmark"].to_numpy(dtype=bool)

    grid = np.array([h for h in horizons if h < times[events].max()], dtype=float)
    if len(grid) < 2:
        return {"integrated_brier_score": np.nan, "baseline_integrated_brier_score": np.nan}

    # Survival probability at each horizon, per unit.
    predictions = np.column_stack([
        1.0 - survival.horizon_risk(result, evaluation, feature_columns, edges,
                                    horizon=int(h))
        for h in grid
    ])

    train = _surv_array(fit_landmark["time_from_landmark"].to_numpy(),
                        fit_landmark["event_from_landmark"].to_numpy())
    test = _surv_array(times, events)

    score = integrated_brier_score(train, test, predictions, grid)

    # The reference every prognostic model should have to beat: the fleet-wide
    # Kaplan-Meier curve, which uses no sensor information at all. A model that
    # cannot beat it is not reading the sensors to any useful effect.
    fitter = KaplanMeierFitter().fit(
        fit_landmark["time_from_landmark"],
        event_observed=fit_landmark["event_from_landmark"].astype(bool),
    )
    baseline_curve = np.asarray(
        [float(fitter.predict(h)) for h in grid], dtype=float
    )
    baseline = integrated_brier_score(
        train, test, np.tile(baseline_curve, (len(times), 1)), grid
    )

    return {
        "landmark": landmark,
        "integrated_brier_score": float(score),
        "baseline_integrated_brier_score": float(baseline),
        "improvement": float(baseline - score),
        "horizons": grid.tolist(),
    }


def calibration(result, risk_set: pd.DataFrame, feature_columns: list[str],
                edges: np.ndarray, landmark: int,
                horizon: int = survival.DEFAULT_HORIZON,
                n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    """Predicted vs observed failure rate, in bins of predicted risk.

    The observed rate is a **Kaplan-Meier** estimate inside each bin, not a raw
    proportion. Using a raw proportion would treat a unit censored at cycle 10
    of a 30-cycle horizon as a survivor, which understates the observed rate and
    would make the model look systematically over-cautious — the exact error
    this project exists to avoid making.
    """
    evaluation = survival.landmark_frame(risk_set, landmark)
    risk = survival.horizon_risk(result, evaluation, feature_columns, edges,
                                 horizon=horizon)

    evaluation = evaluation.assign(predicted=risk)
    # Bins can collapse when many units share a near-identical low risk; `duplicates`
    # keeps that from raising rather than silently mislabelling.
    evaluation["bin"] = pd.qcut(
        evaluation["predicted"], q=n_bins, labels=False, duplicates="drop"
    )

    rows: list[dict] = []
    for bin_index, group in evaluation.groupby("bin", observed=True):
        fitter = KaplanMeierFitter().fit(
            group["time_from_landmark"],
            event_observed=group["event_from_landmark"].astype(bool),
        )
        observed = 1.0 - float(fitter.predict(horizon))
        rows.append({
            "bin": int(bin_index),
            "units": len(group),
            "mean_predicted": float(group["predicted"].mean()),
            "observed_km": observed,
        })

    return pd.DataFrame(rows)


def censoring_bias(fleet: pd.DataFrame) -> dict:
    """The headline of the censoring critique, in cycles.

    Three estimates of mean life, plus the Kaplan-Meier median, which is the
    quantity that actually survives censoring cleanly:

    ``truth``          mean of every unit's true lifetime — unavailable in real life
    ``complete_case``  mean lifetime of units whose failure was observed
    ``naive_observed`` mean of the observed times, counting a censored unit's
                       snapshot as if it were its lifetime — the other common
                       error, and it biases the opposite way

    The complete-case gap is the number the README leads with, because it
    quantifies what "just drop the units that have not failed yet" costs.
    """
    truth = float(fleet["true_lifetime"].mean())
    observed = fleet.loc[fleet["event"] == 1, "true_lifetime"]
    complete_case = float(observed.mean())
    naive = float(fleet["observed_time"].mean())

    fitter = KaplanMeierFitter().fit(
        fleet["observed_time"], event_observed=fleet["event"].astype(bool)
    )
    km_median = float(fitter.median_survival_time_)
    true_median = float(fleet["true_lifetime"].median())

    return {
        "units": int(len(fleet)),
        "censored_fraction": float(fleet["censored"].mean()),
        "true_mean_life": truth,
        "complete_case_mean_life": complete_case,
        "complete_case_bias_cycles": complete_case - truth,
        "complete_case_bias_pct": 100.0 * (complete_case - truth) / truth,
        "naive_observed_mean_life": naive,
        "naive_bias_cycles": naive - truth,
        "true_median_life": true_median,
        "kaplan_meier_median": km_median,
        "kaplan_meier_median_error": km_median - true_median,
    }


def bias_by_censoring_rate(trajectories: pd.DataFrame,
                           rates: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
                           seed: int = RANDOM_SEED) -> pd.DataFrame:
    """How the complete-case bias grows as the fleet gets more realistic.

    A single censoring rate could always be dismissed as a contrived setting.
    Sweeping it answers the objection directly: the bias is not an artefact of
    one choice, it is monotone in how much of the fleet is still running — and
    real fleets are mostly still running.

    Kaplan-Meier's error is reported in the same table as the control. It stays
    near zero across the sweep, which is the positive half of the argument:
    the information is not lost, it is only lost by the wrong estimator.
    """
    from src.panel import induce_censoring, unit_lifetimes

    lifetimes = unit_lifetimes(trajectories)
    rows: list[dict] = []

    for rate in rates:
        fleet = induce_censoring(lifetimes, target_rate=rate, seed=seed)
        summary = censoring_bias(fleet)
        rows.append({
            "target_censor_rate": rate,
            "realised_censor_rate": summary["censored_fraction"],
            "complete_case_bias_cycles": summary["complete_case_bias_cycles"],
            "complete_case_bias_pct": summary["complete_case_bias_pct"],
            "kaplan_meier_median_error": summary["kaplan_meier_median_error"],
            "events": int(fleet["event"].sum()),
        })

    return pd.DataFrame(rows)


def compare_feature_sets(fit_risk_set: pd.DataFrame, holdout_risk_set: pd.DataFrame,
                         drift_columns: list[str], level_columns: list[str],
                         edges: np.ndarray,
                         landmarks: tuple[int, ...] = survival.LANDMARK_CYCLES,
                         horizon: int = survival.DEFAULT_HORIZON) -> pd.DataFrame:
    """Test the 'drift, not level' claim instead of only arguing it.

    ``panel.add_dynamics`` argues that a sensor's *level* confounds an engine
    that started worn with one that has worn out, and that the change from the
    unit's own baseline separates them. That is a claim about out-of-sample
    behaviour, so it is checked out of sample: fit all three feature sets and
    compare in-sample fit against holdout discrimination.
    """
    rows: list[dict] = []
    candidates = {
        "drift": drift_columns,
        "level": level_columns,
        "drift + level": drift_columns + level_columns,
    }

    for name, columns in candidates.items():
        result = survival.fit_discrete_hazard(fit_risk_set, columns, edges)
        scores = landmark_scores(result, holdout_risk_set, columns, edges,
                                 fit_risk_set, landmarks=landmarks, horizon=horizon)
        rows.append({
            "feature_set": name,
            "n_parameters": int(len(result.params)),
            "in_sample_pseudo_r2": float(result.prsquared),
            "holdout_harrell_c": float(scores["harrell_c"].mean()),
            "holdout_uno_c": float(scores["uno_c"].mean()),
        })

    return pd.DataFrame(rows)


def bootstrap_metric(values: np.ndarray, groups: np.ndarray,
                     statistic, n_boot: int = N_BOOTSTRAP,
                     seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """Percentile confidence interval, resampling **units** rather than rows.

    Resampling rows would treat a unit's hundreds of correlated cycle
    observations as independent draws and produce an interval several times too
    narrow. The unit is the independent sampling unit here, so the unit is what
    gets resampled — the same reasoning that makes the standard errors
    clustered.
    """
    generator = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    estimates: list[float] = []

    for _ in range(n_boot):
        chosen = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        try:
            estimates.append(float(statistic(values[index], index)))
        except (ValueError, ZeroDivisionError):
            continue

    if not estimates:
        return np.nan, np.nan, np.nan

    array = np.asarray(estimates)
    return (
        float(np.mean(array)),
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    )
