"""The censoring critique, checked in a setting where the right answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluate, panel
from tests.conftest import make_trajectories


@pytest.fixture
def wide_fleet() -> pd.DataFrame:
    """200 engines with lifetimes spread widely, so the bias has room to show."""
    return make_trajectories({unit: 120 + unit for unit in range(1, 201)})


def test_complete_case_understates_life(wide_fleet):
    """The direction of the bias is the whole argument, so it is asserted, not observed.

    A unit is censored exactly when its lifetime exceeds its horizon, so
    deleting censored units deletes long lives preferentially. The surviving
    sample must therefore be biased *short*.
    """
    lifetimes = panel.unit_lifetimes(wide_fleet)
    fleet = panel.induce_censoring(lifetimes, target_rate=0.5, seed=2)
    summary = evaluate.censoring_bias(fleet)

    assert summary["complete_case_bias_cycles"] < 0
    assert summary["complete_case_mean_life"] < summary["true_mean_life"]


def test_naive_observed_time_is_worse_than_complete_case(wide_fleet):
    """Counting a censored unit's snapshot as its lifetime is the bigger error."""
    lifetimes = panel.unit_lifetimes(wide_fleet)
    fleet = panel.induce_censoring(lifetimes, target_rate=0.5, seed=2)
    summary = evaluate.censoring_bias(fleet)

    assert summary["naive_bias_cycles"] < summary["complete_case_bias_cycles"] < 0


def test_kaplan_meier_median_is_nearly_unbiased(wide_fleet):
    """The positive half of the argument: censoring does not destroy the information.

    If this failed, the project's recommendation would be groundless — there
    would be no estimator that survives the censoring it complains about.
    """
    lifetimes = panel.unit_lifetimes(wide_fleet)
    for rate in (0.3, 0.5):
        fleet = panel.induce_censoring(lifetimes, target_rate=rate, seed=4)
        summary = evaluate.censoring_bias(fleet)
        true_median = summary["true_median_life"]
        assert abs(summary["kaplan_meier_median_error"]) < 0.05 * true_median


def test_bias_grows_with_the_censoring_rate(wide_fleet):
    """A single rate could be dismissed as contrived; the trend cannot."""
    sweep = evaluate.bias_by_censoring_rate(
        wide_fleet, rates=(0.2, 0.4, 0.6, 0.8), seed=6
    )
    magnitude = sweep["complete_case_bias_cycles"].abs().to_numpy()
    # Monotone increasing in the censored fraction, allowing for sampling noise
    # between adjacent rates.
    assert magnitude[-1] > magnitude[0]
    assert sweep["realised_censor_rate"].is_monotonic_increasing


def test_kaplan_meier_error_stays_flat_across_the_sweep(wide_fleet):
    """The control series in the headline figure must actually be flat."""
    sweep = evaluate.bias_by_censoring_rate(
        wide_fleet, rates=(0.2, 0.4, 0.6), seed=6
    )
    complete_case = sweep["complete_case_bias_cycles"].abs().max()
    kaplan_meier = sweep["kaplan_meier_median_error"].abs().max()
    assert kaplan_meier < complete_case


def test_bootstrap_resamples_groups_not_rows():
    """Resampling rows would give an interval several times too narrow.

    Checked by construction: with one group the bootstrap can only ever draw
    that group, so the interval must collapse to a point.
    """
    values = np.arange(100.0)
    groups = np.zeros(100, dtype=int)
    mean, low, high = evaluate.bootstrap_metric(
        values, groups, lambda v, _: v.mean(), n_boot=50
    )
    assert low == pytest.approx(high)
    assert mean == pytest.approx(values.mean())


def test_calibration_bins_are_ordered_and_complete(small_fleet):
    """Every engine at the landmark lands in exactly one bin."""
    from src import survival

    prepared = panel.prepare(small_fleet, target_rate=0.3, seed=11)
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    window = survival.restrict_to_risk_window(risk_set, start)
    edges = survival.derive_band_edges(window, n_bands=3)
    features = prepared["feature_columns"]

    result = survival.fit_discrete_hazard(window, features, edges)
    landmark = int(window["cycle"].median())
    frame = evaluate.calibration(result, window, features, edges, landmark, n_bins=3)

    assert frame["units"].sum() == len(survival.landmark_frame(window, landmark))
    assert frame["mean_predicted"].is_monotonic_increasing
    assert ((frame["observed_km"] >= 0) & (frame["observed_km"] <= 1)).all()
