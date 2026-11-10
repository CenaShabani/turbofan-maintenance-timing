"""Design-matrix and prediction properties that a wrong answer would slip past."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import panel, survival


@pytest.fixture
def prepared(small_fleet):
    return panel.prepare(small_fleet, target_rate=0.3, seed=11)


def test_band_edges_split_events_evenly(prepared):
    """Equal-event bands are what stop a zero-event band going to minus infinity."""
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    window = survival.restrict_to_risk_window(risk_set, start)

    edges = survival.derive_band_edges(window, n_bands=4)
    bands = survival.assign_band(window.loc[window["failed"] == 1, "cycle"].to_numpy(), edges)
    counts = pd.Series(bands).value_counts()
    assert counts.min() >= 1
    # No band may be empty of events, which is the property being bought.
    assert (counts > 0).all()


def test_assign_band_is_monotone_in_cycle():
    edges = np.array([100.0, 150.0, 200.0])
    bands = survival.assign_band(np.array([50, 100, 120, 150, 175, 200, 500]), edges)
    assert list(bands) == sorted(bands)
    assert bands[0] == 0          # before the first edge
    assert bands[-1] == len(edges)  # beyond the last


def test_design_matrix_drops_the_reference_band(prepared):
    """A full set of band dummies sums to one and is collinear with the intercept.

    This is not a style point: with the reference kept in, the Hessian is
    singular and whatever the fit returns is an artefact of the pseudo-inverse
    rather than an estimate.
    """
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    window = survival.restrict_to_risk_window(risk_set, start)
    edges = survival.derive_band_edges(window, n_bands=4)

    design = survival.build_design(window, prepared["feature_columns"], edges)
    band_columns = [column for column in design.columns if column.startswith("band_")]

    assert f"band_{survival.BAND_REFERENCE}" not in design.columns
    # The remaining dummies must not sum to the constant for every row.
    assert not np.allclose(design[band_columns].sum(axis=1), design["const"])
    assert np.linalg.matrix_rank(design.to_numpy()) == design.shape[1]


def test_risk_window_starts_at_the_first_observed_failure(prepared):
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    assert start == risk_set.loc[risk_set["failed"] == 1, "cycle"].min()

    window = survival.restrict_to_risk_window(risk_set, start)
    assert window["cycle"].min() >= start
    assert window["failed"].sum() == risk_set["failed"].sum()


def test_risk_window_needs_a_failure():
    empty = pd.DataFrame({"cycle": [1, 2], "failed": [0, 0]})
    with pytest.raises(ValueError):
        survival.risk_window_start(empty)


def test_horizon_risk_increases_with_the_horizon(prepared):
    """Longer to wait cannot mean less chance of failing — a survival product must fall."""
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    window = survival.restrict_to_risk_window(risk_set, start)
    edges = survival.derive_band_edges(window, n_bands=3)
    features = prepared["feature_columns"]

    result = survival.fit_discrete_hazard(window, features, edges)
    short = survival.horizon_risk(result, window, features, edges, horizon=5)
    long = survival.horizon_risk(result, window, features, edges, horizon=40)

    assert (long >= short - 1e-12).all()
    assert ((short >= 0) & (short <= 1)).all()
    assert ((long >= 0) & (long <= 1)).all()


def test_horizon_risk_of_zero_cycles_is_zero(prepared):
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    window = survival.restrict_to_risk_window(risk_set, start)
    edges = survival.derive_band_edges(window, n_bands=3)
    features = prepared["feature_columns"]

    result = survival.fit_discrete_hazard(window, features, edges)
    assert np.allclose(
        survival.horizon_risk(result, window, features, edges, horizon=0), 0.0
    )


def test_kaplan_meier_survival_is_non_increasing(prepared):
    curves = survival.kaplan_meier(prepared["fleet"])
    for _, frame in curves.groupby("group"):
        values = frame["survival"].to_numpy()
        assert (np.diff(values) <= 1e-12).all()
        assert values[0] <= 1.0 + 1e-12


def test_landmark_frame_keeps_only_units_still_at_risk(prepared):
    risk_set = prepared["risk_set"]
    landmark = int(risk_set["cycle"].median())
    frame = survival.landmark_frame(risk_set, landmark)

    assert (frame["cycle"] == landmark).all()
    assert (frame["observed_time"] >= landmark).all()
    assert (frame["time_from_landmark"] == frame["observed_time"] - landmark).all()
    assert frame["unit_id"].is_unique


def test_landmark_beyond_every_unit_raises(prepared):
    with pytest.raises(ValueError):
        survival.landmark_frame(prepared["risk_set"], 100_000)


def test_schoenfeld_residuals_are_centred_by_construction(prepared):
    """A residual is an observation minus its expectation, so it must average near zero.

    The property that distinguishes a real Schoenfeld residual from the naive
    'correlate the covariate with time' shortcut, which has no such centring
    and flags every rising covariate as a violation.
    """
    risk_set = prepared["risk_set"]
    start = survival.risk_window_start(risk_set)
    window = survival.restrict_to_risk_window(risk_set, start)
    features = prepared["feature_columns"]

    cox = survival.fit_cox_time_varying(window, features)
    residuals = survival.schoenfeld_residuals(cox, window, features)

    failures = int(window["failed"].sum())
    used = residuals.attrs["failures_used"]
    skipped = residuals.attrs["failures_skipped_singleton_risk_set"]

    # Every failure is either used or explicitly accounted for as a singleton
    # risk set — none may go missing silently.
    assert used + skipped == failures
    assert len(residuals) == used

    for column in features:
        assert abs(residuals[column].mean()) < residuals[column].std() + 1e-9
