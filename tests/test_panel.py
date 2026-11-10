"""The censoring construction is the project's central claim; these pin it down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import panel
from tests.conftest import make_trajectories


def test_negative_zero_setting_does_not_create_a_phantom_regime():
    """FD001 stores its condition as -0.0 and FD002 as 0.0 — the same regime.

    Without the ``+ 0.0`` normalisation these format to different strings and
    the fleet appears to fly seven operating conditions instead of six. That is
    exactly the bug this guards, because it would silently split every
    within-regime normalisation in two.
    """
    frame = pd.DataFrame({
        "setting_1": [-0.0, 0.0, -0.001, 0.001],
        "setting_2": [-0.0, 0.0, 0.0, -0.0],
        "setting_3": [100.0, 100.0, 100.0, 100.0],
    })
    assert panel.assign_regime(frame).nunique() == 1


def test_regimes_are_recovered_from_settings(small_fleet):
    assert panel.assign_regime(small_fleet).nunique() == 1


def test_constant_sensors_are_screened_out(small_fleet):
    """Only sensors 2 and 3 vary in the fixture; the other 19 are constant."""
    frame = small_fleet.assign(regime=panel.assign_regime(small_fleet))
    kept, dropped = panel.screen_sensors(frame)
    assert set(kept) == {"sensor_2", "sensor_3"}
    assert len(dropped) == 19


def test_calibrate_horizon_hits_the_target_rate():
    """The censored fraction is a chosen property of the design, not an accident."""
    lifetimes = np.arange(100, 400)
    for target in (0.2, 0.4, 0.6):
        ceiling = panel.calibrate_horizon(lifetimes, target)
        achieved = np.mean(np.minimum(lifetimes - 1, ceiling) / ceiling)
        assert abs(achieved - target) < 0.01


def test_calibrate_horizon_is_monotone():
    """A larger horizon must censor less; the bisection relies on it."""
    lifetimes = np.arange(100, 400)
    ceilings = [panel.calibrate_horizon(lifetimes, rate) for rate in (0.7, 0.5, 0.3, 0.1)]
    assert ceilings == sorted(ceilings)


def test_censoring_is_reproducible(small_fleet):
    """Two runs with the same seed must produce the identical fleet."""
    lifetimes = panel.unit_lifetimes(small_fleet)
    first = panel.induce_censoring(lifetimes, seed=7)
    second = panel.induce_censoring(lifetimes, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_censoring_changes_with_the_seed(small_fleet):
    """A seeded design must still be a random one, or it is not a draw at all."""
    lifetimes = panel.unit_lifetimes(small_fleet)
    first = panel.induce_censoring(lifetimes, seed=7)
    second = panel.induce_censoring(lifetimes, seed=8)
    assert not first["horizon"].equals(second["horizon"])


def test_censoring_is_independent_of_lifetime():
    """The mechanism's defining property: the horizon must not see the lifetime.

    Kaplan-Meier and the partial likelihood both require independent censoring.
    Here it holds by construction, and this test is what stops a future
    'improvement' to the draw from quietly breaking it.
    """
    lifetimes = panel.unit_lifetimes(make_trajectories({u: 100 + u for u in range(1, 400)}))
    fleet = panel.induce_censoring(lifetimes, target_rate=0.4, seed=3)
    correlation = np.corrcoef(fleet["horizon"], fleet["true_lifetime"])[0, 1]
    assert abs(correlation) < 0.12


def test_observed_time_and_event_follow_the_definition(small_fleet):
    lifetimes = panel.unit_lifetimes(small_fleet)
    fleet = panel.induce_censoring(lifetimes, seed=1)

    assert (fleet["observed_time"] == np.minimum(fleet["true_lifetime"], fleet["horizon"])).all()
    assert (fleet["event"] == (fleet["true_lifetime"] <= fleet["horizon"]).astype(int)).all()
    # A censored unit's observed time is its horizon; an uncensored one's is its life.
    censored = fleet["event"] == 0
    assert (fleet.loc[censored, "observed_time"] == fleet.loc[censored, "horizon"]).all()
    assert (fleet.loc[~censored, "observed_time"] == fleet.loc[~censored, "true_lifetime"]).all()


def test_split_is_by_unit_not_by_row(small_fleet):
    """A unit must land entirely in one split, or the model memorises the holdout."""
    prepared = panel.prepare(small_fleet)
    per_unit = prepared["risk_set"].groupby("unit_id")["split"].nunique()
    assert (per_unit == 1).all()


def test_risk_set_is_truncated_at_the_snapshot(small_fleet):
    """Cycles after the snapshot were never observed and must not appear."""
    prepared = panel.prepare(small_fleet)
    risk_set = prepared["risk_set"]
    assert (risk_set["cycle"] <= risk_set["observed_time"]).all()


def test_failure_flag_is_set_only_for_observed_failures(small_fleet):
    prepared = panel.prepare(small_fleet)
    risk_set = prepared["risk_set"]

    # Exactly one failed row per unit whose failure was observed, none otherwise.
    per_unit = risk_set.groupby("unit_id").agg(
        fails=("failed", "sum"), event=("event", "first")
    )
    assert (per_unit.loc[per_unit["event"] == 1, "fails"] == 1).all()
    assert (per_unit.loc[per_unit["event"] == 0, "fails"] == 0).all()

    # And it is the last cycle, never an earlier one.
    failures = risk_set.loc[risk_set["failed"] == 1]
    assert (failures["cycle"] == failures["observed_time"]).all()


def test_evaluation_panel_keeps_the_full_trajectory(small_fleet):
    """The policy simulator needs the cycles the model is forbidden to see."""
    prepared = panel.prepare(small_fleet)
    evaluation = prepared["evaluation_panel"]
    lengths = evaluation.groupby("unit_id")["cycle"].max()
    truths = evaluation.groupby("unit_id")["true_lifetime"].first()
    assert (lengths == truths).all()
    assert len(evaluation) >= len(prepared["risk_set"])


def test_drift_features_never_use_the_future(small_fleet):
    """A feature at cycle t must not change when cycles after t are removed.

    The single most consequential kind of leak in a prognostics pipeline, and
    the cheapest to introduce by using a centred rolling window instead of a
    trailing one.

    ``add_dynamics`` is tested in isolation, on an already-normalised frame.
    Running the whole of ``prepare`` twice would not isolate anything: the
    within-regime z-score is computed over whatever rows are present, so
    truncating the input legitimately shifts every value and the comparison
    would measure re-standardisation rather than leakage.
    """
    frame = small_fleet.copy()
    frame["regime"] = panel.assign_regime(frame)
    kept, _ = panel.screen_sensors(frame)
    frame = pd.concat([frame, panel.normalise_within_regime(frame, kept)], axis=1)

    full = panel.add_dynamics(frame, kept)
    full = pd.concat([frame[["unit", "cycle"]], full], axis=1)

    cutoff = 60
    truncated = frame.loc[frame["cycle"] <= cutoff].reset_index(drop=True)
    short = panel.add_dynamics(truncated, kept)
    short = pd.concat([truncated[["unit", "cycle"]], short], axis=1)

    merged = full.merge(short, on=["unit", "cycle"], suffixes=("_full", "_short"))
    assert len(merged) > 0

    for sensor in kept:
        for kind in ("roll", "drift"):
            column = f"{sensor}_{kind}"
            np.testing.assert_allclose(
                merged[f"{column}_full"].to_numpy(),
                merged[f"{column}_short"].to_numpy(),
                atol=1e-12,
                err_msg=f"{column} at cycle <= {cutoff} changed when later cycles were dropped",
            )


def test_prepare_reports_the_realised_censoring_rate(small_fleet):
    prepared = panel.prepare(small_fleet, target_rate=0.4)
    censoring = prepared["censoring"]
    assert censoring["target_rate"] == 0.4
    assert 0.0 < censoring["realised_rate"] < 1.0
    assert censoring["horizon_ceiling"] > 0


def test_target_rate_must_be_a_probability():
    with pytest.raises(ValueError):
        panel.calibrate_horizon(np.arange(100, 200), 1.5)
