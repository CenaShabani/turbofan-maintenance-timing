"""Policy arithmetic, checked against answers worked out on paper.

The cost model is where a subtle error would be most expensive and least
visible: it would not crash, it would just report a saving that is not real.
So the assertions here are against hand-computed values, not against the
pipeline's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import policy


def make_paths(lifetimes: dict[str, int], alarm_at: dict[str, int] | None = None,
               risk_high: float = 0.9, risk_low: float = 0.01) -> pd.DataFrame:
    """A risk path with the alarm cycle placed exactly where we choose.

    Risk is ``risk_low`` before the chosen alarm cycle and ``risk_high`` from it
    onward, so any threshold strictly between the two fires on exactly that
    cycle and the expected cost is computable by hand.
    """
    alarm_at = alarm_at or {}
    rows: list[pd.DataFrame] = []

    for unit, lifetime in lifetimes.items():
        cycles = np.arange(1, lifetime + 1)
        alarm = alarm_at.get(unit)
        risk = np.full(len(cycles), risk_low)
        if alarm is not None:
            risk[cycles >= alarm] = risk_high
        rows.append(pd.DataFrame({
            "unit_id": unit,
            "cycle": cycles,
            "true_lifetime": lifetime,
            "predicted_risk": risk,
        }))

    return pd.concat(rows, ignore_index=True)


def test_cost_metric_matches_hand_calculation():
    """Two engines, one pulled and one failed, arithmetic done on paper.

    Engine A: life 200, alarm at 150 -> pulled. cost 1, service 150.
    Engine B: life 100, never alarms -> fails.  cost 5, service 100.
    Total cost 6 over 250 cycles = 24.0 per 1,000 cycles.
    """
    paths = make_paths({"A": 200, "B": 100}, alarm_at={"A": 150})
    result = policy.simulate_threshold(paths, threshold=0.5, cost_ratio=5.0)

    assert result["pulled"] == 1
    assert result["failed"] == 1
    assert result["cost_per_1000_cycles"] == pytest.approx(1000.0 * 6.0 / 250.0)
    assert result["mean_service_cycles"] == pytest.approx(125.0)
    assert result["mean_lead_time"] == pytest.approx(50.0)


def test_an_alarm_at_or_after_failure_is_not_a_save():
    """The easiest way to accidentally report a policy that beats reality."""
    paths = make_paths({"A": 100}, alarm_at={"A": 100})
    result = policy.simulate_threshold(paths, threshold=0.5, cost_ratio=5.0)

    assert result["pulled"] == 0
    assert result["failed"] == 1
    assert result["cost_per_1000_cycles"] == pytest.approx(1000.0 * 5.0 / 100.0)


def test_a_threshold_nothing_reaches_is_run_to_failure():
    paths = make_paths({"A": 200, "B": 150}, alarm_at={"A": 100, "B": 100})
    strict = policy.simulate_threshold(paths, threshold=0.999, cost_ratio=4.0)
    baseline = policy.run_to_failure(paths, cost_ratio=4.0)

    assert strict["failed"] == 2
    assert strict["cost_per_1000_cycles"] == pytest.approx(baseline["cost_per_1000_cycles"])


def test_fixed_interval_optimum_is_the_analytic_one(constant_lifetime_fleet):
    """With every engine failing at 200, the best interval is analytically known.

    For an interval ``I < 200`` nothing fails, so cost is ``1`` per engine over
    ``I`` cycles: ``1000 / I``, strictly decreasing in ``I``. The optimum is
    therefore the **largest grid value below 200**. For ``I >= 200`` every
    engine fails first and the cost jumps to ``1000 * 5 / 200 = 25``.

    This is the test the plan promised: a fixture whose optimum is a
    calculation, not a re-run of the code being tested.
    """
    from src import panel

    prepared = panel.prepare(constant_lifetime_fleet, target_rate=0.3, seed=5)
    evaluation = prepared["evaluation_panel"]
    paths = evaluation.assign(predicted_risk=0.0)

    grid = np.arange(100, 340, 5)
    sweep = policy.sweep_fixed_interval(paths, intervals=grid, cost_ratio=5.0)
    best = sweep.loc[sweep["cost_per_1000_cycles"].idxmin()]

    expected_interval = grid[grid < 200].max()
    assert best["interval"] == expected_interval
    assert best["cost_per_1000_cycles"] == pytest.approx(1000.0 / expected_interval)
    assert best["failed"] == 0

    # And the jump on the other side of the cliff is where it should be.
    beyond = sweep.loc[sweep["interval"] == 200].iloc[0]
    assert beyond["cost_per_1000_cycles"] == pytest.approx(1000.0 * 5.0 / 200.0)


def test_run_to_failure_is_cost_ratio_over_mean_life():
    paths = make_paths({"A": 100, "B": 300})
    result = policy.run_to_failure(paths, cost_ratio=5.0)
    # 2 engines x cost 5 = 10, over 400 cycles.
    assert result["cost_per_1000_cycles"] == pytest.approx(1000.0 * 10.0 / 400.0)
    assert result["failure_rate"] == 1.0


def test_vectorised_matrices_reproduce_the_sweep_exactly(constant_lifetime_fleet):
    """The bootstrap must compute the same thing the point estimate does.

    A vectorised re-implementation that drifts from the loop it replaces is a
    real risk, and it would show up as confidence intervals around the wrong
    number rather than as a crash.
    """
    from src import panel

    prepared = panel.prepare(constant_lifetime_fleet, target_rate=0.3, seed=5)
    generator = np.random.default_rng(0)
    paths = prepared["evaluation_panel"].copy()
    paths["predicted_risk"] = generator.uniform(0, 1, len(paths))

    thresholds = np.linspace(0.05, 0.95, 25)
    loop = policy.sweep_thresholds(paths, thresholds=thresholds)["cost_per_1000_cycles"]
    matrices = policy.unit_outcome_matrices(paths, thresholds=thresholds)
    vectorised = 1000.0 * matrices["cost"].sum(axis=0) / matrices["service"].sum(axis=0)

    np.testing.assert_allclose(loop.to_numpy(), vectorised, rtol=1e-12)


def test_lead_time_distribution_matches_the_simulation():
    paths = make_paths({"A": 200, "B": 300, "C": 100}, alarm_at={"A": 150, "B": 250})
    frame = policy.lead_time_distribution(paths, threshold=0.5)

    leads = frame.set_index("unit_id")["lead_time"]
    assert leads["A"] == pytest.approx(50.0)
    assert leads["B"] == pytest.approx(50.0)
    assert np.isnan(leads["C"])
    assert frame["pulled"].sum() == 2


def test_lower_threshold_never_delays_the_pull():
    """The alarm is a rank cut, so relaxing it can only make engines fire earlier."""
    paths = make_paths({"A": 200, "B": 260}, alarm_at={"A": 150, "B": 200})
    loose = policy.simulate_threshold(paths, threshold=0.005, cost_ratio=5.0)
    tight = policy.simulate_threshold(paths, threshold=0.5, cost_ratio=5.0)
    assert loose["mean_service_cycles"] <= tight["mean_service_cycles"]


def test_bootstrap_interval_brackets_the_point_estimate(constant_lifetime_fleet):
    from src import panel

    prepared = panel.prepare(constant_lifetime_fleet, target_rate=0.3, seed=5)
    generator = np.random.default_rng(1)
    paths = prepared["evaluation_panel"].copy()
    paths["predicted_risk"] = generator.uniform(0, 1, len(paths))

    sweep = policy.sweep_thresholds(paths)
    point = sweep["cost_per_1000_cycles"].min()
    boot = policy.bootstrap_optimum(paths, n_boot=100)

    low, high = boot["cost_ci"]
    assert low <= high
    assert low <= point <= high * 1.5
