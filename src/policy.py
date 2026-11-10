"""When to pull an engine — turning a probability into a decision.

Everything before this module produces a number. This module spends it.

The decision
------------
At each cycle an operator can leave the engine in service or pull it for
planned maintenance. Pulling early is cheap but wastes remaining life. Leaving
it in risks an in-service failure, which costs several times a planned removal
because of the collateral damage, the unscheduled downtime and the disruption
to everything else on the schedule.

So the policy is a single number — an alarm threshold on predicted risk — and
the question is where to put it.

Why cost per cycle of service, and not cost per unit
----------------------------------------------------
This is the choice that makes the whole exercise honest, and getting it wrong
produces a policy that looks brilliant and is useless.

Score a policy on cost per *unit* and the optimum is degenerate: pull every
engine on its first cycle. Nothing ever fails, every removal is the cheap kind,
and the fleet does no work. The metric has to charge for the service given up,
so the policy is scored as

    cost per 1,000 cycles of service delivered
        = 1000 × Σ cost_i / Σ cycles_delivered_i

Pulling early now buys a lower numerator at the price of a smaller denominator,
which is the real trade. The metric cannot be gamed by being conservative.

For each engine exactly one of two things happens:

* **Pulled at cycle p** (the alarm fired while it was still running) — cost
  ``c_pull``, service ``p`` cycles.
* **Never pulled** — it fails at its true lifetime ``T`` — cost ``c_fail``,
  service ``T`` cycles.

Why the threshold is swept and not calculated
----------------------------------------------
With a perfectly calibrated model the answer is analytic: act when the expected
cost of waiting exceeds the cost of acting, which puts the threshold at
``c_pull / c_fail``. That formula is in every textbook and it is wrong here —
not because the arithmetic fails, but because the model's probabilities are not
calibrated. ``evaluate.calibration`` shows them understating risk badly in the
top bins, and the mechanism is known: ``survival.horizon_risk`` freezes the
sensor readings at their current value, so an engine that is degrading fast has
its future understated.

Two responses are available. Recalibrate the probabilities — but the outcome
needed to fit a recalibration is itself censored, so that repairs one bias by
introducing another. Or stop relying on the probabilities being right: sweep
the threshold and pick the one that actually costs least. The threshold is then
a *rank* cut, and the model only has to order engines correctly, which
``evaluate.landmark_scores`` says it does well (Uno's C ≈ 0.87–0.89).

``analytic_threshold_penalty`` measures the difference between the two, which
turns a methodological caveat into a number: what trusting the uncalibrated
probability would have cost.

What the baselines are, and why these two
------------------------------------------
The comparison is not against nothing. It is against the two policies operators
actually run:

* **Run-to-failure** — never pull. The right policy when failure is cheap.
* **Fixed-interval** — pull every engine at a set age, swept across ages so the
  comparison is against the *best* version of it, not a strawman. This is the
  incumbent in most maintenance organisations and the honest thing to beat. A
  condition-based policy that cannot beat a well-chosen calendar is not worth
  the sensors.

Ground truth, stated plainly
-----------------------------
Scoring uses each engine's true lifetime from ``evaluation_panel``, which the
models never see. That counterfactual is what makes a policy measurable rather
than merely arguable, and it exists only because this is simulator output. The
README says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import survival

# Cost of an unplanned in-service failure as a multiple of a planned removal.
# Maintenance-economics studies of aviation and heavy equipment put the ratio
# in the 3-10x range once collateral damage, unscheduled downtime and schedule
# disruption are counted. Five is a mid-range default and nothing rests on it:
# `cost_ratio_sensitivity` sweeps 2-20x, and the recommended threshold is
# reported as a function of the ratio rather than for one assumed value.
DEFAULT_COST_RATIO = 5.0
COST_PULL = 1.0

# Grid of alarm thresholds. Log-spaced because the interesting region is near
# zero — predicted 30-cycle risks are mostly small, so a linear grid would
# spend most of its points where no engine ever sits.
THRESHOLD_GRID = np.unique(np.concatenate([
    np.logspace(-4, -0.3, 60),
    np.linspace(0.5, 0.99, 20),
]))

FIXED_INTERVAL_GRID = np.arange(100, 340, 5)

N_BOOTSTRAP = 400
RANDOM_SEED = 20260801


def score_paths(result, evaluation_panel: pd.DataFrame, feature_columns: list[str],
                edges: np.ndarray, risk_window_start: int,
                horizon: int = survival.DEFAULT_HORIZON) -> pd.DataFrame:
    """Predicted risk at every cycle of every engine's real life.

    Computed once and reused across the whole threshold sweep — the risk path
    does not depend on the threshold, only the decision does, so recomputing it
    per threshold would multiply the work by several hundred for no gain.

    Cycles before the risk window are given zero risk. That is not an
    assumption smuggled in: the model is explicitly conditional on surviving to
    that cycle (see ``survival.restrict_to_risk_window``) and has nothing to say
    before it. No engine fails there either, so the alarm has nothing to catch.
    """
    panel = evaluation_panel.copy()
    in_window = panel["cycle"] >= risk_window_start

    risk = np.zeros(len(panel), dtype=float)
    if in_window.any():
        risk[in_window.to_numpy()] = survival.horizon_risk(
            result, panel.loc[in_window], feature_columns, edges, horizon=horizon
        )
    panel["predicted_risk"] = risk
    return panel


def simulate_threshold(paths: pd.DataFrame, threshold: float,
                       cost_ratio: float = DEFAULT_COST_RATIO,
                       cost_pull: float = COST_PULL) -> dict:
    """Cost of running the fleet with the alarm set at ``threshold``.

    An engine is pulled at the first cycle its predicted risk reaches the
    threshold. If that never happens before its true failure, it fails.
    """
    cost_fail = cost_pull * cost_ratio

    alarms = paths.loc[paths["predicted_risk"] >= threshold]
    first_alarm = alarms.groupby("unit_id")["cycle"].min()

    units = paths.groupby("unit_id")["true_lifetime"].first()
    pull_cycle = first_alarm.reindex(units.index)

    # An alarm at or after the failure cycle is not a save — the engine had
    # already failed. Treating it as one is the single easiest way to
    # accidentally report a policy that beats reality.
    pulled = pull_cycle.notna() & (pull_cycle < units)

    service = np.where(pulled, pull_cycle.fillna(0), units)
    cost = np.where(pulled, cost_pull, cost_fail)
    lead_time = np.where(pulled, units - pull_cycle.fillna(0), np.nan)

    total_service = float(service.sum())
    return {
        "threshold": float(threshold),
        "cost_per_1000_cycles": 1000.0 * float(cost.sum()) / total_service,
        "units": int(len(units)),
        "pulled": int(pulled.sum()),
        "failed": int((~pulled).sum()),
        "failure_rate": float((~pulled).mean()),
        "mean_service_cycles": float(service.mean()),
        "service_utilisation": total_service / float(units.sum()),
        "mean_lead_time": float(np.nanmean(lead_time)) if pulled.any() else np.nan,
        "median_lead_time": float(np.nanmedian(lead_time)) if pulled.any() else np.nan,
    }


def sweep_thresholds(paths: pd.DataFrame,
                     thresholds: np.ndarray = THRESHOLD_GRID,
                     cost_ratio: float = DEFAULT_COST_RATIO) -> pd.DataFrame:
    """Cost across the whole grid, so the optimum is located rather than assumed."""
    return pd.DataFrame([
        simulate_threshold(paths, threshold, cost_ratio=cost_ratio)
        for threshold in thresholds
    ])


def simulate_fixed_interval(paths: pd.DataFrame, interval: int,
                            cost_ratio: float = DEFAULT_COST_RATIO,
                            cost_pull: float = COST_PULL) -> dict:
    """The incumbent policy: pull every engine at a fixed age, sensors ignored.

    Scored on exactly the same cost metric as the model policy, so the
    comparison is like for like. An engine that fails before the interval fails;
    everything else is pulled on schedule.
    """
    cost_fail = cost_pull * cost_ratio
    units = paths.groupby("unit_id")["true_lifetime"].first()

    pulled = units > interval
    service = np.where(pulled, interval, units)
    cost = np.where(pulled, cost_pull, cost_fail)

    return {
        "interval": int(interval),
        "cost_per_1000_cycles": 1000.0 * float(cost.sum()) / float(service.sum()),
        "units": int(len(units)),
        "pulled": int(pulled.sum()),
        "failed": int((~pulled).sum()),
        "failure_rate": float((~pulled).mean()),
        "mean_service_cycles": float(service.mean()),
        "mean_lead_time": float((units[pulled] - interval).mean()) if pulled.any() else np.nan,
    }


def sweep_fixed_interval(paths: pd.DataFrame,
                         intervals: np.ndarray = FIXED_INTERVAL_GRID,
                         cost_ratio: float = DEFAULT_COST_RATIO) -> pd.DataFrame:
    """Sweep the calendar policy, so the model is compared against its best form."""
    return pd.DataFrame([
        simulate_fixed_interval(paths, int(interval), cost_ratio=cost_ratio)
        for interval in intervals
    ])


def run_to_failure(paths: pd.DataFrame,
                   cost_ratio: float = DEFAULT_COST_RATIO,
                   cost_pull: float = COST_PULL) -> dict:
    """Never pull. The floor the other policies have to be worth beating."""
    units = paths.groupby("unit_id")["true_lifetime"].first()
    return {
        "policy": "run-to-failure",
        "cost_per_1000_cycles": 1000.0 * cost_pull * cost_ratio * len(units) / float(units.sum()),
        "units": int(len(units)),
        "failed": int(len(units)),
        "failure_rate": 1.0,
        "mean_service_cycles": float(units.mean()),
    }


def compare_policies(paths: pd.DataFrame,
                     cost_ratio: float = DEFAULT_COST_RATIO) -> dict:
    """The headline table: best model policy against the best of each baseline."""
    model_sweep = sweep_thresholds(paths, cost_ratio=cost_ratio)
    interval_sweep = sweep_fixed_interval(paths, cost_ratio=cost_ratio)

    best_model = model_sweep.loc[model_sweep["cost_per_1000_cycles"].idxmin()]
    best_interval = interval_sweep.loc[interval_sweep["cost_per_1000_cycles"].idxmin()]
    baseline = run_to_failure(paths, cost_ratio=cost_ratio)

    return {
        "cost_ratio": cost_ratio,
        "model": best_model.to_dict(),
        "fixed_interval": best_interval.to_dict(),
        "run_to_failure": baseline,
        "saving_vs_fixed_interval_pct": 100.0 * (
            best_interval["cost_per_1000_cycles"] - best_model["cost_per_1000_cycles"]
        ) / best_interval["cost_per_1000_cycles"],
        "saving_vs_run_to_failure_pct": 100.0 * (
            baseline["cost_per_1000_cycles"] - best_model["cost_per_1000_cycles"]
        ) / baseline["cost_per_1000_cycles"],
        "model_sweep": model_sweep,
        "interval_sweep": interval_sweep,
    }


def analytic_threshold_penalty(paths: pd.DataFrame,
                               cost_ratio: float = DEFAULT_COST_RATIO) -> dict:
    """What trusting the uncalibrated probabilities would have cost.

    The textbook threshold for a calibrated model is ``c_pull / c_fail``, i.e.
    ``1 / cost_ratio``. Because the model's probabilities understate risk, that
    threshold fires too late. This puts a number on the gap instead of leaving
    "the model is not well calibrated" as an unquantified caveat — which is the
    difference between naming a limitation and measuring one.
    """
    analytic = 1.0 / cost_ratio
    at_analytic = simulate_threshold(paths, analytic, cost_ratio=cost_ratio)

    sweep = sweep_thresholds(paths, cost_ratio=cost_ratio)
    best = sweep.loc[sweep["cost_per_1000_cycles"].idxmin()]

    return {
        "analytic_threshold": analytic,
        "analytic_cost": at_analytic["cost_per_1000_cycles"],
        "empirical_threshold": float(best["threshold"]),
        "empirical_cost": float(best["cost_per_1000_cycles"]),
        "penalty_pct": 100.0 * (
            at_analytic["cost_per_1000_cycles"] - best["cost_per_1000_cycles"]
        ) / best["cost_per_1000_cycles"],
        "analytic_failure_rate": at_analytic["failure_rate"],
        "empirical_failure_rate": float(best["failure_rate"]),
    }


def cost_ratio_sensitivity(paths: pd.DataFrame,
                           ratios: tuple[float, ...] = (2, 3, 5, 8, 12, 20)) -> pd.DataFrame:
    """The recommended threshold as a function of the cost ratio.

    The ratio of failure cost to removal cost is the one input a maintenance
    organisation genuinely knows better than this analysis does, and it varies
    by fleet and by contract. Reporting one optimum for one assumed ratio would
    hand over a number that quietly depends on a guess. Reporting the curve
    hands over something usable: read off your own ratio, take the threshold.
    """
    rows: list[dict] = []
    for ratio in ratios:
        comparison = compare_policies(paths, cost_ratio=float(ratio))
        rows.append({
            "cost_ratio": float(ratio),
            "optimal_threshold": comparison["model"]["threshold"],
            "model_cost": comparison["model"]["cost_per_1000_cycles"],
            "model_failure_rate": comparison["model"]["failure_rate"],
            "model_mean_lead_time": comparison["model"]["mean_lead_time"],
            "best_interval": comparison["fixed_interval"]["interval"],
            "interval_cost": comparison["fixed_interval"]["cost_per_1000_cycles"],
            "run_to_failure_cost": comparison["run_to_failure"]["cost_per_1000_cycles"],
            "saving_vs_interval_pct": comparison["saving_vs_fixed_interval_pct"],
        })
    return pd.DataFrame(rows)


def lead_time_distribution(paths: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """How much warning the alarm actually gives, engine by engine.

    The number that decides whether any of this is operationally usable. A
    policy with a superb cost curve and a two-cycle median lead time cannot be
    acted on — you cannot get a shop slot, a part and a crew inside two flights.
    Reporting the distribution rather than the mean matters because the left
    tail is the operational risk: the mean is comfortable precisely when a
    minority of engines give almost no notice.
    """
    alarms = paths.loc[paths["predicted_risk"] >= threshold]
    first_alarm = alarms.groupby("unit_id")["cycle"].min()
    units = paths.groupby("unit_id")["true_lifetime"].first()

    frame = pd.DataFrame({"true_lifetime": units, "alarm_cycle": first_alarm.reindex(units.index)})
    frame["pulled"] = frame["alarm_cycle"].notna() & (frame["alarm_cycle"] < frame["true_lifetime"])
    frame["lead_time"] = np.where(
        frame["pulled"], frame["true_lifetime"] - frame["alarm_cycle"], np.nan
    )
    return frame.reset_index()


def unit_outcome_matrices(paths: pd.DataFrame,
                          thresholds: np.ndarray = THRESHOLD_GRID,
                          intervals: np.ndarray = FIXED_INTERVAL_GRID,
                          cost_ratio: float = DEFAULT_COST_RATIO,
                          cost_pull: float = COST_PULL) -> dict:
    """Per-engine service and cost under every threshold and every interval.

    Precomputation for the bootstrap. Each engine's outcome under a given
    threshold does not depend on which other engines are in the sample, so the
    whole ``engines × thresholds`` grid can be built once and every bootstrap
    replicate becomes two column sums over a resampled row index. That turns a
    ten-minute loop into about a second, and — more importantly — it makes each
    replicate exactly the same computation as the point estimate rather than a
    re-implementation of it that could drift.
    """
    lifetimes = paths.groupby("unit_id")["true_lifetime"].first().sort_index()
    unit_index = {unit: position for position, unit in enumerate(lifetimes.index)}
    n_units = len(lifetimes)
    lives = lifetimes.to_numpy(dtype=float)
    cost_fail = cost_pull * cost_ratio

    # First cycle at which each engine's risk reaches each threshold. Computed
    # by sorting risks descending once: an engine alarms at threshold t exactly
    # when its running maximum risk first reaches t.
    alarm = np.full((n_units, len(thresholds)), np.inf)
    for unit, frame in paths.groupby("unit_id"):
        row = unit_index[unit]
        cycles = frame["cycle"].to_numpy(dtype=float)
        running_max = np.maximum.accumulate(frame["predicted_risk"].to_numpy(dtype=float))
        # searchsorted on the monotone running maximum finds, for each
        # threshold, the first cycle at which it is reached.
        positions = np.searchsorted(running_max, thresholds, side="left")
        reached = positions < len(cycles)
        alarm[row, reached] = cycles[positions[reached]]

    pulled = alarm < lives[:, None]
    service = np.where(pulled, alarm, lives[:, None])
    cost = np.where(pulled, cost_pull, cost_fail)

    interval_grid = np.asarray(intervals, dtype=float)
    interval_pulled = lives[:, None] > interval_grid[None, :]
    interval_service = np.where(interval_pulled, interval_grid[None, :], lives[:, None])
    interval_cost = np.where(interval_pulled, cost_pull, cost_fail)

    return {
        "unit_ids": lifetimes.index.to_numpy(),
        "true_lifetime": lives,
        "thresholds": np.asarray(thresholds, dtype=float),
        "service": service,
        "cost": cost,
        "intervals": interval_grid,
        "interval_service": interval_service,
        "interval_cost": interval_cost,
    }


def bootstrap_optimum(paths: pd.DataFrame, cost_ratio: float = DEFAULT_COST_RATIO,
                      n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> dict:
    """Confidence band on the optimal threshold and its cost, resampling engines.

    A single optimum read off a single sweep is a point estimate with no
    interval, and on a couple of hundred holdout engines the sweep is not
    smooth enough to take its argmin at face value. Resampling **engines** —
    the independent unit, not the row — gives the interval, and its width is
    the honest statement of how precisely the threshold is actually known.
    """
    matrices = unit_outcome_matrices(paths, cost_ratio=cost_ratio)
    service, cost = matrices["service"], matrices["cost"]
    interval_service, interval_cost = matrices["interval_service"], matrices["interval_cost"]
    grid = matrices["thresholds"]
    n_units = len(matrices["true_lifetime"])

    generator = np.random.default_rng(seed)
    thresholds: list[float] = []
    costs: list[float] = []
    savings: list[float] = []

    for _ in range(n_boot):
        index = generator.integers(0, n_units, size=n_units)
        curve = 1000.0 * cost[index].sum(axis=0) / service[index].sum(axis=0)
        best = int(np.argmin(curve))

        interval_curve = (
            1000.0 * interval_cost[index].sum(axis=0) / interval_service[index].sum(axis=0)
        )
        best_interval = float(interval_curve.min())

        thresholds.append(float(grid[best]))
        costs.append(float(curve[best]))
        savings.append(100.0 * (best_interval - curve[best]) / best_interval)

    return {
        "n_bootstrap": n_boot,
        "threshold_median": float(np.median(thresholds)),
        "threshold_ci": (float(np.percentile(thresholds, 2.5)),
                         float(np.percentile(thresholds, 97.5))),
        "cost_median": float(np.median(costs)),
        "cost_ci": (float(np.percentile(costs, 2.5)),
                    float(np.percentile(costs, 97.5))),
        "saving_vs_interval_median": float(np.median(savings)),
        "saving_vs_interval_ci": (float(np.percentile(savings, 2.5)),
                                  float(np.percentile(savings, 97.5))),
    }
