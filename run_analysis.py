"""Rebuild every published number and figure, from the raw data, in one command.

    python run_analysis.py

There is exactly one place each number in the README comes from, and this is
it. The capstone notebook calls the same ``src/`` functions rather than
re-implementing anything, so the notebook and the script cannot drift apart —
a figure that exists only in a notebook cell is one nobody can regenerate.

Results land in ``outputs/results`` as JSON and CSV, and figures in
``outputs/figures``. Both directories are safe to delete: this rebuilds them.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate, figures, panel, policy, survival
from src.acquisition import TRAIN_PARQUET

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# Landmarks used for evaluation. Chosen inside the risk window (no engine fails
# before cycle 128) and spaced so the "how far ahead can it see" question is
# asked at three genuinely different fleet ages.
LANDMARKS = (150, 175, 200)
HEADLINE_LANDMARK = 150


class NumpyEncoder(json.JSONEncoder):
    """numpy scalars are not JSON-serialisable; convert rather than lose precision."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _write(name: str, payload) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    else:
        (RESULTS_DIR / f"{name}.json").write_text(
            json.dumps(payload, indent=2, cls=NumpyEncoder), encoding="utf-8"
        )


def main() -> dict:
    # statsmodels warns about the Hessian on the complete-case fit, which is
    # expected and is itself part of the finding. Silenced so the run's output
    # stays readable; every convergence status is reported explicitly below.
    warnings.filterwarnings("ignore")

    started = time.time()

    if not TRAIN_PARQUET.exists():
        raise SystemExit(
            f"{TRAIN_PARQUET} not found. Run `python src/acquisition.py` first."
        )

    trajectories = pd.read_parquet(TRAIN_PARQUET)
    print(f"Loaded {len(trajectories):,} unit-cycles "
          f"from {trajectories.groupby(['subset', 'unit']).ngroups} engines")

    # ---------------------------------------------------------------- panel
    print("\n[1/6] Building the censored fleet and the risk set")
    prepared = panel.prepare(trajectories)
    risk_set = prepared["risk_set"]
    fleet = prepared["fleet"]
    features = prepared["feature_columns"]

    print(f"  {prepared['n_regimes']} operating regimes; "
          f"{len(prepared['sensors_kept'])} sensors kept, "
          f"{len(prepared['sensors_dropped'])} constant and dropped "
          f"({', '.join(prepared['sensors_dropped'])})")
    print(f"  censored fraction {prepared['censoring']['realised_rate']:.1%} "
          f"(target {prepared['censoring']['target_rate']:.0%}, "
          f"horizon ceiling {prepared['censoring']['horizon_ceiling']})")

    fit_full = risk_set.loc[risk_set["split"] == "fit"]
    window_start = survival.risk_window_start(fit_full)
    fit = survival.restrict_to_risk_window(fit_full, window_start)
    holdout = survival.restrict_to_risk_window(
        risk_set.loc[risk_set["split"] == "holdout"], window_start
    )
    edges = survival.derive_band_edges(fit)

    units_before = fit_full["unit_id"].nunique()
    units_after = fit["unit_id"].nunique()
    print(f"  risk window opens at cycle {window_start} (first failure ever seen); "
          f"{units_before - units_after} fit engines censored before it and drop out")

    _write("panel_summary", {
        "engines": int(fleet.shape[0]),
        "unit_cycles": int(len(trajectories)),
        "regimes": prepared["n_regimes"],
        "sensors_kept": prepared["sensors_kept"],
        "sensors_dropped": prepared["sensors_dropped"],
        "censoring": prepared["censoring"],
        "risk_window_start": window_start,
        "fit_engines": int(units_after),
        "fit_engines_dropped_by_window": int(units_before - units_after),
        "holdout_engines": int(holdout["unit_id"].nunique()),
        "hazard_band_edges": edges.tolist(),
        "fit_rows": int(len(fit)),
        "fit_failures": int(fit["failed"].sum()),
    })

    # ------------------------------------------------------- censoring cost
    print("\n[2/6] Measuring what ignoring censoring costs")
    bias = evaluate.censoring_bias(fleet)
    bias_sweep = evaluate.bias_by_censoring_rate(trajectories)
    print(f"  complete-case mean life {bias['complete_case_mean_life']:.1f} vs "
          f"true {bias['true_mean_life']:.1f} cycles "
          f"({bias['complete_case_bias_pct']:+.1f}%)")
    print(f"  Kaplan-Meier median error {bias['kaplan_meier_median_error']:+.0f} cycles")
    _write("censoring_bias", bias)
    _write("censoring_bias_sweep", bias_sweep)

    # --------------------------------------------------------------- models
    print("\n[3/6] Fitting the ladder of models")
    logrank = survival.logrank(fleet)
    curves = survival.kaplan_meier(fleet)
    print(f"  log-rank across subsets: chi2={logrank['test_statistic']:.1f}, "
          f"p={logrank['p_value']:.2e}")

    hazard = survival.fit_discrete_hazard(fit, features, edges)
    print(f"  discrete-time hazard: {len(hazard.params)} parameters, "
          f"{int(fit['failed'].sum())} events, "
          f"converged={hazard.mle_retvals['converged']}, "
          f"pseudo-R2={hazard.prsquared:.3f}")

    complete_case = survival.fit_complete_case(fit, features, edges)
    print(f"  complete-case version fitted on "
          f"{fit.loc[fit['event'] == 1, 'unit_id'].nunique()} of "
          f"{fit['unit_id'].nunique()} engines")

    cox = survival.fit_cox_time_varying(fit, features)
    ph_test = survival.test_proportional_hazards(cox, fit, features)
    violations = int(ph_test["violates_ph"].sum())
    print(f"  Cox PH assumption violated by {violations} of {len(features)} covariates")

    coefficients = survival.tidy(hazard)
    _write("logrank", logrank)
    _write("survival_curves", curves)
    _write("hazard_coefficients", coefficients)
    _write("ph_test", ph_test)

    # ----------------------------------------------------------- evaluation
    print("\n[4/6] Scoring on the holdout")
    scores = evaluate.landmark_scores(hazard, holdout, features, edges, fit,
                                      landmarks=LANDMARKS)
    print(scores.round(4).to_string(index=False))

    feature_comparison = evaluate.compare_feature_sets(
        fit, holdout, features, prepared["level_columns"], edges, landmarks=LANDMARKS
    )
    print("\n  feature-set comparison (the 'drift, not level' claim, tested):")
    print(feature_comparison.round(4).to_string(index=False))

    calibration = evaluate.calibration(hazard, holdout, features, edges, HEADLINE_LANDMARK)
    brier = evaluate.brier(hazard, holdout, features, edges, fit, HEADLINE_LANDMARK)
    auc = evaluate.time_dependent_auc(hazard, holdout, features, edges, fit,
                                      HEADLINE_LANDMARK)
    print(f"\n  integrated Brier {brier['integrated_brier_score']:.4f} vs "
          f"Kaplan-Meier baseline {brier['baseline_integrated_brier_score']:.4f}")

    complete_case_scores = evaluate.landmark_scores(
        complete_case, holdout, features, edges, fit, landmarks=LANDMARKS
    )

    _write("landmark_scores", scores)
    _write("complete_case_scores", complete_case_scores)
    _write("feature_set_comparison", feature_comparison)
    _write("calibration", calibration)
    _write("brier", brier)
    _write("time_dependent_auc", auc)

    # --------------------------------------------------------------- policy
    print("\n[5/6] Turning the probability into a maintenance decision")
    evaluation_panel = prepared["evaluation_panel"]
    paths = policy.score_paths(
        hazard, evaluation_panel.loc[evaluation_panel["split"] == "holdout"],
        features, edges, window_start
    )

    comparison = policy.compare_policies(paths)
    bootstrap = policy.bootstrap_optimum(paths)
    analytic = policy.analytic_threshold_penalty(paths)
    sensitivity = policy.cost_ratio_sensitivity(paths)
    leads = policy.lead_time_distribution(paths, comparison["model"]["threshold"])

    model, interval = comparison["model"], comparison["fixed_interval"]
    print(f"  optimal threshold {model['threshold']:.3f} "
          f"(95% CI {bootstrap['threshold_ci'][0]:.3f}-{bootstrap['threshold_ci'][1]:.3f})")
    print(f"  model policy      : {model['cost_per_1000_cycles']:.2f} per 1,000 cycles, "
          f"{model['failure_rate']:.1%} failures, "
          f"{model['mean_service_cycles']:.0f} cycles service")
    print(f"  best fixed interval: {interval['cost_per_1000_cycles']:.2f} per 1,000 cycles, "
          f"{interval['failure_rate']:.1%} failures, "
          f"{interval['mean_service_cycles']:.0f} cycles service "
          f"(every {interval['interval']:.0f} cycles)")
    print(f"  run to failure     : "
          f"{comparison['run_to_failure']['cost_per_1000_cycles']:.2f} per 1,000 cycles")
    print(f"  SAVING vs best fixed interval: "
          f"{comparison['saving_vs_fixed_interval_pct']:.1f}% "
          f"(95% CI {bootstrap['saving_vs_interval_ci'][0]:.1f}-"
          f"{bootstrap['saving_vs_interval_ci'][1]:.1f}%)")
    print(f"  median lead time: {leads.loc[leads['pulled'], 'lead_time'].median():.0f} cycles")
    print(f"  trusting the uncalibrated probability instead of sweeping would cost "
          f"{analytic['penalty_pct']:.1f}% more")

    _write("policy_comparison", {
        key: value for key, value in comparison.items()
        if key not in {"model_sweep", "interval_sweep"}
    })
    _write("policy_threshold_sweep", comparison["model_sweep"])
    _write("policy_interval_sweep", comparison["interval_sweep"])
    _write("policy_bootstrap", bootstrap)
    _write("analytic_threshold_penalty", analytic)
    _write("cost_ratio_sensitivity", sensitivity)
    _write("lead_time_distribution", leads)

    # -------------------------------------------------------------- figures
    print("\n[6/6] Drawing figures")
    written = [
        figures.degradation_trajectories(risk_set),
        figures.survival_curves(curves),
        figures.censoring_bias(bias_sweep),
        figures.baseline_hazard(coefficients, edges),
        figures.calibration(calibration, HEADLINE_LANDMARK),
        figures.threshold_sweep(comparison["model_sweep"], model, bootstrap,
                                interval["cost_per_1000_cycles"],
                                comparison["run_to_failure"]["cost_per_1000_cycles"]),
        figures.policy_frontier(comparison["model_sweep"], comparison["interval_sweep"], model),
        figures.lead_times(leads, model["threshold"]),
        figures.cost_ratio_sensitivity(sensitivity),
        figures.time_dependent_auc(auc),
    ]
    for path in written:
        print(f"  {path.name}")

    headline = {
        "engines": int(fleet.shape[0]),
        "censored_fraction": prepared["censoring"]["realised_rate"],
        "complete_case_bias_cycles": bias["complete_case_bias_cycles"],
        "complete_case_bias_pct": bias["complete_case_bias_pct"],
        "kaplan_meier_median_error": bias["kaplan_meier_median_error"],
        "holdout_harrell_c": float(scores["harrell_c"].mean()),
        "holdout_uno_c": float(scores["uno_c"].mean()),
        "integrated_brier_score": brier["integrated_brier_score"],
        "baseline_integrated_brier_score": brier["baseline_integrated_brier_score"],
        "optimal_threshold": model["threshold"],
        "optimal_threshold_ci": bootstrap["threshold_ci"],
        "model_cost_per_1000": model["cost_per_1000_cycles"],
        "interval_cost_per_1000": interval["cost_per_1000_cycles"],
        "best_interval_cycles": interval["interval"],
        "run_to_failure_cost_per_1000": comparison["run_to_failure"]["cost_per_1000_cycles"],
        "saving_vs_interval_pct": comparison["saving_vs_fixed_interval_pct"],
        "saving_vs_interval_ci": bootstrap["saving_vs_interval_ci"],
        "model_failure_rate": model["failure_rate"],
        "interval_failure_rate": interval["failure_rate"],
        "model_service_cycles": model["mean_service_cycles"],
        "interval_service_cycles": interval["mean_service_cycles"],
        "median_lead_time": float(leads.loc[leads["pulled"], "lead_time"].median()),
        "lead_time_p10": float(np.percentile(leads.loc[leads["pulled"], "lead_time"], 10)),
        "analytic_threshold_penalty_pct": analytic["penalty_pct"],
        "ph_violations": violations,
        "ph_covariates": len(features),
        "runtime_seconds": round(time.time() - started, 1),
    }
    _write("headline", headline)

    print(f"\nDone in {headline['runtime_seconds']}s. "
          f"Results -> outputs/results, figures -> outputs/figures")
    return headline


if __name__ == "__main__":
    main()
