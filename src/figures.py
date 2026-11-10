"""Figures for the findings.

One function per figure, each taking already-computed results rather than
refitting anything. That keeps the plotting free of analysis: if a number in a
chart disagrees with a number in the README, the cause is in the module that
produced it, not here.

House style is set once in ``apply_style`` — no per-chart colour choices, so
the figures read as one set, and as one set with the other projects in the
portfolio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in a scripted run; must be set before pyplot.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

# One deep tone for the thing being argued for, one muted tone for what it is
# argued against, one warm accent for the decision. Distinguishable in
# greyscale and to the most common forms of colour blindness.
PRIMARY = "#1F3A5F"
MUTED = "#B4B8C5"
ACCENT = "#C4622D"
POSITIVE = "#3E7D5A"
GRID = "#E3E5EA"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#3C4048",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def degradation_trajectories(risk_set: pd.DataFrame, sensor: str = "sensor_11",
                             n_units: int = 60, seed: int = 0) -> Path:
    """Why a survival model is needed at all: wear is gradual and unit-specific.

    Each line is one engine's drift on the most degradation-correlated sensor,
    against its age. They rise, but they rise from different starting points at
    different rates — which is the argument for a model that reads the unit's
    own trajectory rather than one that reads the calendar.
    """
    apply_style()
    generator = np.random.default_rng(seed)
    units = generator.choice(risk_set["unit_id"].unique(), size=n_units, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for unit in units:
        frame = risk_set.loc[risk_set["unit_id"] == unit]
        failed = frame["event"].iloc[0] == 1
        axes[0].plot(frame["cycle"], frame[f"{sensor}_drift"],
                     color=ACCENT if failed else MUTED,
                     alpha=0.55 if failed else 0.45, linewidth=0.9)

    axes[0].set_xlabel("Cycles in service")
    axes[0].set_ylabel(f"{sensor} drift from own baseline (sd)")
    axes[0].set_title("Every engine wears differently")
    axes[0].plot([], [], color=ACCENT, label="failure observed")
    axes[0].plot([], [], color=MUTED, label="still running at snapshot")
    axes[0].legend(loc="upper left")

    # The same signal aligned on remaining life, which is what makes the
    # degradation obvious — and what an operator cannot do, because the
    # x-axis is only knowable in hindsight.
    for unit in units:
        frame = risk_set.loc[risk_set["unit_id"] == unit]
        if frame["event"].iloc[0] != 1:
            continue
        remaining = frame["observed_time"].iloc[0] - frame["cycle"]
        axes[1].plot(-remaining, frame[f"{sensor}_drift"],
                     color=ACCENT, alpha=0.4, linewidth=0.9)

    axes[1].set_xlabel("Cycles until failure")
    axes[1].set_ylabel(f"{sensor} drift from own baseline (sd)")
    axes[1].set_title("Aligned on failure — the signal an operator cannot see")

    return _save(fig, "01_degradation_trajectories.png")


def survival_curves(curves: pd.DataFrame) -> Path:
    """Kaplan-Meier by subset, with confidence bands.

    Included before any model, to establish that the four fleet designs really
    do have different survival — the log-rank test in the same section is what
    licenses conditioning on subset later.
    """
    apply_style()
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    palette = [PRIMARY, ACCENT, POSITIVE, "#8C6BB1"]

    for colour, (group, frame) in zip(palette, curves.groupby("group")):
        axis.step(frame["cycle"], frame["survival"], where="post",
                  color=colour, linewidth=1.8, label=f"{group} (n={frame['units'].iloc[0]})")
        axis.fill_between(frame["cycle"], frame["lower"], frame["upper"],
                          step="post", color=colour, alpha=0.12)

    axis.set_xlabel("Cycles in service")
    axis.set_ylabel("Probability still running")
    axis.set_title("Survival differs by operating condition and fault mode")
    axis.set_ylim(0, 1.02)
    axis.legend(loc="lower left")
    return _save(fig, "02_survival_by_subset.png")


def censoring_bias(bias_sweep: pd.DataFrame) -> Path:
    """The central claim, as a picture: the bias grows with how realistic the fleet is.

    Two series on purpose. The complete-case error walks away from zero as more
    of the fleet is still running; the Kaplan-Meier error stays on it. That
    pairing is the whole argument — the information is not destroyed by
    censoring, it is destroyed by the wrong estimator.
    """
    apply_style()
    fig, axis = plt.subplots(figsize=(7.5, 4.6))

    rate = 100 * bias_sweep["realised_censor_rate"]
    axis.axhline(0, color="#3C4048", linewidth=1.0)
    axis.plot(rate, bias_sweep["complete_case_bias_cycles"], marker="o",
              color=ACCENT, linewidth=2.0,
              label="Complete-case mean life\n(drop unfailed engines)")
    axis.plot(rate, bias_sweep["kaplan_meier_median_error"], marker="s",
              color=PRIMARY, linewidth=2.0,
              label="Kaplan-Meier median error\n(censoring-aware)")

    axis.set_xlabel("Share of the fleet still running at the snapshot (%)")
    axis.set_ylabel("Error vs the known truth (cycles)")
    axis.set_title("Dropping unfailed engines biases life estimates; survival methods do not")
    axis.legend(loc="lower left")
    return _save(fig, "03_censoring_bias.png")


def baseline_hazard(coefficients: pd.DataFrame, edges: np.ndarray) -> Path:
    """The estimated shape of ageing — the thing a linear term would have assumed."""
    apply_style()
    bands = coefficients.loc[coefficients["term"].str.startswith("band_")].copy()
    bands["level"] = bands["term"].str.split("_").str[1].astype(int)
    bands = bands.sort_values("level")
    bands["cycle"] = [edges[min(level - 1, len(edges) - 1)] for level in bands["level"]]

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.axhline(1.0, color="#3C4048", linewidth=1.0)
    axis.plot(bands["cycle"], bands["odds_ratio"], marker="o", color=PRIMARY, linewidth=2.0)
    axis.fill_between(bands["cycle"], bands["or_lower"], bands["or_upper"],
                      color=PRIMARY, alpha=0.15)

    axis.set_yscale("log")
    axis.set_xlabel("Cycles in service (band lower edge)")
    axis.set_ylabel("Odds of failing this cycle\n(relative to youngest band)")
    axis.set_title("The hazard was estimated, not assumed")
    return _save(fig, "04_baseline_hazard.png")


def calibration(frame: pd.DataFrame, landmark: int) -> Path:
    """Predicted against observed — the diagnostic that drives the policy design.

    The departure from the diagonal is the reason ``policy.py`` sweeps the
    threshold instead of computing it. Plotting it prominently rather than
    burying it is the point: it is the project's main weakness and it changes
    what the project does next.
    """
    apply_style()
    fig, axis = plt.subplots(figsize=(6.2, 5.4))

    limit = max(frame["mean_predicted"].max(), frame["observed_km"].max()) * 1.08
    axis.plot([0, limit], [0, limit], color=MUTED, linewidth=1.4, linestyle="--",
              label="perfect calibration")
    axis.scatter(frame["mean_predicted"], frame["observed_km"],
                 s=frame["units"] * 4, color=ACCENT, alpha=0.85, zorder=3,
                 label="risk decile (area = engines)")

    axis.set_xlabel(f"Mean predicted {30}-cycle failure risk")
    axis.set_ylabel("Observed failure rate (Kaplan-Meier)")
    axis.set_title(f"Risk is under-stated: predictions sit below the line (cycle {landmark})")
    axis.legend(loc="upper left")
    return _save(fig, "05_calibration.png")


def threshold_sweep(sweep: pd.DataFrame, optimum: dict, bootstrap: dict,
                    interval_best: float, run_to_failure_cost: float) -> Path:
    """The decision, as one picture: where to put the alarm and what it is worth."""
    apply_style()
    fig, axis = plt.subplots(figsize=(8.0, 4.8))

    axis.plot(sweep["threshold"], sweep["cost_per_1000_cycles"],
              color=PRIMARY, linewidth=2.0, label="Model policy")
    axis.axhline(interval_best, color=ACCENT, linewidth=1.6, linestyle="--",
                 label="Best fixed-interval policy")
    axis.axhline(run_to_failure_cost, color=MUTED, linewidth=1.6, linestyle=":",
                 label="Run to failure")

    low, high = bootstrap["threshold_ci"]
    axis.axvspan(low, high, color=POSITIVE, alpha=0.15,
                 label="95% CI on the optimum")
    axis.axvline(optimum["threshold"], color=POSITIVE, linewidth=1.8)

    axis.set_xscale("log")
    axis.set_xlabel("Alarm threshold (predicted 30-cycle failure risk)")
    axis.set_ylabel("Cost per 1,000 cycles of service")
    axis.set_title("Where to set the alarm")
    axis.legend(loc="upper left")
    return _save(fig, "06_threshold_sweep.png")


def policy_frontier(model_sweep: pd.DataFrame, interval_sweep: pd.DataFrame,
                    optimum: dict) -> Path:
    """Service delivered against failures accepted — the trade the policy is making.

    The clearest single statement of the finding: at any failure rate the
    calendar policy can reach, the condition-based policy delivers more service
    per engine.
    """
    apply_style()
    fig, axis = plt.subplots(figsize=(7.5, 4.8))

    axis.plot(100 * model_sweep["failure_rate"], model_sweep["mean_service_cycles"],
              color=PRIMARY, linewidth=2.0, marker="", label="Model policy")
    axis.plot(100 * interval_sweep["failure_rate"], interval_sweep["mean_service_cycles"],
              color=ACCENT, linewidth=2.0, linestyle="--", label="Fixed-interval policy")
    axis.scatter([100 * optimum["failure_rate"]], [optimum["mean_service_cycles"]],
                 s=110, color=POSITIVE, zorder=4, label="Chosen operating point")

    axis.set_xlabel("In-service failures (% of engines)")
    axis.set_ylabel("Mean cycles of service per engine")
    axis.set_title("Same safety, more service")
    axis.legend(loc="lower right")
    return _save(fig, "07_policy_frontier.png")


def lead_times(distribution: pd.DataFrame, threshold: float) -> Path:
    """How much warning the alarm gives — the operational feasibility check."""
    apply_style()
    pulled = distribution.loc[distribution["pulled"], "lead_time"]

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.hist(pulled, bins=np.arange(0, pulled.max() + 5, 5),
              color=PRIMARY, alpha=0.85, edgecolor="white")
    axis.axvline(pulled.median(), color=ACCENT, linewidth=2.0,
                 label=f"median {pulled.median():.0f} cycles")
    axis.axvline(np.percentile(pulled, 10), color=MUTED, linewidth=2.0, linestyle="--",
                 label=f"10th percentile {np.percentile(pulled, 10):.0f} cycles")

    axis.set_xlabel("Cycles of warning before the engine would have failed")
    axis.set_ylabel("Engines")
    axis.set_title(f"Warning delivered at the chosen threshold ({threshold:.3f})")
    axis.legend()
    return _save(fig, "08_lead_times.png")


def cost_ratio_sensitivity(frame: pd.DataFrame) -> Path:
    """The recommendation as a curve, so a reader can use their own cost ratio."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(frame["cost_ratio"], frame["optimal_threshold"],
                 marker="o", color=PRIMARY, linewidth=2.0)
    axes[0].set_xlabel("Failure cost ÷ planned-removal cost")
    axes[0].set_ylabel("Cost-optimal alarm threshold")
    axes[0].set_title("The threshold barely moves")

    axes[1].plot(frame["cost_ratio"], frame["saving_vs_interval_pct"],
                 marker="o", color=POSITIVE, linewidth=2.0)
    axes[1].set_xlabel("Failure cost ÷ planned-removal cost")
    axes[1].set_ylabel("Saving vs best fixed interval (%)")
    axes[1].set_title("The saving holds across the range")
    axes[1].set_ylim(0, max(40, frame["saving_vs_interval_pct"].max() * 1.15))

    return _save(fig, "09_cost_ratio_sensitivity.png")


def time_dependent_auc(frame: pd.DataFrame) -> Path:
    """How far ahead the model can actually see."""
    apply_style()
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(frame["cycles_ahead"], frame["auc"], marker="o",
              color=PRIMARY, linewidth=2.0)
    axis.axhline(0.5, color=MUTED, linewidth=1.2, linestyle="--", label="no better than chance")
    axis.set_xlabel("Cycles ahead of the landmark")
    axis.set_ylabel("Time-dependent AUC")
    axis.set_ylim(0.4, 1.02)
    axis.set_title("Discrimination decays with how far ahead you ask")
    axis.legend(loc="lower left")
    return _save(fig, "10_time_dependent_auc.png")
