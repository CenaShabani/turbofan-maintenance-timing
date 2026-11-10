"""Turn run-to-failure trajectories into a censored fleet, then into a risk set.

This module carries the methodological argument of the whole project. It does
two things: it *removes* information on purpose, and it reshapes what is left
into the counting-process format survival models need.

Why remove information on purpose
---------------------------------
Almost every published C-MAPSS notebook does the same thing: regress remaining
useful life on the sensor readings, report RMSE against the benchmark's answer
key. That framing quietly assumes you can see every engine's failure. An
operator never can. At any moment the fleet is mostly units that have *not*
failed yet — and those are exactly the units a maintenance decision is about.
A method that needs a completed lifetime to learn from cannot learn from the
fleet you actually have.

So this module builds the fleet an operator would actually see. Each unit is
observed from its first cycle up to a snapshot horizon drawn independently of
how long it was going to last. If it failed before the snapshot, the failure
is observed. If not, all that is known is that it survived that long: the
observation is **right-censored**.

That costs information, and the cost is the point. Because the untruncated
trajectories still exist in ``data/raw``, the analysis can do something a real
fleet snapshot never permits — compare three things side by side:

1. the **known truth**, estimated on the full run-to-failure data;
2. the **complete-case** estimate, which drops censored units (what "regress on
   RUL" effectively does, because a unit with no observed failure has no
   target);
3. the **censoring-aware** estimate, which keeps them.

The complete-case estimate is not merely less precise. It is *biased*, and
biased in a direction that is easy to state: a unit is censored precisely when
its lifetime exceeds its horizon, so the units that survive the filter are
disproportionately the short-lived ones. Dropping censored units therefore
throws away the long lives specifically, and the fleet looks more fragile than
it is. ``evaluate.py`` measures how much.

The censoring mechanism, stated so it can be argued with
--------------------------------------------------------
Unit *i* has a true failure cycle ``T_i`` (known from the raw data, hidden from
every model). It is assigned a snapshot horizon ``C_i`` drawn uniformly from
``{1, …, U}``, independently of ``T_i``. The unit is then observed for
``Y_i = min(T_i, C_i)`` cycles, with event indicator ``δ_i = 1{T_i ≤ C_i}``.

Three properties this buys, each of which matters:

* **Independent censoring holds by construction.** ``C_i`` is drawn from a
  distribution that does not see ``T_i`` or any sensor reading. Kaplan-Meier
  and partial likelihood both require this, and here it is true by design
  rather than hoped for. (Informative censoring — units pulled *because* they
  looked unwell — is a real phenomenon in maintenance data and would break
  these estimators. It is out of scope and named in the README's limitations.)
* **It is administrative, not left-truncating.** The unit is watched from cycle
  1, so every unit's full history up to the snapshot is available. There is no
  delayed entry to handle.
* **It is reproducible.** ``U`` is solved for deterministically to hit a target
  censored fraction, and horizons are drawn from a seeded generator. Two runs
  produce the same fleet.

``U`` is *calibrated*, not guessed: given the lifetimes, the censored fraction
is a monotone decreasing function of ``U``, so a bisection finds the ``U`` that
hits the target rate. The default target is 40%, and ``run_analysis.py`` sweeps
it (20/40/60%) so the conclusions are not an artefact of one setting.

Operating regime, and why the sensors are normalised inside it
--------------------------------------------------------------
FD002 and FD004 fly six operating conditions; FD001 and FD003 fly one. In the
six-condition subsets the raw sensor variance is dominated by *which condition
the engine is in*, not by how worn it is — sensor 7's raw standard deviation is
0.9 in FD001 and 146 in FD002, and that hundredfold difference is altitude and
throttle, not degradation.

Feeding raw sensors to a model across subsets would therefore let it learn the
flight profile and call it wear. Each sensor is standardised **within operating
regime** instead, so a reading means "unusual for this operating point".

The six regimes are recovered by rounding the three operational settings, not
by clustering. The settings fall into six tight, well-separated groups (checked
in notebook 02), so rounding recovers them exactly and — unlike k-means — is
deterministic, needs no seed, and can be read off in one line. Adding ``+ 0.0``
before formatting is not cosmetic: FD001 stores its setting as ``-0.0``, which
would otherwise label the same physical condition differently from FD002's
``0.0`` and produce a phantom seventh regime.

Sensor screening
----------------
Four sensors (1, 5, 18, 19) are exactly constant within every regime, and two
more (10, 16) are near-constant. Constant columns carry no information and make
the design matrix singular. They are dropped by a variance rule applied to the
data rather than by a hard-coded list, so the screen is reproducible on any
subset selection — and the dropped set is reported rather than silently
removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 20260801

# Target share of units still running at the snapshot. 40% is a plausible
# steady-state fleet: enough failures to estimate from, enough survivors that
# ignoring them is clearly wrong. Swept in run_analysis.py.
DEFAULT_CENSOR_RATE = 0.40

# A sensor whose within-regime standard deviation is below this everywhere is
# treated as constant. Set well below the smallest genuinely-varying sensor
# (sensor 16, ~6e-4) and well above exact zero, so the rule separates the two
# groups cleanly instead of splitting a continuum.
MIN_WITHIN_REGIME_STD = 1e-6

# Trailing window for the smoothed sensor level. C-MAPSS sensor noise is large
# relative to the per-cycle degradation increment, so a single reading is a
# poor state estimate. Five cycles is short enough to stay responsive and long
# enough to halve the noise. Strictly trailing: a feature at cycle t may never
# use a reading from cycle t+1, or the model would be predicting failure from
# evidence collected after the decision it is meant to inform.
ROLLING_WINDOW = 5

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]

HOLDOUT_FRACTION = 0.30


def assign_regime(frame: pd.DataFrame) -> pd.Series:
    """Label each row with its operating regime.

    Rounding rather than clustering: the settings form six tight groups, so
    this recovers them exactly and deterministically. ``+ 0.0`` normalises
    negative zero, which otherwise splits one physical condition into two.
    """
    s1 = (frame["setting_1"].round(0) + 0.0).astype(int)
    s2 = (frame["setting_2"].round(2) + 0.0).round(2)
    s3 = (frame["setting_3"].round(0) + 0.0).astype(int)
    return pd.Series(
        [f"{a}|{b:.2f}|{c}" for a, b, c in zip(s1, s2, s3)],
        index=frame.index,
        name="regime",
        dtype="string",
    )


def screen_sensors(frame: pd.DataFrame,
                   min_std: float = MIN_WITHIN_REGIME_STD) -> tuple[list[str], list[str]]:
    """Split sensors into those that vary within regime and those that do not.

    Returns ``(kept, dropped)``. Applied to the data rather than hard-coded, so
    the screen stays correct if the subset selection changes.
    """
    within_std = frame.groupby("regime", observed=True)[SENSOR_COLUMNS].std()
    varies = (within_std.fillna(0.0) > min_std).any(axis=0)
    kept = [column for column in SENSOR_COLUMNS if varies.get(column, False)]
    dropped = [column for column in SENSOR_COLUMNS if column not in kept]
    return kept, dropped


def normalise_within_regime(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Standardise each sensor against its own operating regime.

    A value of +2 then means "two standard deviations high *for this operating
    point*", which is comparable across a six-condition fleet and a
    single-condition one. Without this, a model pooling the subsets would learn
    the flight profile and mistake it for wear.
    """
    grouped = frame.groupby("regime", observed=True)[sensors]
    centred = frame[sensors] - grouped.transform("mean")
    spread = grouped.transform("std").replace(0.0, np.nan)
    return (centred / spread).fillna(0.0).add_suffix("_z")


def add_dynamics(frame: pd.DataFrame, sensors: list[str],
                 window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add the trailing state estimate and the drift away from the unit's own start.

    Two derived features per sensor, both strictly backward-looking:

    * ``_roll`` — trailing mean over ``window`` cycles. The state estimate the
      raw reading is too noisy to be.
    * ``_drift`` — that trailing mean minus the unit's own first-``window``
      average. Engines enter service with different, unstated initial wear, so
      the *level* of a sensor confounds "born worn" with "worn out". The change
      since the unit's own baseline separates them, and it is the quantity a
      maintenance engineer actually watches.

    Both are computed per unit and use only cycles at or before the current
    one, so a feature never sees the future.
    """
    z_columns = [f"{sensor}_z" for sensor in sensors]
    grouped = frame.groupby(["subset", "unit"], observed=True)

    rolled = (
        grouped[z_columns]
        .transform(lambda column: column.rolling(window, min_periods=1).mean())
        .add_suffix("_roll")
    )
    rolled.columns = [column.replace("_z_roll", "_roll") for column in rolled.columns]

    baseline = (
        grouped[z_columns]
        .transform(lambda column: column.head(window).mean())
    )
    drift = (rolled.to_numpy() - baseline.to_numpy())
    drift = pd.DataFrame(
        drift,
        index=frame.index,
        columns=[column.replace("_z", "_drift") for column in z_columns],
    )
    return pd.concat([rolled, drift], axis=1)


def unit_lifetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """The true failure cycle of every unit — the ground truth the models never see."""
    lifetimes = (
        frame.groupby(["subset", "unit"], observed=True)["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "true_lifetime"})
    )
    return lifetimes.sort_values(["subset", "unit"]).reset_index(drop=True)


def calibrate_horizon(lifetimes: np.ndarray, target_rate: float) -> int:
    """Find the horizon ceiling ``U`` that produces the target censored fraction.

    With ``C ~ Uniform{1..U}`` drawn independently of ``T``, a unit is censored
    exactly when ``C < T``, so

        P(censored) = mean_i  min(T_i - 1, U) / U

    which decreases monotonically in ``U``. Bisection therefore finds the
    unique ``U`` hitting the target, and the censored fraction becomes a
    *chosen* property of the design rather than whatever happened to fall out
    of an arbitrary constant.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be strictly between 0 and 1, got {target_rate}")

    def expected_rate(ceiling: int) -> float:
        return float(np.mean(np.minimum(lifetimes - 1, ceiling) / ceiling))

    low, high = 1, int(lifetimes.max())
    # Grow the ceiling until the achievable rate brackets the target. A very
    # low target needs a horizon well beyond the longest lifetime.
    while expected_rate(high) > target_rate:
        high *= 2
        if high > 10_000 * int(lifetimes.max()):
            raise RuntimeError(f"Could not reach a censored fraction of {target_rate}.")

    while low < high:
        middle = (low + high) // 2
        if expected_rate(middle) > target_rate:
            low = middle + 1
        else:
            high = middle
    return int(low)


def induce_censoring(lifetimes: pd.DataFrame,
                     target_rate: float = DEFAULT_CENSOR_RATE,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Draw a snapshot horizon per unit and record what would have been observed.

    Adds four columns to the lifetimes table:

    ``horizon``        the snapshot cycle ``C_i``, drawn independently of ``T_i``
    ``observed_time``  ``Y_i = min(T_i, C_i)`` — how long the unit was watched
    ``event``          ``1`` if the failure was seen, ``0`` if still running
    ``censored``       the complement, kept for readability at call sites

    ``true_lifetime`` stays in the frame deliberately. It is the ground truth
    used to score the three estimates against each other, and every function
    that fits a model takes the censored columns only — never this one.
    """
    result = lifetimes.copy()
    ceiling = calibrate_horizon(result["true_lifetime"].to_numpy(), target_rate)

    generator = np.random.default_rng(seed)
    result["horizon"] = generator.integers(1, ceiling + 1, size=len(result))
    result["observed_time"] = np.minimum(result["true_lifetime"], result["horizon"])
    result["event"] = (result["true_lifetime"] <= result["horizon"]).astype(int)
    result["censored"] = 1 - result["event"]
    result.attrs["horizon_ceiling"] = ceiling
    result.attrs["target_censor_rate"] = target_rate
    result.attrs["realised_censor_rate"] = float(result["censored"].mean())
    result.attrs["seed"] = seed
    return result


def split_units(lifetimes: pd.DataFrame,
                holdout_fraction: float = HOLDOUT_FRACTION,
                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Assign each unit to fit or holdout, stratified by subset.

    The split is by **unit**, never by row: a unit contributes hundreds of
    correlated cycle observations, and splitting rows would put a unit's early
    cycles in the fit set and its late cycles in the holdout, letting the model
    memorise the engine it is about to be scored on.

    Stratifying by subset keeps the four operating-condition/fault-mode designs
    represented in both halves, so holdout performance is not dominated by
    whichever regime happened to land there.
    """
    generator = np.random.default_rng(seed + 1)
    result = lifetimes.copy()
    result["split"] = "fit"

    for subset, group in result.groupby("subset", observed=True):
        n_holdout = int(round(len(group) * holdout_fraction))
        chosen = generator.choice(group.index.to_numpy(), size=n_holdout, replace=False)
        result.loc[chosen, "split"] = "holdout"

    return result


def build_risk_set(trajectories: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """Assemble the unit-cycle risk set the models are fitted on.

    One row per unit per cycle it was *at risk* — that is, cycles 1 through
    ``observed_time``. Cycles after the snapshot are dropped because they were
    never seen; that is what censoring means, and keeping them would be the
    leak the whole exercise is designed to avoid.

    The outcome ``failed`` is 1 only on a unit's final row *and* only if the
    failure was observed. For a censored unit every row is 0: it never failed
    within the window, which is all the data says.

    The baseline-hazard banding is deliberately *not* built here. Its cut
    points must be derived from the fit set alone, or they leak the holdout's
    failure times into the design, so they belong with the model —
    ``survival.derive_band_edges``.
    """
    fleet_columns = ["subset", "unit", "observed_time", "event", "true_lifetime", "split"]
    merged = trajectories.merge(
        fleet.loc[:, [column for column in fleet_columns if column in fleet.columns]],
        on=["subset", "unit"],
        how="inner",
        validate="many_to_one",
    )

    at_risk = merged.loc[merged["cycle"] <= merged["observed_time"]].copy()

    is_last_cycle = at_risk["cycle"] == at_risk["observed_time"]
    at_risk["failed"] = (is_last_cycle & (at_risk["event"] == 1)).astype(int)

    # Counting-process interval for this cycle: (entry, exit].
    at_risk["entry"] = at_risk["cycle"] - 1
    at_risk["exit"] = at_risk["cycle"]

    # A stable per-unit key. The raw unit numbers restart at 1 in each subset,
    # so clustering or grouping on `unit` alone would silently merge four
    # different engines into one.
    at_risk["unit_id"] = at_risk["subset"].astype(str) + "_" + at_risk["unit"].astype(str)

    return at_risk.sort_values(["subset", "unit", "cycle"]).reset_index(drop=True)


def build_evaluation_panel(trajectories: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """Full, untruncated trajectories — for scoring policies, never for fitting.

    A maintenance policy has to be scored on what *would have happened*: if the
    alarm fires at cycle 190 and the engine would have run to 240, the policy
    bought 50 cycles of avoided failure and gave up 50 cycles of service. That
    counterfactual needs the whole trajectory, including the part after the
    snapshot that the models are forbidden to see.

    Hence two panels, and the separation between them is load-bearing:

    * ``risk_set`` — truncated at the snapshot. Everything that fits a model
      uses this and only this.
    * ``evaluation_panel`` — the complete run to failure. Used only to simulate
      what a policy would have cost.

    Being able to do this at all is the reason a simulator benchmark is worth
    using. On a real fleet the counterfactual is unobservable and a policy can
    only be argued for, never measured. The price is that the *evaluation* is
    as realistic as C-MAPSS is, which the README states plainly.
    """
    fleet_columns = ["subset", "unit", "observed_time", "event", "true_lifetime", "split"]
    panel = trajectories.merge(
        fleet.loc[:, [column for column in fleet_columns if column in fleet.columns]],
        on=["subset", "unit"],
        how="inner",
        validate="many_to_one",
    )
    panel["unit_id"] = panel["subset"].astype(str) + "_" + panel["unit"].astype(str)
    return panel.sort_values(["subset", "unit", "cycle"]).reset_index(drop=True)


def prepare(trajectories: pd.DataFrame,
            target_rate: float = DEFAULT_CENSOR_RATE,
            seed: int = RANDOM_SEED) -> dict:
    """Run the whole construction: regimes, features, censoring, split, risk set.

    Returns a dict with the risk set, the fleet table, the sensors kept and
    dropped, and the censoring design — everything the notebooks and
    ``run_analysis.py`` need, assembled in one place so the capstone notebook
    and the script can never build it differently.
    """
    frame = trajectories.copy()
    frame["regime"] = assign_regime(frame)

    kept, dropped = screen_sensors(frame)
    normalised = normalise_within_regime(frame, kept)
    frame = pd.concat([frame, normalised], axis=1)
    frame = pd.concat([frame, add_dynamics(frame, kept)], axis=1)

    lifetimes = unit_lifetimes(trajectories)
    fleet = induce_censoring(lifetimes, target_rate=target_rate, seed=seed)
    fleet = split_units(fleet, seed=seed)
    fleet.attrs = {**lifetimes.attrs, **fleet.attrs}

    risk_set = build_risk_set(frame, fleet)
    evaluation_panel = build_evaluation_panel(frame, fleet)

    return {
        "risk_set": risk_set,
        "evaluation_panel": evaluation_panel,
        "fleet": fleet,
        "sensors_kept": kept,
        "sensors_dropped": dropped,
        # Drift, not level, is the model's feature set. The argument is in
        # `add_dynamics`: engines enter service with different unstated initial
        # wear, so a sensor's level confounds "born worn" with "worn out".
        # The claim is not left as an argument — `evaluate.compare_feature_sets`
        # fits all three and reports that levels fit better in sample and
        # discriminate worse out of it, which is what overfitting looks like.
        "feature_columns": [f"{sensor}_drift" for sensor in kept],
        "level_columns": [f"{sensor}_roll" for sensor in kept],
        "n_regimes": int(frame["regime"].nunique()),
        "censoring": {
            "target_rate": target_rate,
            "realised_rate": float(fleet["censored"].mean()),
            "horizon_ceiling": int(fleet.attrs["horizon_ceiling"]),
            "seed": seed,
        },
    }
