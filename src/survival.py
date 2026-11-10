"""Four views of when an engine fails, in increasing order of what they assume.

The models are deliberately laid out as a ladder, because the point of the
project is the *decision*, and each rung exists to justify the next one:

1. **Kaplan-Meier + log-rank** — no covariates, no functional form. Establishes
   that the four subsets have genuinely different survival before any model is
   allowed to assume so.
2. **Cox proportional hazards with time-varying covariates** — the reflex
   choice for duration data. Fitted, and then its central assumption is
   *tested* rather than inherited.
3. **Discrete-time logistic hazard** — the workhorse. Produces the calibrated
   per-cycle failure probability the maintenance decision actually needs.
4. **Random Survival Forest** — a comparator whose only job is to bound how
   much discrimination the interpretable model gives up.

Why the discrete-time model is the workhorse and not Cox
--------------------------------------------------------
Three reasons, in order of weight:

* **The decision needs an absolute probability, not a ranking.** Cox's partial
  likelihood deliberately factors out the baseline hazard: it estimates
  *relative* risk and is silent about the absolute level unless a baseline is
  bolted back on afterwards. "Pull the unit when its 30-cycle failure risk
  exceeds p" is a statement about an absolute probability, so a model that
  estimates one directly is the honest fit to the question.
* **Time is discrete here.** An engine is inspected once per flight cycle and
  either fails or does not. That is a Bernoulli trial per cycle at risk, which
  is exactly what a logistic regression on the risk set estimates. Cox assumes
  continuous time in which exact ties cannot occur; with integer cycles, ties
  are not a nuisance to approximate away, they are the data.
* **Duration dependence is the question.** Whether the hazard accelerates with
  age is precisely what a maintenance policy turns on. Entering cycle as a set
  of band dummies lets the hazard take any shape; a linear term would impose
  the answer.

Cox is still fitted, because "I used the simpler model" is only credible if
the standard one was tried and its assumption checked. The PH test in
``test_proportional_hazards`` is that check.

Standard errors
---------------
Clustered by unit throughout. One engine contributes hundreds of cycle
observations that are anything but independent — a unit degrading fast does so
on every one of its rows. Treating them as independent would shrink the
standard errors by roughly the square root of the average unit length and make
every coefficient look far more precise than it is.

Predicting forward, and the assumption that makes it operational
----------------------------------------------------------------
The decision layer needs P(fail within the next *h* cycles | alive now). From a
discrete-time hazard that is

    risk_h(t) = 1 - Π_{s=t+1..t+h} (1 - λ(s, x))

which requires covariates at future cycles that have not happened yet. They are
held at their current value. This is not a convenience: at the moment of the
decision, today's readings are all the operator has, so a model that needed
tomorrow's would be unusable. The cost is that a unit degrading unusually fast
has its future risk understated, which makes the reported policy gain
conservative rather than flattering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

RANDOM_SEED = 20260801

# Cycles ahead the maintenance alarm looks. Chosen to be an operationally
# meaningful lead time — long enough to schedule a shop visit, short enough
# that the frozen-covariate assumption above is not doing heavy lifting.
# Swept in run_analysis.py so the policy conclusion is not tied to one horizon.
DEFAULT_HORIZON = 30

# Landmark cycles at which the models are scored. Landmarking is what makes a
# time-varying-covariate model comparable to a plain survival model: fix a
# cycle, keep the units still at risk then, use their readings at that instant,
# and score how well the predicted risk orders what happens next. Without it,
# "the model's C-index" is ambiguous about which covariate values it used.
LANDMARK_CYCLES = (50, 100, 150)

# Bands the baseline hazard is estimated in. Ten gives roughly 30 observed
# failures per band in the fit set — enough to estimate a level, few enough
# that the shape is not read out of noise.
N_HAZARD_BANDS = 10

# The band left out of the design as the reference category. Band 1 is the
# youngest band in which failures occur; band 0 (younger than any observed
# failure) is empty once the risk window is applied, so it cannot serve.
BAND_REFERENCE = 1


def kaplan_meier(fleet: pd.DataFrame, group_column: str = "subset") -> pd.DataFrame:
    """Survival curves by group, handling censoring correctly.

    A censored unit contributes everything it knows — that it lasted at least
    this long — without being counted as having failed. That is the whole
    reason to use Kaplan-Meier rather than averaging observed lifetimes, and
    the gap between the two is measured in ``evaluate.py``.
    """
    curves: list[pd.DataFrame] = []
    fitter = KaplanMeierFitter()

    for group, subset in fleet.groupby(group_column, observed=True):
        fitter.fit(
            subset["observed_time"],
            event_observed=subset["event"].astype(bool),
            label=str(group),
        )
        frame = fitter.survival_function_.reset_index()
        frame.columns = ["cycle", "survival"]
        confidence = fitter.confidence_interval_.reset_index(drop=True)
        frame["lower"] = confidence.iloc[:, 0].to_numpy()
        frame["upper"] = confidence.iloc[:, 1].to_numpy()
        frame["group"] = str(group)
        frame["units"] = len(subset)
        frame["events"] = int(subset["event"].sum())
        curves.append(frame)

    return pd.concat(curves, ignore_index=True)


def logrank(fleet: pd.DataFrame, group_column: str = "subset") -> dict:
    """Test whether the groups' survival curves differ at all.

    Run before any model is allowed to condition on the group. If the curves
    were indistinguishable, adding subset indicators would be fitting noise.
    """
    result = multivariate_logrank_test(
        fleet["observed_time"],
        fleet[group_column],
        fleet["event"].astype(bool),
    )
    return {
        "test_statistic": float(result.test_statistic),
        "p_value": float(result.p_value),
        "degrees_of_freedom": int(result.degrees_of_freedom),
        "groups": sorted(fleet[group_column].unique().tolist()),
    }


def risk_window_start(risk_set: pd.DataFrame) -> int:
    """The earliest cycle at which any failure has ever been observed.

    Derived from the fit set, never the holdout: it becomes part of the model
    specification, and a cut point chosen with knowledge of the holdout's
    failure times is a leak like any other.
    """
    failures = risk_set.loc[risk_set["failed"] == 1, "cycle"]
    if failures.empty:
        raise ValueError("Cannot locate the risk window: no observed failures.")
    return int(failures.min())


def restrict_to_risk_window(risk_set: pd.DataFrame, start: int) -> pd.DataFrame:
    """Condition the analysis on having survived to ``start``.

    In this fleet no engine fails before cycle 128. Those early cycles are not
    weak evidence of low risk — they are ~40,000 rows in which the outcome is
    constant, and a logistic model asked to explain them drives its intercept
    to minus infinity. That is textbook complete separation: the fit does not
    converge and the Hessian is singular.

    Restricting the risk set is the honest response, and it is a *modelling*
    decision with a name: the hazard is estimated conditional on surviving to
    cycle ``start``, which is delayed entry (left truncation) at a fixed
    landmark. Two things follow, both stated rather than buried:

    * Every quantity the model produces is conditional on the unit having
      reached ``start``. It says nothing about the first 127 cycles, and it
      should not — the data contains no failure there to learn from.
    * Units censored before ``start`` drop out entirely. They contribute no
      information about a window they never entered. ``evaluate.py`` reports
      how many are lost so the reader can judge the cost rather than take it
      on trust.

    Operationally this matches the decision anyway: an engine younger than the
    shortest life ever seen is not a maintenance candidate.
    """
    return risk_set.loc[risk_set["cycle"] >= start].copy()


def derive_band_edges(risk_set: pd.DataFrame, n_bands: int = N_HAZARD_BANDS) -> np.ndarray:
    """Cut points for the piecewise-constant baseline hazard, from failures alone.

    Equal-width bands do not work on this fleet. In the sparse tail several
    20-cycle bands hold only a few dozen rows and no failures at all, and a
    band with zero events has a coefficient of minus infinity — perfect
    separation, singular Hessian, no fit.

    Equal-*event* bands remove the problem by construction. Cut points are the
    quantiles of the observed failure cycles, so every band contains roughly
    the same number of events and none is empty: with ten bands and ~300
    failures in the fit set, about thirty events each.

    Derived from the **fit** risk set only. Quantiles of holdout failure times
    would be the holdout's answers leaking into the design matrix.
    """
    failure_cycles = risk_set.loc[risk_set["failed"] == 1, "cycle"].to_numpy(dtype=float)
    if len(failure_cycles) == 0:
        raise ValueError("Cannot derive hazard bands: the risk set has no failures.")

    quantiles = np.linspace(0.0, 1.0, n_bands + 1)[1:-1]
    interior = np.unique(np.quantile(failure_cycles, quantiles).round())
    return np.concatenate([[failure_cycles.min()], interior])


def assign_band(cycles: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Band index for each cycle. Band 0 is everything before the first failure."""
    return np.searchsorted(edges, np.asarray(cycles, dtype=float), side="right")


def build_design(risk_set: pd.DataFrame, feature_columns: list[str],
                 edges: np.ndarray | None = None) -> pd.DataFrame:
    """Assemble the design matrix for the discrete-time hazard model.

    Cycle enters as band dummies so the baseline hazard is free to take any
    shape — whether risk accelerates with age is the question a maintenance
    policy turns on, so it is estimated, not imposed by a linear term.

    The **lowest** band is the reference and is left out. Dropping it is not
    optional bookkeeping: after ``restrict_to_risk_window`` every row falls in
    some band, so a full set of dummies would sum to one in every row and be
    exactly collinear with the intercept. The fit would then depend on whatever
    the linear algebra happened to do with a singular Hessian rather than on
    the data. Every band coefficient therefore reads as log-odds of failing
    this cycle relative to the youngest band at which failures occur.
    """
    design = risk_set.loc[:, feature_columns].astype(float).copy()

    if edges is not None:
        bands = assign_band(risk_set["cycle"].to_numpy(), edges)
        for level in range(1, len(edges) + 1):
            if level == BAND_REFERENCE:
                continue
            design[f"band_{level}"] = (bands == level).astype(float)

    return sm.add_constant(design, has_constant="add")


def fit_discrete_hazard(risk_set: pd.DataFrame, feature_columns: list[str],
                        edges: np.ndarray | None = None):
    """Fit the per-cycle failure probability with unit-clustered standard errors.

    The outcome is one Bernoulli trial per cycle at risk. Censored units
    contribute every cycle they were watched with outcome 0 — which is the
    whole point: they are informative about *not* failing, and the complete-case
    alternative throws that away.
    """
    design = build_design(risk_set, feature_columns, edges=edges)
    outcome = risk_set["failed"].astype(int)

    model = sm.Logit(outcome, design)
    return model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": risk_set["unit_id"]},
        maxiter=200,
    )


def fit_complete_case(risk_set: pd.DataFrame, feature_columns: list[str],
                      edges: np.ndarray | None = None):
    """The same model, fitted only on units whose failure was observed.

    This is the censoring-blind alternative, and it is not a straw man: any
    method that needs a completed lifetime as its target — which is what
    "regress on remaining useful life" needs — can only be trained on these
    units. Fitting it explicitly is how the bias gets measured instead of
    asserted.
    """
    observed_only = risk_set.loc[risk_set["event"] == 1]
    return fit_discrete_hazard(observed_only, feature_columns, edges=edges)


def fit_cox_time_varying(risk_set: pd.DataFrame, feature_columns: list[str],
                         penalizer: float = 0.1):
    """Cox PH on the counting-process form, clustered by unit.

    A small ridge penalty is used because several sensor pairs are near-
    duplicates by construction (corrected and uncorrected fan speed, for
    instance, correlate at 0.97). Without it the partial likelihood is close to
    flat along those directions and the fit wanders; with it the model
    converges and the *ranking* it produces — which is all Cox is used for
    here — is stable. The penalty is reported, not hidden: it biases individual
    coefficients toward zero, which is why the coefficient table is read at the
    level of "which sensors matter", not "this one is worth exactly 0.31".
    """
    columns = ["unit_id", "entry", "exit", "failed", *feature_columns]
    frame = risk_set.loc[:, columns].copy()
    frame[feature_columns] = frame[feature_columns].astype(float)

    fitter = CoxTimeVaryingFitter(penalizer=penalizer)
    fitter.fit(
        frame,
        id_col="unit_id",
        event_col="failed",
        start_col="entry",
        stop_col="exit",
        show_progress=False,
    )
    return fitter


def schoenfeld_residuals(fitter: CoxTimeVaryingFitter, risk_set: pd.DataFrame,
                         feature_columns: list[str]) -> pd.DataFrame:
    """Schoenfeld residuals: observed covariate at each failure minus its expectation.

    At each failure time the model implies a distribution over *which* of the
    engines still at risk should be the one to fail — engine *k* with weight
    proportional to ``exp(βᵀx_k)``. The residual is what actually failed minus
    that weighted mean:

        r_j = x_(failed at t_j) − Σ_k w_k x_k ,  w_k ∝ exp(βᵀx_k) over the risk set

    Under proportional hazards these residuals are noise with no relation to
    time. If a covariate's effect strengthens as engines age, its residuals
    trend with time, and that trend is the diagnostic.

    Computing them explicitly matters here. The obvious shortcut — correlating
    the raw covariate values at failures with failure time — tests the wrong
    thing entirely: the drift features *rise with age by construction*, so that
    shortcut flags essentially every covariate as a violation whether or not
    the model's assumption is actually broken. The residual subtracts the
    model's own expectation, which is precisely what removes that artefact.
    """
    coefficients = np.array([float(fitter.params_.get(column, 0.0))
                             for column in feature_columns])
    covariates = risk_set.loc[:, feature_columns].to_numpy(dtype=float)
    entry = risk_set["entry"].to_numpy(dtype=float)
    exit_ = risk_set["exit"].to_numpy(dtype=float)
    failed = risk_set["failed"].to_numpy(dtype=bool)

    linear = covariates @ coefficients
    # Subtract the max before exponentiating: without it a large linear
    # predictor overflows and the weights come out as nan.
    failure_times = np.unique(exit_[failed])

    rows: list[np.ndarray] = []
    kept_times: list[float] = []
    skipped = 0

    for time in failure_times:
        at_risk = (entry < time) & (exit_ >= time)
        # A risk set of one carries no information: the engine that failed is
        # the only candidate, so its residual is identically zero and would
        # dilute the correlation toward "no violation". Skipped and counted,
        # never silently included.
        if at_risk.sum() < 2:
            skipped += int((failed & (exit_ == time)).sum())
            continue
        weights = np.exp(linear[at_risk] - linear[at_risk].max())
        weights /= weights.sum()
        expected = weights @ covariates[at_risk]

        for index in np.flatnonzero(failed & (exit_ == time)):
            rows.append(covariates[index] - expected)
            kept_times.append(time)

    if not rows:
        raise ValueError("No usable failure times; the PH test has nothing to work with.")

    residuals = pd.DataFrame(rows, columns=feature_columns)
    residuals.insert(0, "failure_time", kept_times)
    residuals.attrs["failures_skipped_singleton_risk_set"] = skipped
    residuals.attrs["failures_used"] = len(residuals)
    return residuals


def test_proportional_hazards(fitter: CoxTimeVaryingFitter,
                              risk_set: pd.DataFrame,
                              feature_columns: list[str]) -> pd.DataFrame:
    """Check Cox's central assumption instead of inheriting it.

    Proportional hazards says a covariate's effect on the log-hazard is the
    same at cycle 130 as at cycle 350. For a degrading engine that is a strong
    claim worth testing: a sensor drifting high may matter far more late in
    life than early.

    The test rank-correlates each covariate's Schoenfeld residuals with failure
    time. Under PH the correlation is zero; a significant one means that
    covariate's effect moves with age. Implemented here rather than through
    ``check_assumptions`` because lifelines does not expose that helper for the
    time-varying fitter.

    Rank correlation rather than Pearson, because a handful of extreme
    residuals late in life would otherwise drive the statistic on their own.
    """
    residuals = schoenfeld_residuals(fitter, risk_set, feature_columns)
    rank_times = residuals["failure_time"].rank().to_numpy()
    n = len(residuals)

    rows: list[dict] = []
    for column in feature_columns:
        values = residuals[column].to_numpy(dtype=float)
        if np.std(values) == 0 or n < 5:
            rows.append({"covariate": column, "correlation": np.nan, "p_value": np.nan,
                         "coefficient": float(fitter.params_.get(column, np.nan))})
            continue
        correlation = float(np.corrcoef(pd.Series(values).rank().to_numpy(), rank_times)[0, 1])
        z = np.arctanh(np.clip(correlation, -0.999999, 0.999999)) * np.sqrt(n - 3)
        rows.append({
            "covariate": column,
            "correlation": correlation,
            "p_value": float(2 * (1 - _standard_normal_cdf(abs(z)))),
            "coefficient": float(fitter.params_.get(column, np.nan)),
        })

    result = pd.DataFrame(rows)
    # Seventeen covariates tested at once is seventeen chances to find a
    # spurious violation. Holm-Bonferroni controls the family-wise error rate
    # without assuming the tests are independent — they are not, since the
    # sensors are correlated.
    result = result.sort_values("p_value").reset_index(drop=True)
    n_tests = result["p_value"].notna().sum()
    adjusted = []
    running = 0.0
    for position, p_value in enumerate(result["p_value"]):
        if pd.isna(p_value):
            adjusted.append(np.nan)
            continue
        running = max(running, min(1.0, p_value * (n_tests - position)))
        adjusted.append(running)
    result["p_holm"] = adjusted
    result["violates_ph"] = result["p_holm"] < 0.05
    return result


def _standard_normal_cdf(x: float) -> float:
    """Normal CDF via the error function, so scipy is not needed for one call."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def predict_hazard(result, risk_set: pd.DataFrame, feature_columns: list[str],
                   edges: np.ndarray | None = None) -> np.ndarray:
    """Per-cycle failure probability for each row of a risk set."""
    design = build_design(risk_set, feature_columns, edges=edges)
    design = design.reindex(columns=result.params.index, fill_value=0.0)
    return np.asarray(result.predict(design), dtype=float)


def horizon_risk(result, risk_set: pd.DataFrame, feature_columns: list[str],
                 edges: np.ndarray, horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """P(fail within ``horizon`` cycles | alive now), per row.

    Built from the per-cycle hazard by compounding survival across the horizon:
    the unit survives the horizon only if it survives every cycle in it, so the
    risk is one minus the product of per-cycle survivals.

    Covariates are frozen at their current value (argued in the module
    docstring) but the **cycle band advances**, so the model's estimated ageing
    is carried forward even though the sensor readings are not. A unit that
    would cross into a higher-hazard band during the horizon has that reflected
    in its risk.
    """
    design = build_design(risk_set, feature_columns, edges=edges)
    design = design.reindex(columns=result.params.index, fill_value=0.0)

    band_columns = [column for column in design.columns if column.startswith("band_")]
    band_index = {column: design.columns.get_loc(column) for column in band_columns}
    band_levels = {column: int(column.split("_")[1]) for column in band_columns}

    coefficients = result.params.to_numpy()
    base = design.to_numpy(dtype=float).copy()
    current_cycle = risk_set["cycle"].to_numpy(dtype=float)
    survival = np.ones(len(design), dtype=float)

    for step in range(1, horizon + 1):
        future_band = assign_band(current_cycle + step, edges)
        for column, level in band_levels.items():
            base[:, band_index[column]] = (future_band == level).astype(float)
        hazard = 1.0 / (1.0 + np.exp(-(base @ coefficients)))
        survival *= (1.0 - hazard)

    return 1.0 - survival


def landmark_frame(risk_set: pd.DataFrame, landmark: int) -> pd.DataFrame:
    """The units still at risk at ``landmark``, with their readings at that cycle.

    This is what makes a time-varying model scoreable against a plain survival
    metric. Fix a cycle; keep the units that have neither failed nor been
    censored by then; take each one's covariates at exactly that cycle; and
    define the outcome as time from the landmark to failure or censoring.

    Units already gone by the landmark are excluded, not treated as failures.
    They are not part of the population the question is about — an operator
    standing at cycle 100 is deciding about engines that are running at cycle
    100.
    """
    at_landmark = risk_set.loc[risk_set["cycle"] == landmark].copy()
    if at_landmark.empty:
        raise ValueError(f"No units are at risk at cycle {landmark}.")

    at_landmark["time_from_landmark"] = at_landmark["observed_time"] - landmark
    at_landmark["event_from_landmark"] = at_landmark["event"].astype(int)
    return at_landmark.reset_index(drop=True)


def fit_random_survival_forest(landmark_data: pd.DataFrame,
                               feature_columns: list[str],
                               n_estimators: int = 300,
                               seed: int = RANDOM_SEED):
    """A flexible comparator, to bound what the interpretable model gives up.

    Its job is not to be deployed. If the forest is barely ahead, the
    discrete-time model's interpretability is free and the project says so; if
    it is far ahead, the linear form is costing something real and the project
    says that instead. Either way the number is reported rather than assumed —
    the same move the breast-cancer project made when nine classifiers turned
    out to sit inside one standard deviation of each other.
    """
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    y = Surv.from_arrays(
        event=landmark_data["event_from_landmark"].astype(bool).to_numpy(),
        time=landmark_data["time_from_landmark"].astype(float).to_numpy(),
    )
    X = landmark_data.loc[:, feature_columns].astype(float).to_numpy()

    forest = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X, y)
    return forest


def tidy(result) -> pd.DataFrame:
    """Coefficients as odds ratios with clustered confidence intervals.

    Odds ratios because the readable statement is "a unit one standard
    deviation further from its own baseline on sensor 11 is X times as likely
    to fail this cycle", not "the log-odds rise by 0.24".
    """
    confidence = result.conf_int()
    frame = pd.DataFrame({
        "coefficient": result.params,
        "std_error": result.bse,
        "p_value": result.pvalues,
        "odds_ratio": np.exp(result.params),
        "or_lower": np.exp(confidence[0]),
        "or_upper": np.exp(confidence[1]),
    })
    frame.index.name = "term"
    return frame.reset_index()
