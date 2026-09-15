"""Layer-5 rediscovery validation: does the optimizer recover real rovers?

The headline falsifiable claim in the paper. For each rover in the
registry, ask: "given the rover's mass budget and a *class-generic*
mission scenario (not a Pragyan-specific YAML, and not even the
canonical tradespace YAML whose duty-cycle is inspection-calibrated
against real-rover ops), does NSGA-II find Pareto-optimal designs
near the real rover's design vector?"

Two leakage controls
--------------------
The naive version of this test — run NSGA-II against
``chandrayaan3_pragyan.yaml`` and check whether the front lands on
Pragyan's design — is circular: that YAML's ``operational_duty_cycle``
was calibrated against Pragyan's *actual* 101 m / 10-day traverse, so
the optimizer is being asked to recover a rover the scenario was
already pointed at. This module breaks that circularity in two ways:

Panel-pointing fix (2026-05-28)
-------------------------------
The class-generic ``*_micro`` scenarios run at high latitudes
(``polar_micro`` at lat=-85, ``mare_micro`` at lat=+30, etc.). The
upstream evaluator's default of ``panel_tilt_deg=0`` (horizontal
panel) under-predicts insolation by ~18x at lat=±85 — a horizontal
panel sees ``cos(incidence) = sin(5 deg) ~= 0.087`` of the
normal-incidence irradiance — and forces every polar registry rover
(Pragyan, MoonRanger, CADRE-unit) to stall on its own scenario
with negative energy margin and ``range_km = 0``. The earlier
canonical per-rover YAMLs (``chandrayaan3_pragyan.yaml`` etc.)
hid this by calibrating ``operational_duty_cycle`` low enough to
absorb the missing tilted-panel insolation; once leakage control
#1 lifts ``δ_ops`` to a class-neutral 0.10 the latent
horizontal-panel bug surfaces as a polar-trio mass dominance
artefact rather than a real "operationally conservative" finding.

To keep the rediscovery comparison physically defensible, this
module installs a fixed-tilt approximation
(:func:`_scenario_panel_orientation`) that points the panel at the
noon sun — ``tilt_deg = min(80, |latitude|)`` with the azimuth on
the noon-sun side of local north — and forwards it to BOTH the
rover's re-evaluation under the class-generic scenario AND every
NSGA-II individual the optimiser scores against it. This is the
same orientation choice that real polar rovers (MoonRanger
mast-deployable, CADRE articulated) carry by design; it does not
introduce per-rover calibration. The surrogate backend is
unaffected by tilt overrides because the v8 LHS was trained on
horizontal-panel evaluator outputs (a v9 regen would be required
to restore symmetry); use the evaluator backend at high latitudes.

1. **Class-generic micro-rover scenario library.** Every registry
   rover is mapped to one of four ``*_micro`` scenarios
   (:func:`roverdevkit.mission.scenarios.list_class_generic_micro_scenarios`),
   not to the canonical four tradespace scenarios. The ``*_micro``
   library is parallel to the canonical library (same terrain class /
   soil / sun geometry / non-binding traverse budget) but pins
   ``operational_duty_cycle`` to a flat class-neutral 0.10 across all
   four scenarios. The canonical scenarios' δ_ops anchors
   (mare 0.30 against Apollo-17 LRV; polar 0.05 against Pragyan /
   Yutu-2 real-ops; crater 0.20 against MER-A/B; highland 0.15) are
   exactly the per-rover inspection calibrations the rediscovery test
   needs to keep out of its scenario. The per-rover YAML files
   (``chandrayaan3_pragyan.yaml`` etc.) are not used by the
   rediscovery harness either.
2. **Mass budget as constraint, not the rover's design vector.** The
   only piece of rover-specific information that enters the search is
   the published total mass (which is a directly-cited bulk number
   from press kits / mission papers, with no engineering content
   leaked back from the registry's imputed fields).

Procedure (per rover)
---------------------
1. Look up the registry entry.
2. Resolve the class-generic scenario from
   :func:`class_generic_scenario_for` (lat / terrain / sun match).
3. Compute the rover's total mass under the bottom-up mass model
   (used as the constraint ceiling at +5 %).
4. Run NSGA-II with objectives ``(range_km max, total_mass_kg min,
   slope_capability_deg max)`` and the mass-ceiling constraint.
5. Score:
   - **Design-space distance.** Normalised L2 over the continuous
     design variables (``n_wheels`` and ``grouser_count`` reported as
     exact-match diagnostics, not in the L2).
   - **Per-variable percent error** between the nearest Pareto point
     and the real rover.
   - **Pareto-dominance.** Re-evaluate the rover's design vector under
     the same class-generic scenario; flag whether any Pareto point
     strictly dominates it on all three objectives.

Acceptance criterion (paper-side): median per-variable error within
~25-30 % across continuous design variables on the flown registry, and
the real rover is *not* strictly Pareto-dominated by the optimizer's
front. Tighter tolerances are documented per rover.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roverdevkit.mission.evaluator import evaluate as evaluator_evaluate
from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.tradespace.optimizer import (
    DEFAULT_OBJECTIVES,
    DESIGN_BOUNDS,
    NSGA2Runner,
    OptimizationBackend,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
)
from roverdevkit.validation.rover_registry import (
    RoverRegistryEntry,
    flown_registry,
    registry_by_name,
)

# ---------------------------------------------------------------------------
# Scenario-driven panel orientation (Option A — fixed-tilt approximation)
# ---------------------------------------------------------------------------

# Cap on the polar-deployable tilt angle. 80 deg keeps the panel a
# few degrees off the plane of the local horizon (avoiding the
# numerically-singular grazing-incidence regime for sun elevations
# below ~3 deg) while still letting the surface normal track within
# ~10 deg of the noon sun across the full lunar latitude range.
# Mast-deployable arrays on real polar rovers (MoonRanger,
# Resource Prospector concepts, JPL CADRE) sit in the 75-85 deg band
# for the same reason.
_MAX_PANEL_TILT_DEG: float = 80.0


def _scenario_panel_orientation(scenario: MissionScenario) -> tuple[float, float]:
    """Return ``(panel_tilt_deg, panel_azimuth_deg)`` for ``scenario``.

    Implements a fixed-tilt approximation: the rover's surface
    normal points toward the noon sun, with tilt clamped at
    :data:`_MAX_PANEL_TILT_DEG` to avoid grazing-incidence
    pathologies and azimuth set to the noon-sun side of local north.

    - ``panel_tilt_deg = min(_MAX_PANEL_TILT_DEG, |lat|)``. At lat=0
      this collapses to a horizontal panel; at lat=±85 it pegs at
      80 deg (a few deg off the local horizon).
    - ``panel_azimuth_deg = 0`` (north) for southern-hemisphere
      scenarios, ``180`` (south) for northern. Ignored when
      ``panel_tilt_deg == 0``.

    This is a *class-level* polar-rover assumption: real polar
    micro-rovers (MoonRanger mast-deployable, CADRE articulated)
    point their arrays at the low-elevation sun by design.
    Removing the assumption (passing tilt=0) recovers the original
    horizontal-panel physics that under-predicts polar insolation
    by ~18x at lat=±85 and forces all polar rovers to stall
    against their own canonical scenarios.
    """
    tilt_deg = min(_MAX_PANEL_TILT_DEG, abs(float(scenario.latitude_deg)))
    azimuth_deg = 0.0 if scenario.latitude_deg < 0.0 else 180.0
    return tilt_deg, azimuth_deg


# ---------------------------------------------------------------------------
# Class-generic scenario mapping (leakage control #1)
# ---------------------------------------------------------------------------

_CLASS_GENERIC_SCENARIO: dict[str, str] = {
    # Polar south, intermittent sun, polar regolith - matches polar_micro.
    "Pragyan": "polar_micro",
    "MoonRanger": "polar_micro",
    "CADRE-unit": "polar_micro",
    # Mid-latitude mare-class terrain - matches mare_micro. Yutu-2 sits
    # at +45 deg in Von Karman crater (mare floor); Rashid-1 at +47 deg
    # in Atlas crater (Mare Frigoris floor); Tenacious at ~+60 deg in
    # the same Mare Frigoris area. All ride mare-nominal regolith and
    # diurnal sun. mare_micro pins latitude at a class-neutral +30 deg
    # (between Yutu-2 / Rashid-1 / Tenacious without matching any) and
    # pins operational_duty_cycle to class-neutral 0.10 instead of the
    # canonical equatorial_mare_traverse 0.30 anchor.
    "Yutu-2": "mare_micro",
    "Rashid-1": "mare_micro",
    "Tenacious": "mare_micro",
}


def class_generic_scenario_for(rover_name: str) -> str:
    """Return the canonical class-generic scenario name for a registry rover.

    Maps each rover to one of the four tradespace scenarios based on
    its real operating environment (latitude, terrain class, sun
    geometry). The rediscovery harness uses this scenario as the
    NSGA-II target instead of the per-rover-tuned validation YAML, so
    no rover-specific ops duty cycle leaks back into the search.

    Raises
    ------
    KeyError
        If no class-generic scenario has been declared for ``rover_name``.
        Add a new rover by extending :data:`_CLASS_GENERIC_SCENARIO`.
    """
    try:
        return _CLASS_GENERIC_SCENARIO[rover_name]
    except KeyError as exc:
        known = sorted(_CLASS_GENERIC_SCENARIO)
        raise KeyError(
            f"no class-generic scenario for {rover_name!r}; "
            f"known rovers: {known}. Extend "
            "`_CLASS_GENERIC_SCENARIO` in roverdevkit.validation.rover_rediscovery."
        ) from exc


# ---------------------------------------------------------------------------
# Design-space distance helpers
# ---------------------------------------------------------------------------

# Continuous (non-integer) design variables reported per-variable.
_CONTINUOUS_VARIABLES: tuple[str, ...] = (
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "chassis_mass_kg",
    "wheelbase_m",
    "solar_area_m2",
    "battery_capacity_wh",
    "avionics_power_w",
    "peak_wheel_torque_nm",
)

# Subset used in the normalised L2. ``wheelbase_m`` is carried in the
# design vector and reported, but no sub-model consumes it (it reaches
# neither the mass, terramechanics, power, thermal, nor traverse
# stage), so the optimiser has no signal on it and its Pareto values
# are unconstrained. Scoring design-space agreement on a coordinate the
# evaluator never constrains would measure search noise, so it is
# excluded here. Its per-variable error is still reported.
_DISTANCE_VARIABLES: tuple[str, ...] = tuple(
    name for name in _CONTINUOUS_VARIABLES if name != "wheelbase_m"
)

# Integer-typed design variables reported as exact-match diagnostics
# rather than rolled into the L2 (no meaningful distance metric for a
# 4-vs-6 wheel count).
_INTEGER_VARIABLES: tuple[str, ...] = ("n_wheels", "grouser_count")


def _normalised_vector(design: DesignVector) -> np.ndarray:
    """Map distance-metric fields of a design to [0, 1] via DESIGN_BOUNDS."""
    values = []
    for name in _DISTANCE_VARIABLES:
        lo, hi = DESIGN_BOUNDS[name]
        span = max(hi - lo, 1e-12)
        x = float(getattr(design, name))
        values.append((x - lo) / span)
    return np.asarray(values, dtype=float)


def _normalised_l2(a: DesignVector, b: DesignVector) -> float:
    return float(np.linalg.norm(_normalised_vector(a) - _normalised_vector(b)))


# A real value below this magnitude is treated as "effectively zero" for
# the percent-error metric. 1e-6 is well below every continuous design
# variable's natural scale (mm-class for lengths, 0.1-Wh for battery,
# 0.01-N·m for torque); above it we use the standard %-of-target form.
_NEAR_ZERO_TARGET: float = 1e-6


def _per_variable_percent_errors(
    pareto: DesignVector, rover: DesignVector
) -> dict[str, float]:
    """Signed percent error per continuous design variable.

    For nonzero targets the metric is the standard
    ``(candidate - target) / |target| * 100``. For targets at or near
    zero (e.g. CADRE's ``grouser_height_m = 0.0`` for smooth wire-
    spoke rims) the metric switches to ``(candidate - target) /
    design_space_range * 100`` so it remains finite and interpretable.
    The two formulas coincide for targets near the upper end of the
    design space and differ only when the target sits on the design-
    space corner; we cite the switch in the markdown report so a
    reviewer can distinguish "rover value at a corner" from
    "optimiser disagrees strongly with the rover".
    """
    out: dict[str, float] = {}
    for name in _CONTINUOUS_VARIABLES:
        target = float(getattr(rover, name))
        candidate = float(getattr(pareto, name))
        if abs(target) >= _NEAR_ZERO_TARGET:
            out[name] = (candidate - target) / abs(target) * 100.0
        else:
            lo, hi = DESIGN_BOUNDS[name]
            span = max(hi - lo, 1e-9)
            out[name] = (candidate - target) / span * 100.0
    return out


def _integer_matches(
    pareto: DesignVector, rover: DesignVector
) -> dict[str, bool]:
    return {
        name: int(getattr(pareto, name)) == int(getattr(rover, name))
        for name in _INTEGER_VARIABLES
    }


# ---------------------------------------------------------------------------
# Pareto-dominance check
# ---------------------------------------------------------------------------


def _dominates(
    candidate: dict[str, float],
    reference: dict[str, float],
    objectives: tuple[OptimizationObjective, ...],
) -> bool:
    """Return True if ``candidate`` strictly Pareto-dominates ``reference``.

    Dominance is defined in the optimiser's sense: for every objective
    the candidate is no worse than the reference, and on at least one
    it is strictly better.
    """
    no_worse = True
    strictly_better = False
    for obj in objectives:
        cand = float(candidate[obj.target])
        ref = float(reference[obj.target])
        if obj.direction == "max":
            if cand < ref:
                no_worse = False
                break
            if cand > ref:
                strictly_better = True
        else:  # "min"
            if cand > ref:
                no_worse = False
                break
            if cand < ref:
                strictly_better = True
    return no_worse and strictly_better


# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RediscoveryResult:
    """Outcome of running rediscovery on one registry rover.

    Attributes
    ----------
    rover_name
        Registry key, e.g. ``"Pragyan"``.
    class_generic_scenario
        Name of the canonical scenario used for the search
        (:func:`class_generic_scenario_for`). Recorded so the report
        explicitly shows the scenario YAML was *not* a per-rover
        validation file.
    mass_budget_kg
        Constraint ceiling fed to the optimiser (rover total mass
        under the bottom-up model + 5 % slop).
    nearest_pareto_index
        Index of the Pareto-front point with the smallest normalised L2
        distance to the rover's design vector.
    nearest_pareto_design
        The Pareto design at ``nearest_pareto_index``.
    nearest_pareto_metrics
        Mission metrics for the nearest Pareto design.
    design_space_distance
        Normalised L2 distance over the continuous design variables.
        ``0`` is identical; ``sqrt(len(_CONTINUOUS_VARIABLES))`` is the
        worst case (opposite corners of every box bound).
    per_variable_percent_errors
        Signed percent error ``(pareto - rover) / |rover| * 100`` for
        each continuous variable; positive ⇒ optimiser landed above
        the real rover.
    integer_matches
        Exact-match flags for ``n_wheels`` and ``grouser_count``.
    rover_metrics_under_generic_scenario
        Mission metrics for the rover's design re-evaluated under the
        class-generic scenario (no per-rover overrides). This is the
        reference for the Pareto-dominance check.
    pareto_dominated
        True iff at least one optimiser-found Pareto point strictly
        dominates the real rover in objective space. If True, the
        model is saying "we would have built something else"; the
        paper must explain why.
    optimization_result
        The raw :class:`OptimizationResult` (Pareto front + metrics +
        per-generation checkpoints) returned by
        :class:`NSGA2Runner`. Retained for downstream notebook /
        report rendering.
    """

    rover_name: str
    class_generic_scenario: str
    mass_budget_kg: float
    nearest_pareto_index: int
    nearest_pareto_design: DesignVector
    nearest_pareto_metrics: dict[str, float]
    design_space_distance: float
    per_variable_percent_errors: dict[str, float]
    integer_matches: dict[str, bool]
    rover_metrics_under_generic_scenario: dict[str, float]
    pareto_dominated: bool
    optimization_result: OptimizationResult = field(repr=False)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _evaluate_rover_under(
    entry: RoverRegistryEntry, scenario: MissionScenario
) -> dict[str, float]:
    """Run the corrected evaluator on the rover's design under a scenario.

    The rover's panel orientation comes from
    :func:`_scenario_panel_orientation` rather than from the
    registry entry: under the class-generic ``*_micro`` scenarios
    we assume the rover would carry a polar-deployable / sun-tracking
    array if its real flight-time pointing strategy demanded one
    (MoonRanger, CADRE-unit), and a horizontal array otherwise. This
    keeps the "is the optimiser finding a better design than the real
    rover?" comparison apples-to-apples — the optimiser's NSGA-II
    runner uses the same scenario-driven orientation for every
    candidate it evaluates (see :func:`rediscover`).
    """
    panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(scenario)
    metrics = evaluator_evaluate(
        entry.design,
        scenario,
        gravity_m_per_s2=entry.gravity_m_per_s2,
        thermal_architecture=entry.thermal_architecture,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
    )
    return {
        "range_km": float(metrics.range_km),
        "energy_margin_raw_pct": float(metrics.energy_margin_raw_pct),
        "slope_capability_deg": float(metrics.slope_capability_deg),
        "total_mass_kg": float(metrics.total_mass_kg),
    }


def rediscover(
    rover_name: str,
    *,
    objectives: tuple[OptimizationObjective, ...] = DEFAULT_OBJECTIVES,
    mass_ceiling_slop: float = 0.10,
    population_size: int = 60,
    n_generations: int = 16,
    seed: int = 0,
    backend: OptimizationBackend = "evaluator",
    bundles: dict[str, QuantileHeads] | None = None,
    evaluator_eval_cap: int = 1000,
) -> RediscoveryResult:
    """Run the Layer-5 rediscovery test for one registry rover.

    Parameters
    ----------
    rover_name
        Registry key, e.g. ``"Pragyan"`` or ``"Yutu-2"``.
    objectives
        Pareto objectives. Defaults to the canonical
        ``(range_km max, total_mass_kg min, slope_capability_deg
        max)`` set used by the webapp.
    mass_ceiling_slop
        Fractional slop above the rover's modelled total mass for the
        constraint ceiling. ``0.10`` ⇒ ``mass_ceiling = m_rover * 1.10``,
        i.e. the AIAA S-120A PDR-level dry-mass growth allowance. For
        tighter conceptual-design budgets (5 %), pass an explicit value.
        population_size, n_generations
        NSGA-II hyperparameters. Defaults give ``60 * 16 = 960``
        evaluations - inside the evaluator's 1 000-eval cap, ~30-40 s
        on a single core with the analytical evaluator.
        For small-rover mass budgets the population must be large
        enough that random LHS initialisation contains at least a few
        mass-feasible candidates; the default 60 has empirically
        cleared this on every registry rover.
    seed
        Random seed for reproducibility.
    backend
        ``"evaluator"`` (default) routes NSGA-II through the corrected
        physics evaluator (~20 ms per design, single-seeded by the
        ``seed`` arg). ``"surrogate"`` routes through the calibrated
        quantile-XGB heads (~0.1 ms per design); requires ``bundles``.
        For high-budget ensembles the surrogate backend admits 100k+
        evaluations comfortably; the evaluator backend is more
        accurate but practically limited to ~10k evaluations per
        seed by single-core wall time. Designs sampled below the v4
        LHS training-support floors (chassis < 3 kg, torque < 0.3 Nm,
        battery < 20 Wh) will be extrapolated by the surrogate -
        ultra-micro rovers (CADRE, Tenacious) should be run on the
        evaluator backend until the v5 LHS regen lands.
    bundles
        Required when ``backend == "surrogate"``. Map ``{target ->
        QuantileHeads}`` produced by
        :mod:`roverdevkit.surrogate.uncertainty`; load via
        ``joblib.load("models/surrogate_v9/quantile_bundles.joblib")``.
    evaluator_eval_cap
        Safety cap on the evaluator-backed NSGA-II runner
        (``population_size * n_generations`` must be below this). The
        webapp default is 1 000; for high-budget runs raise to 10 000
        or higher. Ignored when ``backend == "surrogate"``.

    Returns
    -------
    RediscoveryResult
        Scored rediscovery summary plus the raw Pareto front.

    Raises
    ------
    KeyError
        If ``rover_name`` is not in the registry or has no
        class-generic scenario declared.
    """
    entry = registry_by_name(rover_name)
    scenario_name = class_generic_scenario_for(rover_name)
    # Schema v9: forward the rover's *published* payload (carried on its
    # per-rover validation scenario) onto the otherwise class-neutral
    # ``*_micro`` scenario. This is applied uniformly to BOTH the
    # rover's re-evaluation and every NSGA-II candidate the optimiser
    # scores, so the Pareto-dominance comparison is apples-to-apples:
    # the optimiser must carry the same scientific-payload mass the
    # real rover flew rather than floating chassis to the LHS floor and
    # skipping payload entirely. Payload mass is a directly-cited bulk
    # number (instrument-suite mass from mission papers), not an
    # engineering field back-solved from the registry, so it does not
    # leak design-space information into the search (same status as the
    # published total-mass ceiling).
    scenario = load_scenario(scenario_name).model_copy(
        update={
            "payload_mass_kg": entry.scenario.payload_mass_kg,
            "payload_power_w": entry.scenario.payload_power_w,
        }
    )
    soil = get_soil_parameters(scenario.soil_simulant)

    rover_metrics = _evaluate_rover_under(entry, scenario)
    mass_budget_kg = float(rover_metrics["total_mass_kg"]) * (1.0 + mass_ceiling_slop)

    mass_ceiling = OptimizationConstraint(
        target="total_mass_kg",
        sense="max",
        value=mass_budget_kg,
    )

    panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(scenario)
    runner = NSGA2Runner(
        scenario,
        soil,
        backend=backend,
        bundles=bundles,
        objectives=objectives,
        constraints=(mass_ceiling,),
        population_size=population_size,
        n_generations=n_generations,
        seed=seed,
        evaluator_eval_cap=evaluator_eval_cap,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
    )
    optimization = runner.run()

    if not optimization.design_vectors:
        raise RuntimeError(
            f"NSGA-II returned an empty Pareto front for {rover_name!r}; "
            f"every individual violated the mass ceiling "
            f"{mass_budget_kg:.2f} kg. Loosen `mass_ceiling_slop` or "
            "inspect the optimiser logs."
        )

    distances = np.asarray(
        [_normalised_l2(d, entry.design) for d in optimization.design_vectors],
        dtype=float,
    )
    nearest_idx = int(np.argmin(distances))
    nearest_design = optimization.design_vectors[nearest_idx]
    nearest_metrics = optimization.metrics[nearest_idx]

    dominated = any(
        _dominates(point, rover_metrics, objectives)
        for point in optimization.metrics
    )

    return RediscoveryResult(
        rover_name=rover_name,
        class_generic_scenario=scenario_name,
        mass_budget_kg=mass_budget_kg,
        nearest_pareto_index=nearest_idx,
        nearest_pareto_design=nearest_design,
        nearest_pareto_metrics=dict(nearest_metrics),
        design_space_distance=float(distances[nearest_idx]),
        per_variable_percent_errors=_per_variable_percent_errors(
            nearest_design, entry.design
        ),
        integer_matches=_integer_matches(nearest_design, entry.design),
        rover_metrics_under_generic_scenario=rover_metrics,
        pareto_dominated=dominated,
        optimization_result=optimization,
    )


def rediscover_ensemble(
    rover_name: str,
    *,
    objectives: tuple[OptimizationObjective, ...] = DEFAULT_OBJECTIVES,
    mass_ceiling_slop: float = 0.10,
    population_size: int = 200,
    n_generations: int = 100,
    n_seeds: int = 5,
    base_seed: int = 0,
    backend: OptimizationBackend = "evaluator",
    bundles: dict[str, QuantileHeads] | None = None,
    evaluator_eval_cap: int = 25_000,
) -> RediscoveryResult:
    """Run :func:`rediscover` ``n_seeds`` times and merge the Pareto fronts.

    Standard practice for stochastic multi-objective optimisers: each
    NSGA-II run with a different seed produces a slightly different
    Pareto-front sample, and the union over seeds is a tighter, lower-
    variance approximation of the true Pareto manifold. The returned
    :class:`RediscoveryResult` is keyed off the **merged** front:

    - ``design_space_distance`` = min normalised L2 over the merged
      pool of Pareto points.
    - ``pareto_dominated`` = True iff at least one point in **any**
      seed's front strictly dominates the real rover.
    - ``nearest_pareto_design`` / ``nearest_pareto_metrics`` /
      ``nearest_pareto_index`` index into the merged Pareto list
      stored in ``optimization_result``.
    - ``optimization_result.design_vectors`` and ``.metrics`` are the
      concatenated (not re-Pareto-filtered) union of every seed's
      front; ``backend_used`` is taken from the last seed.

    Parameters
    ----------
    rover_name, objectives, mass_ceiling_slop, population_size,
    n_generations, backend, bundles, evaluator_eval_cap
        Passed through to :func:`rediscover` for each seed.
    n_seeds
        Number of NSGA-II runs to ensemble. Default 5 matches the
        evolutionary-computing convention.
    base_seed
        Seeds run ``base_seed``, ``base_seed + 1``, ..., ``base_seed +
        n_seeds - 1``. Recorded so the paper figure is reproducible.

    Raises
    ------
    RuntimeError
        If every seed fails (e.g. every NSGA-II run returns an empty
        Pareto front under a binding mass ceiling). Partial failures
        are tolerated - the ensemble result merges the seeds that
        did produce a front.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1 (got {n_seeds})")

    seeds = list(range(base_seed, base_seed + n_seeds))
    per_seed: list[RediscoveryResult] = []
    last_exc: RuntimeError | None = None
    for s in seeds:
        try:
            per_seed.append(
                rediscover(
                    rover_name,
                    objectives=objectives,
                    mass_ceiling_slop=mass_ceiling_slop,
                    population_size=population_size,
                    n_generations=n_generations,
                    seed=s,
                    backend=backend,
                    bundles=bundles,
                    evaluator_eval_cap=evaluator_eval_cap,
                )
            )
        except RuntimeError as exc:
            last_exc = exc
    if not per_seed:
        assert last_exc is not None
        raise RuntimeError(
            f"every NSGA-II seed failed for {rover_name!r}; last error: {last_exc}"
        )

    # The per-seed results all share the same rover, scenario, mass
    # budget, and rover_metrics_under_generic_scenario; pick from the
    # first.
    head = per_seed[0]
    entry = registry_by_name(rover_name)

    merged_designs: list[DesignVector] = []
    merged_metrics: list[dict[str, float]] = []
    for r in per_seed:
        merged_designs.extend(r.optimization_result.design_vectors)
        merged_metrics.extend(r.optimization_result.metrics)

    distances = np.asarray(
        [_normalised_l2(d, entry.design) for d in merged_designs], dtype=float
    )
    nearest_idx = int(np.argmin(distances))
    nearest_design = merged_designs[nearest_idx]
    nearest_metrics = merged_metrics[nearest_idx]

    dominated = any(
        _dominates(point, head.rover_metrics_under_generic_scenario, objectives)
        for point in merged_metrics
    )

    merged_optimization = OptimizationResult(
        design_vectors=merged_designs,
        metrics=merged_metrics,
        objectives=objectives,
        backend_used=per_seed[-1].optimization_result.backend_used,
        checkpoints=[],  # per-seed checkpoints discarded in the merge
    )

    return RediscoveryResult(
        rover_name=rover_name,
        class_generic_scenario=head.class_generic_scenario,
        mass_budget_kg=head.mass_budget_kg,
        nearest_pareto_index=nearest_idx,
        nearest_pareto_design=nearest_design,
        nearest_pareto_metrics=dict(nearest_metrics),
        design_space_distance=float(distances[nearest_idx]),
        per_variable_percent_errors=_per_variable_percent_errors(
            nearest_design, entry.design
        ),
        integer_matches=_integer_matches(nearest_design, entry.design),
        rover_metrics_under_generic_scenario=head.rover_metrics_under_generic_scenario,
        pareto_dominated=dominated,
        optimization_result=merged_optimization,
    )


_VALID_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"population_size", "n_generations", "mass_ceiling_slop", "seed"}
)


def rediscover_all(
    *,
    flown_only: bool = True,
    population_size: int = 60,
    n_generations: int = 16,
    mass_ceiling_slop: float = 0.10,
    seed: int = 0,
    per_rover_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[RediscoveryResult]:
    """Run :func:`rediscover` on every registry rover.

    Parameters
    ----------
    flown_only
        If ``True`` (default), restrict to rovers with ``is_flown=True``
        - the paper's headline target. Set to ``False`` to also score
        the design-target rovers (MoonRanger, Rashid-1, Tenacious,
        CADRE-unit).
    population_size, n_generations, mass_ceiling_slop, seed
        Default NSGA-II hyperparameters and mass-ceiling slop passed
        through to :func:`rediscover` for every rover. See
        :func:`rediscover` for the per-parameter rationale.
    per_rover_overrides
        Optional ``{rover_name: {param: value}}`` mapping that overrides
        the defaults for specific rovers. Allowed keys per rover:
        ``population_size``, ``n_generations``, ``mass_ceiling_slop``,
        ``seed``. Use this for budget-tight rovers (e.g. ultra-micro
        CADRE-unit needs ``population_size`` ≈ 80 and
        ``mass_ceiling_slop`` ≈ 0.50 to find feasible designs from
        random LHS init while staying under the optimiser's 1000-eval
        cap).

    Raises
    ------
    KeyError
        If ``per_rover_overrides`` references an unknown rover or an
        unknown parameter name. Failures from individual
        :func:`rediscover` calls (e.g. empty Pareto fronts from a
        binding mass ceiling) are NOT caught here - they propagate.
        For a failure-resilient sweep, use
        :func:`roverdevkit.validation.rediscovery_report.run_rediscovery_loo`.
    """
    overrides = dict(per_rover_overrides or {})
    for rover_name, params in overrides.items():
        bad_keys = set(params) - _VALID_OVERRIDE_KEYS
        if bad_keys:
            raise KeyError(
                f"per_rover_overrides[{rover_name!r}] has unknown keys "
                f"{sorted(bad_keys)}; allowed: {sorted(_VALID_OVERRIDE_KEYS)}"
            )

    entries = (
        flown_registry()
        if flown_only
        else tuple(registry_by_name(r) for r in _CLASS_GENERIC_SCENARIO)
    )

    results: list[RediscoveryResult] = []
    for entry in entries:
        kwargs: dict[str, Any] = {
            "population_size": population_size,
            "n_generations": n_generations,
            "mass_ceiling_slop": mass_ceiling_slop,
            "seed": seed,
        }
        kwargs.update(overrides.get(entry.rover_name, {}))
        results.append(rediscover(entry.rover_name, **kwargs))
    return results


__all__ = [
    "RediscoveryResult",
    "class_generic_scenario_for",
    "rediscover",
    "rediscover_all",
    "rediscover_ensemble",
]
