"""Feasible-design null baseline for the Layer-5 rediscovery check.

The headline rediscovery metric (§5.4) reports each rover's normalised
design-space distance to the nearest optimiser Pareto point *relative to
a null baseline*. The unit-cube null is the **mean** pairwise L2 between
uniformly random points in the 8-D unit cube, ~1.13 (computed with the
same estimator as the feasible null below; note the closed-form
*root-mean-square* separation ``sqrt(9 / 6) ~= 1.22`` is slightly larger,
but is the RMS rather than the mean). That null is
**generous**: the unit cube is mostly filled with physically infeasible
designs (rovers that stall on the slope, freeze, run an energy deficit,
or bust the mass budget), so any optimiser that merely lands somewhere
in the small feasible sub-volume beats it trivially. A reviewer will
flag a ratio that rests on a null dominated by infeasible space.

This module builds the *tougher* null the paper outline calls for:
restrict the random comparison to **feasible** (physically viable)
designs only — the designs that produce a working rover, which is
exactly the "feasible space" the unit-cube null wrongly dilutes with
dead designs (stalled on the slope, running an energy deficit, making
no forward progress). For each registry rover we

1. resolve the same class-generic ``*_micro`` scenario, payload
   requirement, and panel orientation the rediscovery harness uses (so
   the comparison is apples-to-apples with the optimiser run);
2. draw designs uniformly from the optimiser's box bounds
   (:data:`roverdevkit.tradespace.optimizer.DESIGN_BOUNDS`);
3. full-evaluate up to ``max_full_evals`` of them under the analytical
   evaluator and keep the **feasible** designs (not stalled,
   non-negative mission-integrated energy balance, non-zero range);
4. report three N-stable feasibility-aware null statistics plus the
   sampling diagnostics:

   - ``feasible_random_pair_mean`` / ``feasible_random_pair_median`` —
     the mean / median pairwise normalised L2 *within* the feasible
     set. This is the direct, tougher analogue of the ~1.13 unit-cube
     null: the typical separation between two random *feasible* rovers.
   - ``rover_to_centroid_distance`` — the rover's distance to the
     feasible-region centroid (the "typical feasible design").
   - ``rover_to_nearest_feasible_distance`` — the rover's distance to
     the single nearest of the ``n_feasible`` random feasible draws
     (N-dependent; reported for completeness, not as the headline
     null).

The rediscovery ratio can then be reported against **both** nulls:
``design_space_distance / UNIT_CUBE_RANDOM_PAIR`` (unit cube, ~1.13) and
``design_space_distance / feasible_random_pair_mean`` (feasible region).
Both nulls are mean pairwise distances, so the comparison is
apples-to-apples; the feasible null is the defensible number.

Feasibility definition
----------------------
By default a sampled design counts as feasible iff, under the rover's
class-generic scenario:

- ``stalled is False`` — the drivetrain develops the drawbar pull and
  torque to climb the scenario's worst-case slope;
- ``energy_margin_raw_pct >= 0`` — non-negative mission-integrated
  energy balance (generation covers consumption);
- ``range_km > 0`` — the traverse loop makes forward progress.

``thermal_survival`` is intentionally excluded: under the class-generic
``*_micro`` scenarios it is degenerate (``False`` for every design, the
real registry rovers included), so gating on it would empty the
feasible set and would not match the rediscovery NSGA-II run, which
carries no thermal constraint.

Empirically this physically-feasible region fills most of the box
(``feasible_fraction`` ~0.77-0.92 across the registry), so its
random-pair null comes out at ~1.10 — only marginally below the ~1.13
unit-cube value. That is itself the reportable result: the rediscovery
ratio is **not** an artifact of a null dominated by infeasible space.

An optional stricter mode (``require_mass_ceiling=True``) additionally
requires ``total_mass_kg <= modelled_rover_mass * (1 + mass_ceiling_slop)``
— the same budget NSGA-II carried — and cheaply mass-pre-filters the
draw before the full evaluator. It is a sensitivity mode only: for the
ultra-micro rovers (CADRE-unit ~2 kg, Tenacious ~5 kg) the in-budget
feasible corner is effectively measure-zero under uniform sampling
(0 hits in 3x10^5 draws), so the optimiser reaching it at all is itself
evidence that random sampling overstates the achievable spread there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from roverdevkit.mass.parametric_mers import (
    MassModelParams,
    estimate_mass_from_design,
)
from roverdevkit.mission.evaluator import evaluate as evaluator_evaluate
from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.schema import DesignVector
from roverdevkit.tradespace.optimizer import (
    DESIGN_BOUNDS,
    DESIGN_VARIABLES,
    _vector_to_design,
)
from roverdevkit.validation.rover_registry import (
    flown_registry,
    registry_by_name,
)
from roverdevkit.validation.rover_rediscovery import (
    _CLASS_GENERIC_SCENARIO,
    _DISTANCE_VARIABLES,
    _evaluate_rover_under,
    _normalised_l2,
    _normalised_vector,
    _scenario_panel_orientation,
    class_generic_scenario_for,
)

_LOG = logging.getLogger(__name__)

# Cap on the number of feasible designs used in the O(M^2) pairwise-mean
# computation. With more than this many feasible draws we subsample
# (seeded) so the pairwise statistic stays cheap; the mean/median of a
# 1000-point subsample is a tight estimate of the full-set value.
_MAX_PAIRWISE_SAMPLES: int = 1000


@dataclass(frozen=True)
class FeasibleBaselineResult:
    """Feasible-design null statistics for one registry rover.

    Attributes
    ----------
    rover_name
        Registry key, e.g. ``"Pragyan"``.
    class_generic_scenario
        ``*_micro`` scenario used (matches the rediscovery harness).
    mass_budget_kg
        Mass-ceiling budget (``modelled_rover_mass * (1 + slop)``).
        ``None`` when ``require_mass_ceiling=False``.
    n_sampled
        Number of uniform random designs drawn from the box bounds.
    n_mass_feasible
        How many draws survived the cheap mass-model pre-filter
        (``modelled_mass <= mass_budget``). ``n_sampled`` when
        ``require_mass_ceiling=False``.
    n_full_evaluated
        How many mass-feasible draws were run through the full
        evaluator (capped at ``max_full_evals``; subsampled when the
        mass-feasible set is larger).
    n_feasible
        How many full-evaluated designs were feasible under the rover's
        scenario.
    feasible_fraction
        Estimated share of the **box** that is feasible for this rover's
        scenario/budget, ``(n_mass_feasible / n_sampled) *
        (n_feasible / n_full_evaluated)``. The smaller this is, the more
        generous the unit-cube null was.
    rover_to_centroid_distance
        Normalised L2 from the real rover's design to the feasible
        region's centroid. ``None`` if no feasible designs were drawn.
    rover_to_nearest_feasible_distance
        Normalised L2 from the real rover to the single nearest random
        feasible design (N-dependent). ``None`` if none feasible.
    feasible_random_pair_mean, feasible_random_pair_median
        Mean / median pairwise normalised L2 within the feasible set —
        the tougher analogue of :data:`UNIT_CUBE_RANDOM_PAIR`. ``None``
        if fewer than two feasible designs were drawn.
    unit_cube_random_pair
        :data:`UNIT_CUBE_RANDOM_PAIR`, stored per-row for convenience.
    seed
        RNG seed used for the draw (reproducibility).
    """

    rover_name: str
    class_generic_scenario: str
    mass_budget_kg: float | None
    n_sampled: int
    n_mass_feasible: int
    n_full_evaluated: int
    n_feasible: int
    feasible_fraction: float
    rover_to_centroid_distance: float | None
    rover_to_nearest_feasible_distance: float | None
    feasible_random_pair_mean: float | None
    feasible_random_pair_median: float | None
    unit_cube_random_pair: float
    seed: int


def _sample_designs(n_samples: int, rng: np.random.Generator) -> list[DesignVector]:
    """Draw ``n_samples`` designs uniformly from the optimiser box bounds.

    Uses the same field order and integer-repair logic
    (:func:`roverdevkit.tradespace.optimizer._vector_to_design`) the
    NSGA-II runner applies, so the sampled designs live in exactly the
    space the optimiser searches.
    """
    lo = np.asarray([DESIGN_BOUNDS[name][0] for name in DESIGN_VARIABLES], dtype=float)
    hi = np.asarray([DESIGN_BOUNDS[name][1] for name in DESIGN_VARIABLES], dtype=float)
    raw = rng.uniform(lo, hi, size=(n_samples, len(DESIGN_VARIABLES)))
    return [_vector_to_design(row) for row in raw]


def _is_feasible(
    metrics: dict[str, float],
    mass_budget_kg: float | None,
) -> bool:
    """Physical-viability + (optional) mass-ceiling feasibility gate.

    The ``thermal_survival`` flag is deliberately **not** consulted: it
    is degenerate under the class-generic ``*_micro`` scenarios (it
    fires ``False`` for every design, the real registry rovers
    included), so gating on it would empty the feasible set and would
    not match the rediscovery harness, whose NSGA-II run does not carry
    a thermal constraint either. The feasibility classifier the
    surrogate trains is likewise keyed on ``stalled`` alone
    (:data:`roverdevkit.surrogate.features.FEASIBILITY_COLUMN`).
    """
    if metrics["stalled"]:
        return False
    if metrics["energy_margin_raw_pct"] < 0.0:
        return False
    if metrics["range_km"] <= 0.0:
        return False
    if mass_budget_kg is not None and metrics["total_mass_kg"] > mass_budget_kg:
        return False
    return True


def _mean_pairwise_l2(vectors: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Mean and median pairwise Euclidean distance over ``vectors``.

    ``vectors`` is an ``(M, d)`` array of already-normalised design
    vectors. Subsamples to :data:`_MAX_PAIRWISE_SAMPLES` rows when
    ``M`` is large to keep the computation O(M^2) on a bounded M.
    """
    m = vectors.shape[0]
    if m > _MAX_PAIRWISE_SAMPLES:
        idx = rng.choice(m, size=_MAX_PAIRWISE_SAMPLES, replace=False)
        vectors = vectors[idx]
        m = _MAX_PAIRWISE_SAMPLES
    # Upper-triangular pairwise distances (exclude the zero diagonal).
    diffs = vectors[:, None, :] - vectors[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(m, k=1)
    pair = dists[iu]
    return float(np.mean(pair)), float(np.median(pair))


def _unit_cube_random_pair_mean(
    d: int, *, n: int = _MAX_PAIRWISE_SAMPLES, seed: int = 0
) -> float:
    """Mean pairwise normalised L2 between uniform points in the unit cube.

    Computed with the **same estimator** (:func:`_mean_pairwise_l2`) as the
    feasible-region null so the two nulls are directly comparable mean
    pairwise distances. The closed-form *root-mean-square* separation
    between two i.i.d. uniform points in the d-cube is ``sqrt(d / 6)``
    (~1.225 for d = 9), but that is the RMS, not the mean: by Jensen the
    mean separation is strictly smaller (~1.13 for d = 9). The feasible
    null reports a mean, so we match it here rather than using the RMS.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n, d))
    mean, _ = _mean_pairwise_l2(pts, rng)
    return mean


# Mean pairwise normalised L2 between uniform random points in the
# d-dimensional unit cube (d = len(_DISTANCE_VARIABLES) = 8), computed
# with the same estimator as the feasible null. ~1.13; this is the
# random-pair null the feasible baseline is compared against.
UNIT_CUBE_RANDOM_PAIR: float = _unit_cube_random_pair_mean(len(_DISTANCE_VARIABLES))


def compute_feasible_baseline(
    rover_name: str,
    *,
    n_samples: int = 200_000,
    max_full_evals: int = 3000,
    seed: int = 0,
    mass_ceiling_slop: float = 0.10,
    require_mass_ceiling: bool = False,
) -> FeasibleBaselineResult:
    """Compute the feasible-design null statistics for one rover.

    Parameters
    ----------
    rover_name
        Registry key, e.g. ``"Pragyan"`` or ``"CADRE-unit"``.
    n_samples
        Number of uniform random designs to draw from the optimiser box
        bounds. These are first screened by the cheap mass-model
        pre-filter; only mass-feasible draws reach the full evaluator,
        so this can be large (the feasible region is a small slice of
        the box).
    max_full_evals
        Cap on the number of mass-feasible draws run through the full
        (~25 ms) evaluator. When the mass-feasible set exceeds this, a
        seeded subsample is taken; ``feasible_fraction`` corrects for
        the subsample.
    seed
        RNG seed for the draw and subsamples.
    mass_ceiling_slop
        Fractional slop above the rover's modelled total mass for the
        feasibility mass ceiling, matching
        :func:`roverdevkit.validation.rover_rediscovery.rediscover`.
        Ignored when ``require_mass_ceiling=False``.
    require_mass_ceiling
        If ``False`` (default) feasibility is physical viability only
        (not stalled, energy balance, range) — the robust per-scenario
        null computable for every rover. If ``True`` a feasible design
        must *also* sit within the rover's mass-ceiling budget; this is
        a stricter sensitivity mode that is **not uniformly sampleable
        for the ultra-micro rovers** (CADRE-unit, Tenacious), whose
        in-budget feasible corner is effectively measure-zero under
        uniform sampling — NSGA-II only reaches it by directed search.

    Returns
    -------
    FeasibleBaselineResult
    """
    entry = registry_by_name(rover_name)
    scenario_name = class_generic_scenario_for(rover_name)
    scenario = load_scenario(scenario_name).model_copy(
        update={
            "payload_mass_kg": entry.scenario.payload_mass_kg,
            "payload_power_w": entry.scenario.payload_power_w,
        }
    )

    mass_budget_kg: float | None = None
    if require_mass_ceiling:
        rover_metrics = _evaluate_rover_under(entry, scenario)
        mass_budget_kg = float(rover_metrics["total_mass_kg"]) * (1.0 + mass_ceiling_slop)

    panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(scenario)
    payload_mass = scenario.payload_mass_kg
    payload_power = scenario.payload_power_w

    rng = np.random.default_rng(seed)

    # With the mass ceiling on, the dominant feasibility cut is the
    # budget; draw a large pool and screen it on the cheap (microsecond)
    # mass model before paying for the ~25 ms full evaluator. Without
    # the mass ceiling, physical viability fills most of the box, so we
    # draw only what we will full-evaluate (no point constructing a
    # large pool we immediately subsample).
    n_draw = n_samples if mass_budget_kg is not None else min(n_samples, max_full_evals)
    designs = _sample_designs(n_draw, rng)

    mass_params = MassModelParams()
    if mass_budget_kg is not None:
        mass_feasible = [
            d
            for d in designs
            if estimate_mass_from_design(
                d, params=mass_params, payload_mass_kg=payload_mass
            ).total_kg
            <= mass_budget_kg
        ]
    else:
        mass_feasible = designs
    n_sampled = n_draw
    n_mass_feasible = len(mass_feasible)

    # Subsample the mass-feasible set down to the full-eval cap.
    to_eval = mass_feasible
    if n_mass_feasible > max_full_evals:
        idx = rng.choice(n_mass_feasible, size=max_full_evals, replace=False)
        to_eval = [mass_feasible[i] for i in idx]
    n_full_evaluated = len(to_eval)

    feasible_designs: list[DesignVector] = []
    for design in to_eval:
        try:
            m = evaluator_evaluate(
                design,
                scenario,
                gravity_m_per_s2=entry.gravity_m_per_s2,
                payload_mass_kg=payload_mass,
                payload_power_w=payload_power,
                panel_tilt_deg=panel_tilt_deg,
                panel_azimuth_deg=panel_azimuth_deg,
            )
        except Exception:  # noqa: BLE001 -- a failed eval is simply infeasible
            continue
        metrics = {
            "range_km": float(m.range_km),
            "energy_margin_raw_pct": float(m.energy_margin_raw_pct),
            "total_mass_kg": float(m.total_mass_kg),
            "thermal_survival": bool(m.thermal_survival),
            "stalled": bool(m.stalled),
        }
        if _is_feasible(metrics, mass_budget_kg):
            feasible_designs.append(design)

    n_feasible = len(feasible_designs)
    # Estimated feasible share of the box, correcting for the mass
    # pre-filter and the full-eval subsample.
    if n_sampled and n_full_evaluated:
        feasible_fraction = (n_mass_feasible / n_sampled) * (
            n_feasible / n_full_evaluated
        )
    else:
        feasible_fraction = 0.0

    rover_to_centroid: float | None = None
    rover_to_nearest: float | None = None
    pair_mean: float | None = None
    pair_median: float | None = None

    if n_feasible >= 1:
        rover_vec = _normalised_vector(entry.design)
        feasible_mat = np.asarray(
            [_normalised_vector(d) for d in feasible_designs], dtype=float
        )
        centroid = feasible_mat.mean(axis=0)
        rover_to_centroid = float(np.linalg.norm(rover_vec - centroid))
        rover_to_nearest = float(
            min(_normalised_l2(d, entry.design) for d in feasible_designs)
        )
    if n_feasible >= 2:
        pair_mean, pair_median = _mean_pairwise_l2(feasible_mat, rng)

    _LOG.info(
        "feasible-baseline %s: %d feasible of %d full-evals "
        "(%d mass-feasible / %d drawn; est box feas=%.3f%%), pair_mean=%s",
        rover_name,
        n_feasible,
        n_full_evaluated,
        n_mass_feasible,
        n_sampled,
        feasible_fraction * 100.0,
        f"{pair_mean:.3f}" if pair_mean is not None else "n/a",
    )

    return FeasibleBaselineResult(
        rover_name=rover_name,
        class_generic_scenario=scenario_name,
        mass_budget_kg=mass_budget_kg,
        n_sampled=n_sampled,
        n_mass_feasible=n_mass_feasible,
        n_full_evaluated=n_full_evaluated,
        n_feasible=n_feasible,
        feasible_fraction=feasible_fraction,
        rover_to_centroid_distance=rover_to_centroid,
        rover_to_nearest_feasible_distance=rover_to_nearest,
        feasible_random_pair_mean=pair_mean,
        feasible_random_pair_median=pair_median,
        unit_cube_random_pair=UNIT_CUBE_RANDOM_PAIR,
        seed=seed,
    )


def compute_feasible_baseline_all(
    *,
    flown_only: bool = True,
    n_samples: int = 200_000,
    max_full_evals: int = 3000,
    seed: int = 0,
    mass_ceiling_slop: float = 0.10,
    require_mass_ceiling: bool = False,
    per_rover_mass_ceiling_slop: dict[str, float] | None = None,
) -> list[FeasibleBaselineResult]:
    """Run :func:`compute_feasible_baseline` over the registry.

    Parameters
    ----------
    flown_only
        If ``True`` (default), restrict to ``is_flown=True`` rovers; set
        ``False`` to include the design-target rovers (MoonRanger,
        Rashid-1, Tenacious, CADRE-unit), matching the rediscovery
        sweep's scope.
    n_samples, max_full_evals, seed, mass_ceiling_slop, require_mass_ceiling
        Passed through to :func:`compute_feasible_baseline`.
    per_rover_mass_ceiling_slop
        Optional ``{rover_name: slop}`` override so the feasible region
        for budget-tight rovers (e.g. CADRE-unit at slop 0.50) matches
        the slop NSGA-II used in the rediscovery sweep.
    """
    overrides = dict(per_rover_mass_ceiling_slop or {})
    if flown_only:
        entries = flown_registry()
    else:
        entries = tuple(registry_by_name(r) for r in _CLASS_GENERIC_SCENARIO)

    results: list[FeasibleBaselineResult] = []
    for entry in entries:
        slop = overrides.get(entry.rover_name, mass_ceiling_slop)
        results.append(
            compute_feasible_baseline(
                entry.rover_name,
                n_samples=n_samples,
                max_full_evals=max_full_evals,
                seed=seed,
                mass_ceiling_slop=slop,
                require_mass_ceiling=require_mass_ceiling,
            )
        )
    return results


__all__ = [
    "UNIT_CUBE_RANDOM_PAIR",
    "FeasibleBaselineResult",
    "compute_feasible_baseline",
    "compute_feasible_baseline_all",
]
