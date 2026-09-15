"""Tests for the §5.4 feasible-design null baseline.

Groups:

1. **Helpers.** Uniform sampling stays inside the box bounds; the
   feasibility gate honours each clause and excludes the (degenerate)
   thermal flag; the pairwise-distance helper matches a hand value; the
   unit-cube null is the mean pairwise L2 (~1.20).
2. **End-to-end (smoke budget).** A small-``max_full_evals`` run on one
   rover populates the feasible set and produces finite, ordered null
   statistics.
"""

from __future__ import annotations

import numpy as np

from roverdevkit.tradespace.optimizer import DESIGN_BOUNDS, DESIGN_VARIABLES
from roverdevkit.validation.rediscovery_baseline import (
    _DISTANCE_VARIABLES,
    UNIT_CUBE_RANDOM_PAIR,
    _is_feasible,
    _mean_pairwise_l2,
    _sample_designs,
    compute_feasible_baseline,
)


def test_unit_cube_constant() -> None:
    # The null is the *mean* pairwise L2 between uniform unit-cube points,
    # matched to the feasible-null estimator. It is strictly below the
    # closed-form RMS separation sqrt(d/6) (Jensen) and lands near 1.13 for
    # the d = 8 variables the evaluator consumes (wheelbase is excluded).
    d = len(_DISTANCE_VARIABLES)
    assert d == 8
    rms = float(np.sqrt(d / 6.0))
    assert UNIT_CUBE_RANDOM_PAIR < rms
    assert UNIT_CUBE_RANDOM_PAIR == 1.1320906765526066


def test_sample_designs_within_bounds() -> None:
    rng = np.random.default_rng(0)
    designs = _sample_designs(200, rng)
    assert len(designs) == 200
    for d in designs:
        for name in DESIGN_VARIABLES:
            if name == "mobility_architecture":
                assert d.mobility_architecture in ("rigid_4wheel", "rocker_bogie_6wheel")
                continue
            lo, hi = DESIGN_BOUNDS[name]
            assert lo - 1e-9 <= float(getattr(d, name)) <= hi + 1e-9
        assert int(d.n_wheels) in (4, 6)
        expected_wheels = 6 if d.mobility_architecture == "rocker_bogie_6wheel" else 4
        assert int(d.n_wheels) == expected_wheels


def _metrics(*, stalled=False, energy=10.0, rng_km=1.0, mass=5.0, thermal=False):
    return {
        "stalled": stalled,
        "energy_margin_raw_pct": energy,
        "range_km": rng_km,
        "total_mass_kg": mass,
        "thermal_survival": thermal,
    }


def test_feasibility_gate_clauses() -> None:
    # A working rover with a (degenerate) thermal_survival=False still
    # counts feasible: thermal is intentionally excluded.
    assert _is_feasible(_metrics(thermal=False), None)
    assert not _is_feasible(_metrics(stalled=True), None)
    assert not _is_feasible(_metrics(energy=-0.1), None)
    assert not _is_feasible(_metrics(rng_km=0.0), None)
    # Mass-ceiling clause only bites when a budget is supplied.
    assert _is_feasible(_metrics(mass=9.0), None)
    assert _is_feasible(_metrics(mass=9.0), 10.0)
    assert not _is_feasible(_metrics(mass=11.0), 10.0)


def test_mean_pairwise_l2_known_value() -> None:
    # Three points on a line at 0, 3, 4 in 1-D: pairwise dists 3, 4, 1.
    vectors = np.array([[0.0], [3.0], [4.0]])
    mean, median = _mean_pairwise_l2(vectors, np.random.default_rng(0))
    assert mean == (3.0 + 4.0 + 1.0) / 3.0
    assert median == 3.0


def test_compute_feasible_baseline_smoke() -> None:
    result = compute_feasible_baseline(
        "Pragyan", max_full_evals=80, seed=0, require_mass_ceiling=False
    )
    assert result.rover_name == "Pragyan"
    assert result.class_generic_scenario == "polar_micro"
    assert result.mass_budget_kg is None  # physical-viability mode
    assert result.n_full_evaluated == 80
    assert result.n_feasible > 0
    assert 0.0 < result.feasible_fraction <= 1.0
    # Null statistics are finite and the centroid distance is a sane
    # normalised-L2 magnitude (well under the 3-unit cube diagonal).
    assert result.feasible_random_pair_mean is not None
    assert 0.0 < result.feasible_random_pair_mean < 3.0
    assert result.rover_to_nearest_feasible_distance is not None
    # The nearest feasible draw is no farther than the centroid.
    assert (
        result.rover_to_nearest_feasible_distance
        <= result.rover_to_centroid_distance + 1e-9
    )
