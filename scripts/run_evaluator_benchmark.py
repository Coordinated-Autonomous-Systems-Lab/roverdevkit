"""Benchmark the analytical evaluator's per-mission wall-clock cost.

Reports the per-evaluation cost on each canonical paper scenario together with
the machine specification and the simulated mission duration, so that the
runtime claim in the paper can be stated relative to mission length and
reproduced on other hardware.

Designs are drawn uniformly from :data:`DESIGN_BOUNDS` and decoded with the
same ``_vector_to_design`` path NSGA-II uses, so the measured distribution is
the one the optimizer actually pays for, including infeasible designs.

Usage::

    python scripts/run_evaluator_benchmark.py --samples 500
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roverdevkit.mission.evaluator import evaluate  # noqa: E402
from roverdevkit.mission.scenarios import load_scenario  # noqa: E402
from roverdevkit.terramechanics.soils import get_soil_parameters  # noqa: E402
from roverdevkit.tradespace.optimizer import (  # noqa: E402
    DESIGN_BOUNDS,
    DESIGN_VARIABLES,
    _vector_to_design,
)

#: The four scenarios whose Pareto fronts are reported in the paper.
PAPER_SCENARIOS = (
    "equatorial_mare_traverse",
    "polar_prospecting",
    "crater_rim_survey",
    "highland_slope_capability",
)

#: NSGA-II settings used to produce the published fronts, for the
#: front-level cost projection.
FRONT_POPULATION = 50
FRONT_GENERATIONS = 60

SECONDS_PER_EARTH_DAY = 86_400.0


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _environment() -> dict[str, Any]:
    return {
        "cpu": _cpu_name(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "pymoo": _pkg_version("pymoo"),
        "note": "single-threaded, one core, no surrogate; times are wall clock",
    }


def _sample_designs(n: int, seed: int) -> list[Any]:
    lo = np.array([DESIGN_BOUNDS[name][0] for name in DESIGN_VARIABLES], dtype=float)
    hi = np.array([DESIGN_BOUNDS[name][1] for name in DESIGN_VARIABLES], dtype=float)
    rng = np.random.default_rng(seed)
    unit = rng.random((n, lo.size))
    return [_vector_to_design(row) for row in lo + unit * (hi - lo)]


def _benchmark_scenario(name: str, n_samples: int, seed: int) -> dict[str, Any]:
    scenario = load_scenario(name)
    soil = get_soil_parameters(scenario.soil_simulant)
    designs = _sample_designs(n_samples, seed)

    # Warm up so first-call import and cache effects stay out of the sample.
    for design in designs[: min(10, len(designs))]:
        try:
            evaluate(design, scenario, soil_override=soil)
        except Exception:  # noqa: BLE001
            pass

    # The optimizer catches evaluator failures (e.g. a fully buried wheel the
    # slip solver cannot balance) and substitutes a deeply-infeasible sentinel,
    # so time those calls separately rather than letting them skew the physics cost.
    per_call_ms: list[float] = []
    failed_ms: list[float] = []
    for design in designs:
        t0 = time.perf_counter()
        try:
            evaluate(design, scenario, soil_override=soil)
        except Exception:  # noqa: BLE001
            failed_ms.append((time.perf_counter() - t0) * 1e3)
        else:
            per_call_ms.append((time.perf_counter() - t0) * 1e3)

    per_call_ms.sort()
    median_ms = statistics.median(per_call_ms)
    duration_days = float(scenario.mission_duration_earth_days)
    simulated_s = duration_days * SECONDS_PER_EARTH_DAY

    return {
        "scenario": name,
        "mission_duration_earth_days": duration_days,
        "timesteps": round(duration_days * 24),
        "samples_evaluated": len(per_call_ms),
        "samples_failed": len(failed_ms),
        "failed_median_ms": statistics.median(failed_ms) if failed_ms else None,
        "median_ms": median_ms,
        "mean_ms": statistics.fmean(per_call_ms),
        "p05_ms": per_call_ms[int(0.05 * (len(per_call_ms) - 1))],
        "p95_ms": per_call_ms[int(0.95 * (len(per_call_ms) - 1))],
        "evaluations_per_second": 1e3 / median_ms,
        # How much simulated mission time one wall-clock second buys.
        "realtime_factor": simulated_s / (median_ms * 1e-3),
        "front_minutes": FRONT_POPULATION * FRONT_GENERATIONS * median_ms * 1e-3 / 60.0,
    }


#: Mission durations (Earth days) used to separate the mission-length-independent
#: fixed cost from the per-timestep cost of the traverse loop.
DURATION_SWEEP_DAYS = (1.0, 2.0, 5.0, 14.0, 30.0, 60.0, 120.0, 365.0)


def _duration_scaling(
    scenario_name: str, n_samples: int, seed: int
) -> dict[str, Any]:
    """Fit per-mission cost against simulated mission length.

    The slip-balance solve is loop-invariant and lifted out of the time loop, so
    cost is expected to be dominated by a fixed term. Reporting the fit lets the
    runtime claim be transferred to mission durations other than the ones here.
    """
    base = load_scenario(scenario_name)
    soil = get_soil_parameters(base.soil_simulant)
    designs = _sample_designs(n_samples, seed)

    points = []
    for days in DURATION_SWEEP_DAYS:
        scenario = base.model_copy(update={"mission_duration_earth_days": days})
        for design in designs[: min(10, len(designs))]:
            try:
                evaluate(design, scenario, soil_override=soil)
            except Exception:  # noqa: BLE001
                pass
        times = []
        for design in designs:
            t0 = time.perf_counter()
            try:
                evaluate(design, scenario, soil_override=soil)
            except Exception:  # noqa: BLE001
                continue
            times.append((time.perf_counter() - t0) * 1e3)
        points.append(
            {
                "mission_duration_earth_days": days,
                "timesteps": round(days * 24),
                "median_ms": statistics.median(times),
            }
        )

    hours = np.array([p["timesteps"] for p in points], dtype=float)
    ms = np.array([p["median_ms"] for p in points], dtype=float)
    slope, intercept = np.polyfit(hours, ms, 1)
    return {
        "scenario": scenario_name,
        "points": points,
        "fixed_cost_ms": float(intercept),
        "per_simulated_hour_us": float(slope * 1e3),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--scaling-scenario", default="equatorial_mare_traverse")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "evaluator_benchmark",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env = _environment()

    print(f"{env['cpu']}, {env['platform']}, Python {env['python']}")
    print(f"{args.samples} random designs per scenario, single core\n")
    header = f"{'scenario':<28}{'days':>6}{'median ms':>11}{'p95 ms':>9}{'eval/s':>9}{'x real time':>13}{'front min':>11}"
    print(header)
    print("-" * len(header))

    rows = [_benchmark_scenario(name, args.samples, args.seed) for name in PAPER_SCENARIOS]
    for r in rows:
        print(
            f"{r['scenario']:<28}{r['mission_duration_earth_days']:>6.0f}"
            f"{r['median_ms']:>11.1f}{r['p95_ms']:>9.1f}"
            f"{r['evaluations_per_second']:>9.0f}"
            f"{r['realtime_factor']:>13.2e}{r['front_minutes']:>11.1f}"
        )

    medians = [r["median_ms"] for r in rows]
    print(f"\nacross scenarios: median {statistics.median(medians):.1f} ms, "
          f"range {min(medians):.1f}-{max(medians):.1f} ms")

    scaling = _duration_scaling(args.scaling_scenario, args.samples, args.seed)
    print(f"\ncost vs mission length ({scaling['scenario']}):")
    for p in scaling["points"]:
        print(f"  {p['mission_duration_earth_days']:>6.0f} d"
              f"{p['timesteps']:>7} steps{p['median_ms']:>9.2f} ms")
    print(f"  fit: {scaling['fixed_cost_ms']:.2f} ms fixed"
          f" + {scaling['per_simulated_hour_us']:.2f} us per simulated hour")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "evaluator_benchmark.json"
    out_path.write_text(
        json.dumps(
            {
                "environment": env,
                "samples_per_scenario": args.samples,
                "seed": args.seed,
                "nsga2_front": {
                    "population_size": FRONT_POPULATION,
                    "generations": FRONT_GENERATIONS,
                },
                "scenarios": rows,
                "duration_scaling": scaling,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
