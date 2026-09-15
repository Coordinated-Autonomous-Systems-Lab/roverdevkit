"""Layer-5 rediscovery harness and artifact writer.

A thin orchestration layer on top of
:func:`roverdevkit.validation.rover_rediscovery.rediscover` that:

1. Loops over the registry under the class-generic ``*_micro``
   scenarios with per-rover budget overrides for ultra-micro rovers.
2. Catches per-rover :class:`RuntimeError` (e.g. empty Pareto fronts
   from binding mass ceilings) so one rover's feasibility failure
   does not abort the whole sweep.
3. Emits ``reports/rediscovery_loo/`` with a summary CSV, per-rover
   JSON detail dumps, and a markdown rollup of the paper-side
   acceptance gate.

Leakage-free by construction (not cross-validation)
----------------------------------------------------
This is a rediscovery comparison, **not** leave-one-out
cross-validation: nothing is fit to the rover registry, so there is
no per-rover hold-out/refit loop. The bottom-up mass model's
specific-mass coefficients live in :class:`MassModelParams` and are
cited from external space-hardware sources (SMAD, AIAA S-120A,
vendor catalogues) -
**none** of them are regressed against the rover registry. The
registry is used downstream as a cross-check, not as training data,
so each rover is rediscovered without any information leaking from
the others.

The only piece of rover-specific information that enters each
rediscovery run is the rover's published total mass (used as the
mass-ceiling constraint for NSGA-II). The class-generic ``*_micro`` scenarios
break the canonical ``polar_prospecting``/etc. δ_ops anchor leakage
(see :mod:`roverdevkit.validation.rover_rediscovery`), and the rover's
design vector is used **only** after the optimiser returns for
distance scoring - it never enters the search.

Paper-side acceptance gate (informational, not pass/fail)
----------------------------------------------------------
The rollup reports two complementary signals per rover:

- ``design_space_distance`` - normalised L2 over the nine continuous
  design variables between the real rover and the nearest Pareto
  point. **Primary signal.** Acceptance target: median across the
  flown registry within ~0.5 of the design-space cube diagonal
  (≈ 1.0 for a uniform random pair after normalisation).
- ``pareto_dominated`` - whether any Pareto point strictly dominates
  the real rover on all three objectives under the class-generic
  scenario. **Secondary signal**, informative not pass/fail: as A3
  surfaced, several real rovers flip to "dominated" once the
  canonical δ_ops anchor is removed, indicating they were over-
  designed for class-neutral assumptions and conservative for their
  own actual ops profile.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.tradespace.optimizer import DEFAULT_OBJECTIVES, OptimizationBackend
from roverdevkit.validation.rover_registry import (
    flown_registry,
    registry_by_name,
)
from roverdevkit.validation.rover_rediscovery import (
    _CLASS_GENERIC_SCENARIO,
    RediscoveryResult,
    rediscover,
    rediscover_ensemble,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-rover NSGA-II budget overrides for the paper run
# ---------------------------------------------------------------------------


DEFAULT_PER_ROVER_OVERRIDES: Mapping[str, Mapping[str, Any]] = {
    "CADRE-unit": {
        # CADRE's modelled total mass under the bottom-up model is
        # ~4.0 kg (the model over-predicts CADRE's published 2.0 kg
        # by ~100 % because the specific-mass calibration regime is
        # 5-50 kg micro-rovers, see
        # roverdevkit.mass.validation). At the default
        # mass_ceiling_slop=0.10 the budget is ~4.4 kg, which random
        # LHS init at pop=60 cannot reliably hit - every individual
        # ends up infeasible and rediscover() raises.
        # Bumping pop to 80 doubles the chance of a feasible
        # initial draw; widening the slop to 0.50 (->budget ~6 kg)
        # gives the optimiser meaningful room to construct a Pareto
        # front. Total evals 80*12=960, just under the optimiser's
        # 1000-eval safety cap. Documented as a methodological
        # caveat in the paper-side report; the CADRE rediscovery
        # number is reported in a separate column from the
        # uniform-budget results so a reviewer can read both
        # signals.
        "population_size": 80,
        "n_generations": 12,
        "mass_ceiling_slop": 0.50,
    },
}
"""Per-rover NSGA-II hyperparameter / slop overrides for the paper run.

Empty entries inherit the rediscovery defaults. CADRE-unit is the only entry
today because it is the only registry rover whose modelled mass sits
in the bottom-up model's out-of-regime zone (<5 kg). Future ultra-
micro additions will likely need their own override entries.
"""


# ---------------------------------------------------------------------------
# Aggregate result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RediscoveryRunSummary:
    """Outcome of a rediscovery sweep over the registry.

    Attributes
    ----------
    results
        One :class:`RediscoveryResult` per rover that succeeded.
    failures
        ``{rover_name: error_message}`` for rovers whose
        :func:`rediscover` call raised. Empty when every rover
        succeeded.
    seed
        Master RNG seed used for the sweep (per-rover overrides may
        bump individual rovers' seeds; recorded for reproducibility).
    default_kwargs
        The default NSGA-II hyperparameters used for rovers without
        per-rover overrides. Snapshotted so the artifact writer can
        record them in the markdown rollup.
    per_rover_overrides
        Snapshot of the per-rover override map used for this sweep.
    """

    results: list[RediscoveryResult]
    failures: dict[str, str]
    seed: int
    default_kwargs: dict[str, Any]
    per_rover_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def all_succeeded(self) -> bool:
        return not self.failures

    def by_rover(self, rover_name: str) -> RediscoveryResult:
        for r in self.results:
            if r.rover_name == rover_name:
                return r
        raise KeyError(
            f"no successful RediscoveryResult for {rover_name!r}; "
            f"sweep contains: {[r.rover_name for r in self.results]}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_rediscovery_loo(
    *,
    flown_only: bool = True,
    seed: int = 0,
    default_population_size: int = 60,
    default_n_generations: int = 16,
    default_mass_ceiling_slop: float = 0.10,
    per_rover_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    n_seeds: int = 1,
    backend: OptimizationBackend = "evaluator",
    bundles: dict[str, QuantileHeads] | None = None,
    evaluator_eval_cap: int = 1000,
) -> RediscoveryRunSummary:
    """Run rediscovery over every registry rover.

    Wraps :func:`roverdevkit.validation.rover_rediscovery.rediscover`
    (or :func:`rediscover_ensemble` when ``n_seeds > 1``) with
    per-rover failure capture: an empty-Pareto-front
    :class:`RuntimeError` on one rover does not abort the whole sweep,
    it is recorded in :attr:`RediscoveryRunSummary.failures` and the
    next rover is run.

    Parameters
    ----------
    flown_only
        If ``True`` (default), restrict to ``is_flown=True`` rovers.
        Set to ``False`` to also score the design-target rovers
        (MoonRanger, Rashid-1, Tenacious, CADRE-unit).
    seed
        Master RNG seed (base seed when ``n_seeds > 1``).
    default_population_size, default_n_generations, default_mass_ceiling_slop
        NSGA-II defaults for rovers without per-rover overrides.
    per_rover_overrides
        Optional override map (see :data:`DEFAULT_PER_ROVER_OVERRIDES`
        for the paper-run defaults). Pass ``{}`` to disable all
        overrides.
    n_seeds
        Number of NSGA-II seeds to ensemble per rover. ``1`` (default)
        preserves the single-seed historical behaviour; ``>= 2``
        routes through :func:`rediscover_ensemble` and merges Pareto
        fronts across seeds.
    backend
        ``"evaluator"`` (default) or ``"surrogate"``. The surrogate
        backend requires ``bundles`` and runs ~200x faster, enabling
        100k+ effective evaluations per rover in a few seconds; expect
        small but non-zero approximation error from the quantile-XGB
        heads. Designs sampled outside the v4 LHS training support
        (chassis < 3 kg, torque < 0.3 Nm, battery < 20 Wh) will be
        extrapolated.
    bundles
        Required iff ``backend == "surrogate"``. Map ``{target ->
        QuantileHeads}`` produced by
        :mod:`roverdevkit.surrogate.uncertainty`.
    evaluator_eval_cap
        Safety cap on the evaluator-backed NSGA-II runner per seed.
        Defaults to 1 000 (webapp value); raise to 10 000 or higher
        for high-budget validation runs.

    Returns
    -------
    RediscoveryRunSummary
        Successes + per-rover failures + the kwargs snapshot used.
    """
    overrides = (
        dict(per_rover_overrides)
        if per_rover_overrides is not None
        else {k: dict(v) for k, v in DEFAULT_PER_ROVER_OVERRIDES.items()}
    )

    if flown_only:
        entries = flown_registry()
    else:
        entries = tuple(registry_by_name(r) for r in _CLASS_GENERIC_SCENARIO)

    results: list[RediscoveryResult] = []
    failures: dict[str, str] = {}
    for entry in entries:
        kwargs: dict[str, Any] = {
            "population_size": default_population_size,
            "n_generations": default_n_generations,
            "mass_ceiling_slop": default_mass_ceiling_slop,
            "seed": seed,
        }
        kwargs.update(overrides.get(entry.rover_name, {}))
        common = {
            "backend": backend,
            "bundles": bundles,
            "evaluator_eval_cap": evaluator_eval_cap,
        }
        _LOG.info(
            "rediscover %s with %s; backend=%s, n_seeds=%d",
            entry.rover_name,
            kwargs,
            backend,
            n_seeds,
        )
        try:
            if n_seeds == 1:
                result = rediscover(entry.rover_name, **kwargs, **common)
            else:
                ens_kwargs = {
                    "objectives": DEFAULT_OBJECTIVES,
                    "mass_ceiling_slop": kwargs["mass_ceiling_slop"],
                    "population_size": kwargs["population_size"],
                    "n_generations": kwargs["n_generations"],
                    "n_seeds": n_seeds,
                    "base_seed": kwargs["seed"],
                }
                result = rediscover_ensemble(entry.rover_name, **ens_kwargs, **common)
        except RuntimeError as exc:
            _LOG.warning("rediscover %s failed: %s", entry.rover_name, exc)
            failures[entry.rover_name] = str(exc)
            continue
        results.append(result)

    default_kwargs = {
        "population_size": default_population_size,
        "n_generations": default_n_generations,
        "mass_ceiling_slop": default_mass_ceiling_slop,
        "seed": seed,
        "n_seeds": n_seeds,
        "backend": backend,
        "evaluator_eval_cap": evaluator_eval_cap,
    }
    return RediscoveryRunSummary(
        results=results,
        failures=failures,
        seed=seed,
        default_kwargs=default_kwargs,
        per_rover_overrides={k: dict(v) for k, v in overrides.items()},
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


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


def _abs_err_stats(per_var_pct: Mapping[str, float]) -> dict[str, float]:
    """Median / max / argmax of absolute per-variable percent errors."""
    abs_errs = {var: abs(err) for var, err in per_var_pct.items()}
    max_var = max(abs_errs, key=lambda v: abs_errs[v])
    return {
        "abs_err_median_pct": float(median(abs_errs.values())),
        "abs_err_max_pct": float(abs_errs[max_var]),
        "abs_err_max_var": max_var,
    }


def summarize_results(summary: RediscoveryRunSummary) -> pd.DataFrame:
    """One-row-per-rover summary table.

    Schema (column → meaning):

    - ``rover_name``: registry key
    - ``is_flown``: flown vs design-target
    - ``class_generic_scenario``: which ``*_micro`` scenario was used
    - ``mass_modelled_kg``: rover's total mass under the bottom-up
      model evaluated against the class-generic scenario
    - ``mass_budget_kg``: NSGA-II constraint ceiling (modelled × (1+slop))
    - ``pareto_front_size``: how many feasible Pareto points the
      optimiser returned
    - ``design_space_distance``: normalised L2 (over the 9 continuous
      design variables) between the real rover and the nearest Pareto
      point. **Primary signal.**
    - ``pareto_dominated``: whether any Pareto point strictly
      dominates the real rover under the class-generic scenario.
      **Secondary signal**, informative.
    - ``abs_err_median_pct``, ``abs_err_max_pct``, ``abs_err_max_var``:
      median / max / argmax of |per-variable % error| across the
      9 continuous design variables.
    - ``n_wheels_matches``, ``grouser_count_matches``: integer-variable
      exact-match flags (not rolled into the L2 because there is no
      meaningful continuous distance between 4 and 6 wheels).
    - ``population_size``, ``n_generations``, ``mass_ceiling_slop``:
      NSGA-II hyperparameters used for this rover (default or override).

    Rovers whose :func:`rediscover` call raised are **not** in the
    DataFrame; see :attr:`RediscoveryRunSummary.failures` for them.
    """
    rows: list[dict[str, Any]] = []
    for result in summary.results:
        rover_name = result.rover_name
        entry = registry_by_name(rover_name)
        kwargs = {**summary.default_kwargs, **summary.per_rover_overrides.get(rover_name, {})}
        stats = _abs_err_stats(result.per_variable_percent_errors)
        rows.append(
            {
                "rover_name": rover_name,
                "is_flown": entry.is_flown,
                "class_generic_scenario": result.class_generic_scenario,
                "mass_modelled_kg": float(
                    result.rover_metrics_under_generic_scenario["total_mass_kg"]
                ),
                "mass_budget_kg": float(result.mass_budget_kg),
                "pareto_front_size": int(len(result.optimization_result.design_vectors)),
                "design_space_distance": float(result.design_space_distance),
                "pareto_dominated": bool(result.pareto_dominated),
                "abs_err_median_pct": stats["abs_err_median_pct"],
                "abs_err_max_pct": stats["abs_err_max_pct"],
                "abs_err_max_var": stats["abs_err_max_var"],
                "n_wheels_matches": bool(result.integer_matches["n_wheels"]),
                "grouser_count_matches": bool(result.integer_matches["grouser_count"]),
                "population_size": int(kwargs["population_size"]),
                "n_generations": int(kwargs["n_generations"]),
                "mass_ceiling_slop": float(kwargs["mass_ceiling_slop"]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------


def _result_to_json_payload(result: RediscoveryResult) -> dict[str, Any]:
    """Per-rover detail dump. Includes the full Pareto front."""
    payload: dict[str, Any] = {
        "rover_name": result.rover_name,
        "class_generic_scenario": result.class_generic_scenario,
        "mass_budget_kg": float(result.mass_budget_kg),
        "design_space_distance": float(result.design_space_distance),
        "pareto_dominated": bool(result.pareto_dominated),
        "rover_metrics_under_generic_scenario": {
            k: float(v) for k, v in result.rover_metrics_under_generic_scenario.items()
        },
        "nearest_pareto_index": int(result.nearest_pareto_index),
        "nearest_pareto_design": result.nearest_pareto_design.model_dump(),
        "nearest_pareto_metrics": {
            k: float(v) for k, v in result.nearest_pareto_metrics.items()
        },
        "per_variable_percent_errors": {
            k: float(v) for k, v in result.per_variable_percent_errors.items()
        },
        "integer_matches": {k: bool(v) for k, v in result.integer_matches.items()},
        "pareto_front": [
            {
                "design": d.model_dump(),
                "metrics": {k: float(v) for k, v in m.items()},
            }
            for d, m in zip(
                result.optimization_result.design_vectors,
                result.optimization_result.metrics,
                strict=True,
            )
        ],
    }
    return payload


def _df_to_markdown_table(df: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Render ``df`` as a pipe-delimited Markdown table.

    Avoids the ``df.to_markdown()`` path which optionally depends on
    the ``tabulate`` package - we want this writer to work in a clean
    install without pulling another dep just for one report.
    """
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, bool):
                cells.append(str(v))
            elif isinstance(v, (int,)):
                cells.append(str(v))
            elif isinstance(v, float):
                cells.append(float_format.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _markdown_for_summary(
    df: pd.DataFrame,
    summary: RediscoveryRunSummary,
) -> str:
    """Human-readable rollup of the rediscovery sweep."""
    kw = summary.default_kwargs
    lines: list[str] = [
        "# Layer-5 rediscovery validation (rediscovery sweep)",
        "",
        f"- Master seed: `{summary.seed}`",
        f"- Default NSGA-II: pop={kw['population_size']}, "
        f"gen={kw['n_generations']}, "
        f"mass_ceiling_slop={kw['mass_ceiling_slop']}",
        f"- Ensemble: n_seeds={kw.get('n_seeds', 1)} "
        f"(seeds = {summary.seed}..{summary.seed + kw.get('n_seeds', 1) - 1})",
        f"- Backend: {kw.get('backend', 'evaluator')} "
        f"(evaluator_eval_cap per seed = {kw.get('evaluator_eval_cap', 'n/a')})",
        f"- Objectives: "
        + ", ".join(f"{o.target} ({o.direction})" for o in DEFAULT_OBJECTIVES),
        f"- Rovers attempted: {len(summary.results) + len(summary.failures)}",
        f"- Rovers succeeded: {len(summary.results)}",
        f"- Rovers failed: {len(summary.failures)}",
        "",
        "## Methodology",
        "",
        "Each rover is run independently under one of the four class-",
        "generic micro-rover scenarios (`polar_micro`, `mare_micro`,",
        "`highland_micro`, `crater_rim_micro`); see the per-YAML header",
        "comments and the `rover_rediscovery` module docstring for the",
                "two leakage controls (class-neutral operational duty cycle and mass-only budget).",
        "",
        "The bottom-up mass model's specific-mass coefficients in",
        "`MassModelParams` are cited from external space-hardware sources",
        "(SMAD, AIAA S-120A, vendor",
        "catalogues) - none are regressed against the registry - so this",
        "is a leakage-free rediscovery comparison rather than leave-one-",
        "out cross-validation: nothing is fit to the registry, so no",
        "per-rover hold-out or refit is needed.",
        "",
        "## Per-rover results",
        "",
    ]
    if df.empty:
        lines.append("_(no successful rediscovery results to report)_")
    else:
        report_cols = [
            "rover_name",
            "is_flown",
            "class_generic_scenario",
            "mass_modelled_kg",
            "mass_budget_kg",
            "pareto_front_size",
            "design_space_distance",
            "pareto_dominated",
            "abs_err_median_pct",
            "abs_err_max_pct",
            "abs_err_max_var",
        ]
        lines.append(_df_to_markdown_table(df[report_cols]))
    lines.append("")
    if summary.failures:
        lines.extend(
            [
                "## Failures",
                "",
                "These rovers raised during `rediscover()` and were skipped:",
                "",
            ]
        )
        for rover_name, msg in summary.failures.items():
            lines.append(f"- **{rover_name}**: {msg}")
        lines.append("")

    if not df.empty:
        lines.extend(
            [
                "## Aggregate statistics",
                "",
                f"- Median design-space distance: "
                f"`{df['design_space_distance'].median():.3f}`",
                f"- Median of per-rover median |err|: "
                f"`{df['abs_err_median_pct'].median():.1f} %`",
                f"- Pareto-dominated fraction: "
                f"`{df['pareto_dominated'].mean():.0%}` "
                f"({int(df['pareto_dominated'].sum())} of {len(df)})",
                f"- n_wheels exact-match fraction: "
                f"`{df['n_wheels_matches'].mean():.0%}`",
                "",
                "## Interpretation",
                "",
                "- **Design-space distance** is the primary signal: the",
                "  median normalised L2 between each real rover and the",
                "  nearest Pareto point under its class-generic scenario.",
                "  In an 8-D unit cube the mean L2 between two uniformly",
                "  random points is ~1.13 (the closed-form RMS sqrt(8/6) is",
                "  1.155), so distance / ~1.13",
                "  is the fraction of the random-pair baseline. The smaller",
                "  this fraction, the closer the optimiser's front lands to",
                "  the rover's published design; ~0.7-1.0 indicates the front",
                "  lands in the broader neighbourhood rather than at the",
                "  design vector. Distances are reported relative to this",
                "  baseline rather than against any fixed pass/fail cutoff.",
                "- **Pareto-dominated** is a *secondary* signal. Several",
                "  rovers flip to `True` under a class-neutral",
                "  operational-duty-cycle assumption: the optimiser's front",
                "  contains lighter designs that beat real rovers in every",
                "  modelled objective. This reflects design constraints the",
                "  conceptual model does not carry (radiation, deployability,",
                "  integration and redundancy margins) and does **not** by",
                "  itself indicate a problem with the optimiser.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_loo_artifacts(
    summary: RediscoveryRunSummary,
    out_dir: Path,
) -> dict[str, Path]:
    """Write the rediscovery sweep's artifacts to ``out_dir``.

    Files
    -----
    - ``summary.csv``: one-row-per-rover summary table.
    - ``<rover>.json``: per-rover full Pareto front + scoring detail
      (one file per rover, lowercase + ``-`` → ``_`` filename slug).
    - ``rediscovery_loo_report.md``: human-readable rollup with the
      methodology, per-rover table, failures, and acceptance gates.
    - ``failures.json``: ``{rover_name: error_message}`` (empty dict
      if all succeeded). Always written so downstream consumers can
      key off its presence rather than its content.

    Returns the ``{name: path}`` map of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    df = summarize_results(summary)
    summary_csv = out_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)
    written["summary"] = summary_csv

    failures_json = out_dir / "failures.json"
    failures_json.write_text(json.dumps(summary.failures, indent=2, sort_keys=True))
    written["failures"] = failures_json

    for result in summary.results:
        slug = result.rover_name.lower().replace("-", "_")
        payload = _result_to_json_payload(result)
        path = out_dir / f"{slug}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        written[slug] = path

    md = _markdown_for_summary(df, summary)
    md_path = out_dir / "rediscovery_loo_report.md"
    md_path.write_text(md)
    written["report"] = md_path

    return written


__all__ = [
    "DEFAULT_PER_ROVER_OVERRIDES",
    "RediscoveryRunSummary",
    "run_rediscovery_loo",
    "summarize_results",
    "write_loo_artifacts",
]
