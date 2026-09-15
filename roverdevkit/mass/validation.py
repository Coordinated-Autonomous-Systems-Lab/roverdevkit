"""Cross-check the bottom-up mass model against published rover total masses.

The validation set lives in ``data/mass_validation_set.csv``. Each row is a
best-effort full design vector for a published rover, with an
``imputation_notes`` column documenting every field that was not directly
published and how it was estimated.

The ``in_class`` flag (True/False) marks whether the rover is inside the
bottom-up mass model's specific-mass calibration regime (5-50 kg
lunar micro-rovers). At sub-5-kg total mass the bottom-up
model's fixed-cost terms (per-wheel motor base mass, avionics base
mass, harness / thermal / margin fractions) come to dominate, and the
model systematically over-predicts total mass relative to ultra-micro
hardware which uses mass-optimised custom motors and avionics that
the SMAD/AIAA/vendor-catalogue specific-mass constants do not reflect.
Updating the constants for the ultra-micro regime would invalidate
the model's calibration on the 5-50 kg class, so we keep the
calibration unchanged and explicitly mark sub-5-kg rovers as
``in_class=False``.

The primary validation statistic is **median absolute percent error on
in-class rovers**; the target is <= 30 % (plan §8). Out-of-regime
rovers (Yutu-2 at 135 kg, MARSOKHOD at 70 kg) are reported alongside
but excluded from the primary statistic.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from roverdevkit.architecture import architecture_for_wheel_count
from roverdevkit.drivetrain.motor import sizing_peak_torque_anchor_nm
from roverdevkit.mass.parametric_mers import (
    MassBreakdown,
    MassModelParams,
    estimate_mass,
)

DEFAULT_VALIDATION_CSV: Path = (
    Path(__file__).resolve().parents[2] / "data" / "mass_validation_set.csv"
)


@dataclass(frozen=True)
class RoverValidationRow:
    """One row of the validation set: a published rover plus imputations.

    ``in_class`` marks whether the rover sits inside the bottom-up
    mass model's specific-mass calibration regime (5-50 kg lunar
    micro-rovers). See the
    module docstring for why the two diverged on 2026-05-27.
    """

    rover_name: str
    mass_total_kg: float
    wheel_radius_m: float
    wheel_width_m: float
    n_wheels: int
    chassis_mass_kg: float
    solar_area_m2: float
    battery_capacity_wh: float
    avionics_power_w: float
    grouser_height_m: float
    grouser_count: int
    payload_mass_kg: float
    """Scientific-payload mass, kg.

    Separated out of the back-solved ``chassis_mass_kg`` bucket so the
    bottom-up model sizes only the *bus* and adds payload as a flat,
    ungrown line item — matching how payload enters the live evaluator.
    See ``data/mass_validation_set.csv`` ``citation`` and
    ``imputation_notes`` for the per-rover literature source."""
    in_class: bool
    citation: str
    imputation_notes: str


@dataclass(frozen=True)
class RoverValidationResult:
    """Outcome of running the bottom-up mass model on one rover.

    ``in_class`` mirrors :class:`RoverValidationRow.in_class`: True iff
    the rover sits inside the mass-model calibration regime.
    """

    rover_name: str
    in_class: bool
    mass_published_kg: float
    mass_predicted_kg: float
    breakdown: MassBreakdown

    @property
    def absolute_error_kg(self) -> float:
        return self.mass_predicted_kg - self.mass_published_kg

    @property
    def percent_error(self) -> float:
        return 100.0 * self.absolute_error_kg / self.mass_published_kg


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregate statistics over a batch of validation rows."""

    n_total: int
    n_in_class: int
    median_abs_percent_error_in_class: float
    mean_abs_percent_error_in_class: float
    worst_in_class: RoverValidationResult
    per_rover: tuple[RoverValidationResult, ...]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"unparseable boolean: {value!r}")


def load_validation_set(csv_path: Path | str | None = None) -> list[RoverValidationRow]:
    """Read ``data/mass_validation_set.csv`` into a list of dataclasses."""
    path = Path(csv_path) if csv_path else DEFAULT_VALIDATION_CSV
    rows: list[RoverValidationRow] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                RoverValidationRow(
                    rover_name=row["rover_name"],
                    mass_total_kg=float(row["mass_total_kg"]),
                    wheel_radius_m=float(row["wheel_radius_m"]),
                    wheel_width_m=float(row["wheel_width_m"]),
                    n_wheels=int(row["n_wheels"]),
                    chassis_mass_kg=float(row["chassis_mass_kg"]),
                    solar_area_m2=float(row["solar_area_m2"]),
                    battery_capacity_wh=float(row["battery_capacity_wh"]),
                    avionics_power_w=float(row["avionics_power_w"]),
                    grouser_height_m=float(row["grouser_height_m"]),
                    grouser_count=int(row["grouser_count"]),
                    payload_mass_kg=float(row.get("payload_mass_kg", 0.0) or 0.0),
                    in_class=_parse_bool(row["in_class"]),
                    citation=row.get("citation", ""),
                    imputation_notes=row["imputation_notes"],
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Running the comparison
# ---------------------------------------------------------------------------


def predict_row(
    row: RoverValidationRow,
    params: MassModelParams | None = None,
) -> RoverValidationResult:
    """Run ``estimate_mass`` on a single validation row.
    """
    peak_wheel_torque_nm = sizing_peak_torque_anchor_nm(
        total_mass_kg=row.mass_total_kg,
        wheel_radius_m=row.wheel_radius_m,
        n_wheels=row.n_wheels,
    )
    breakdown = estimate_mass(
        wheel_radius_m=row.wheel_radius_m,
        wheel_width_m=row.wheel_width_m,
        n_wheels=row.n_wheels,
        chassis_mass_kg=row.chassis_mass_kg,
        solar_area_m2=row.solar_area_m2,
        battery_capacity_wh=row.battery_capacity_wh,
        avionics_power_w=row.avionics_power_w,
        peak_wheel_torque_nm=peak_wheel_torque_nm,
        grouser_height_m=row.grouser_height_m,
        grouser_count=row.grouser_count,
        payload_mass_kg=row.payload_mass_kg,
        mobility_architecture=architecture_for_wheel_count(row.n_wheels),
        params=params,
    )
    return RoverValidationResult(
        rover_name=row.rover_name,
        in_class=row.in_class,
        mass_published_kg=row.mass_total_kg,
        mass_predicted_kg=breakdown.total_kg,
        breakdown=breakdown,
    )


def validate_against_published_rovers(
    csv_path: Path | str | None = None,
    params: MassModelParams | None = None,
) -> ValidationSummary:
    """Run the bottom-up mass model on the full validation set and summarise.

    The primary statistic returned is the median absolute percent error on
    in-class (5-50 kg) rovers. Out-of-class rovers (nano, medium, large)
    are included in ``per_rover`` but excluded from the in-class
    statistics, reflecting the 5-50 kg calibration range of the specific
    mass constants in :class:`MassModelParams`.
    """
    rows = load_validation_set(csv_path)
    results = tuple(predict_row(r, params=params) for r in rows)

    in_class_results = [r for r in results if r.in_class]
    if not in_class_results:
        raise ValueError("Validation set contains no in-class rovers.")

    in_class_abs_errors = [abs(r.percent_error) for r in in_class_results]
    worst = max(in_class_results, key=lambda r: abs(r.percent_error))

    return ValidationSummary(
        n_total=len(results),
        n_in_class=len(in_class_results),
        median_abs_percent_error_in_class=float(median(in_class_abs_errors)),
        mean_abs_percent_error_in_class=float(mean(in_class_abs_errors)),
        worst_in_class=worst,
        per_rover=results,
    )


def format_report(summary: ValidationSummary) -> str:
    """Human-readable table for notebooks and reports."""
    lines = [
        "Rover                 in_class  published (kg)  predicted (kg)     err %",
        "-" * 73,
    ]
    for r in summary.per_rover:
        flag = "yes" if r.in_class else "no "
        lines.append(
            f"{r.rover_name:20s}  {flag:>8s}  {r.mass_published_kg:14.2f}  "
            f"{r.mass_predicted_kg:14.2f}  {r.percent_error:+7.1f}"
        )
    lines.append("-" * 73)
    lines.append(
        f"Aggregates on in-class rovers (n={summary.n_in_class}): "
        f"median |err| = {summary.median_abs_percent_error_in_class:.1f} %, "
        f"mean |err| = {summary.mean_abs_percent_error_in_class:.1f} %, "
        f"worst = {summary.worst_in_class.rover_name} "
        f"({summary.worst_in_class.percent_error:+.1f} %)."
    )
    return "\n".join(lines)
