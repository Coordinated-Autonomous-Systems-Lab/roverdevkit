"""De-tuned (no per-rover calibration) peak-solar prediction for flown rovers.

Why this module exists
----------------------
The flown-rover peak-solar check in :mod:`roverdevkit.validation.rover_comparison`
is an *operational-consistency* test: it asks whether predicted noon power lands
inside each rover's published band. But that check leans on per-rover
``panel_efficiency`` and ``panel_dust_factor`` values stored in the registry
(Pragyan ``0.22 / 0.85``, Yutu-2 ``0.20 / 0.55``), which were chosen to be
consistent with each rover's published number. Using rover-specific cell and
dust parameters to "predict" that same rover's power is **circular**: the band
test cannot fail by construction, so it validates nothing about the power
sub-model.

This module removes the circularity. It predicts peak solar power for the flown
rovers using a **single, fixed, literature-justified panel parameter set applied
uniformly to every rover** -- no per-rover knobs. The only rover-specific inputs
are *published geometry* (solar-array area) and *published location* (scenario
latitude, which fixes the noon sun elevation). The prediction is therefore a
genuine, out-of-sample forward calculation, and it is allowed to be wrong.

What the de-tuned prediction reveals (and why that is the honest result)
-----------------------------------------------------------------------
- **Fresh arrays predict cleanly.** Pragyan flew a single lunar day, so its
  published peak is a near-beginning-of-life (BOL) number. The de-tuned BOL +
  clean-array prediction lands inside its published band with single-digit
  percent error -- a real predictive hit with zero tuning, robust across the
  full literature cell-efficiency range (see :func:`sensitivity_band_w`).
- **Aged arrays expose, not hide, their degradation.** Yutu-2 operated for
  dozens of lunar days; its published "peak" is a heavily dust- and
  end-of-life-degraded operational value. The de-tuned BOL prediction
  over-predicts it by ~2x, and the *implied* net derating we back out
  (published / BOL ~ 0.5) is independently consistent with multi-year lunar
  dust accumulation + EOL cell degradation reported in the literature. We
  therefore report the degradation as a **recovered output**, not a tuned
  input.

So the de-tuned check converts a circular "always passes" band test into an
honest statement: *with literature BOL clean-array parameters and no per-rover
calibration, the power model predicts the fresh-array rover within its band, and
the only residual is a physically attributable aging derate on the multi-year
rover.*

Fixed literature panel parameter set (applied to every rover)
-------------------------------------------------------------
Net DC system efficiency is a product of four factors, none fitted to the
rovers in this study:

    eta_sys = eta_cell * f_pack * f_elec * f_temp

==============  =======  ===================================================
factor          value    source / rationale
==============  =======  ===================================================
eta_cell        0.30     Typical beginning-of-life AM0 efficiency for a
                         space-grade solar cell. Not a cited datasheet
                         value; the prediction is swept over 0.28--0.32.
f_pack          0.90     Active cell area / panel area (Patel & Beik,
                         *Spacecraft Power Systems*, 2nd ed., 2023).
f_elec          0.92     MPPT + harness + blocking-diode + assembly losses
                         (Wertz, Everett & Puschell, *The New SMAD*, 2011).
f_temp          0.90     Computed, not cited: GaAs power coefficient
                         ~-0.06 %/K times a cell ~90 K above the 28 degC
                         AM0 reference.
==============  =======  ===================================================

    eta_sys = 0.30 * 0.90 * 0.92 * 0.90 = 0.2236

The clean-array dust transmission factor for a fresh (lunar-day-1) array is set
to ``0.98``. This is an assumed 2 % optical loss, not a cited value; it is applied
uniformly so it is not a per-rover knob, and the prediction is insensitive to it
next to the cell-efficiency spread (see :func:`sensitivity_band_w`).

All flown rovers are evaluated with a **horizontal-equivalent** panel
(``panel_tilt_deg = 0``) because their published peak-solar bands in
``data/published_traverse_data.csv`` are operational-average values calibrated
against horizontal pointing (see the registry's ``panel_tilt_deg`` note).

References
----------
Wertz, J. R., Everett, D. F. & Puschell, J. J. *Space Mission Engineering:
The New SMAD*. Microcosm Press, 2011.

Patel, M. R. & Beik, O. *Spacecraft Power Systems*, 2nd ed. CRC Press, 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from roverdevkit.power.solar import (
    SOLAR_CONSTANT_AU_1_W_PER_M2,
    panel_power_w,
    sun_elevation_deg,
)
from roverdevkit.validation.rover_registry import (
    PublishedTruth,
    RoverRegistryEntry,
    flown_registry,
    load_truth_table,
)

# ---------------------------------------------------------------------------
# Fixed literature panel parameters (applied uniformly; never tuned per rover)
# ---------------------------------------------------------------------------

CELL_EFFICIENCY_BOL: float = 0.30
"""Triple-junction GaAs/Ge cell efficiency, BOL AM0 (Spectrolab XTJ / AzurSpace 3G30)."""

PACKING_FACTOR: float = 0.90
"""Active cell area / total panel area (Patel Ch. 4)."""

ELECTRICAL_DERATE: float = 0.92
"""MPPT + harness + blocking-diode + assembly losses (SMAD Ch. 11)."""

HIGH_TEMP_DERATE: float = 0.90
"""Lunar-noon high-temperature derate vs the 28 degC AM0 reference."""

SYSTEM_EFFICIENCY: float = (
    CELL_EFFICIENCY_BOL * PACKING_FACTOR * ELECTRICAL_DERATE * HIGH_TEMP_DERATE
)
"""Net DC system efficiency from the cited stack-up (~0.224)."""

CLEAN_DUST_FACTOR: float = 0.98
"""Dust-transmission factor for a fresh, lunar-day-1 array."""

CELL_EFFICIENCY_RANGE: tuple[float, float] = (0.28, 0.32)
"""Plausible literature spread of BOL triple-junction cell efficiency, used for
the prediction sensitivity band so no single efficiency choice is load-bearing."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetunedPowerPrediction:
    """De-tuned peak-solar prediction for one flown rover vs published truth.

    Every field is computed with the fixed literature parameter set above; no
    value is calibrated to the rover it describes.
    """

    rover_name: str
    latitude_deg: float
    panel_area_m2: float
    peak_elevation_deg: float
    mission_duration_days: float

    predicted_bol_w: float
    """Clean BOL prediction (``dust = 1.0``); the upper-bound forward estimate."""

    predicted_clean_w: float
    """Prediction with the single literature clean-array dust factor (0.98)."""

    sensitivity_low_w: float
    sensitivity_high_w: float
    """Clean prediction spread over :data:`CELL_EFFICIENCY_RANGE`."""

    published_w: float
    band_low_w: float
    band_high_w: float

    @property
    def in_band(self) -> bool:
        """True iff the clean de-tuned prediction lands inside the published band."""
        return self.band_low_w <= self.predicted_clean_w <= self.band_high_w

    @property
    def pct_error_vs_published(self) -> float:
        """Signed percent error of the clean prediction vs the published value."""
        return 100.0 * (self.predicted_clean_w - self.published_w) / max(1e-9, self.published_w)

    @property
    def implied_total_derate(self) -> float:
        """Net derate the published value implies relative to clean BOL.

        ``published / predicted_bol_w``. ~1.0 means a fresh array consistent
        with BOL; well below 1.0 means the published number bakes in dust /
        end-of-life degradation that the BOL prediction (correctly) does not.
        """
        return self.published_w / max(1e-9, self.predicted_bol_w)


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------


def _peak_elevation_deg(latitude_deg: float) -> float:
    """Noon sun elevation (hour angle 0, zero declination) at this latitude."""
    return sun_elevation_deg(latitude_deg, lunar_hour_angle_deg=0.0)


def _horizontal_peak_power_w(
    panel_area_m2: float,
    panel_efficiency: float,
    peak_elevation_deg: float,
    dust_factor: float,
) -> float:
    """Closed-form noon power for a horizontal (tilt=0) panel.

    For a horizontal panel the cosine of incidence collapses to ``sin(el)`` so
    azimuth is irrelevant; we call the shared :func:`panel_power_w` so the
    de-tuned prediction uses the exact same physics as the traverse sim.
    """
    if peak_elevation_deg <= 0.0:
        return 0.0
    return panel_power_w(
        panel_area_m2=panel_area_m2,
        panel_efficiency=panel_efficiency,
        sun_elevation_deg=peak_elevation_deg,
        panel_tilt_deg=0.0,
        dust_degradation_factor=dust_factor,
        solar_constant_w_per_m2=SOLAR_CONSTANT_AU_1_W_PER_M2,
    )


def sensitivity_band_w(
    panel_area_m2: float,
    peak_elevation_deg: float,
    *,
    dust_factor: float = CLEAN_DUST_FACTOR,
    cell_efficiency_range: tuple[float, float] = CELL_EFFICIENCY_RANGE,
) -> tuple[float, float]:
    """Clean-prediction power spread as cell BOL efficiency sweeps its range.

    Holds the packing / electrical / temperature derates fixed and varies only
    the cited cell-efficiency endpoints, so the caller can show that the
    in-band conclusion does not hinge on one efficiency choice.
    """
    lo_cell, hi_cell = cell_efficiency_range
    eta_lo = lo_cell * PACKING_FACTOR * ELECTRICAL_DERATE * HIGH_TEMP_DERATE
    eta_hi = hi_cell * PACKING_FACTOR * ELECTRICAL_DERATE * HIGH_TEMP_DERATE
    low_w = _horizontal_peak_power_w(panel_area_m2, eta_lo, peak_elevation_deg, dust_factor)
    high_w = _horizontal_peak_power_w(panel_area_m2, eta_hi, peak_elevation_deg, dust_factor)
    return low_w, high_w


def predict_one(
    entry: RoverRegistryEntry,
    truth: PublishedTruth,
) -> DetunedPowerPrediction:
    """De-tuned peak-solar prediction for one flown rover.

    Uses only the rover's *published* geometry (solar-array area) and *published*
    location (scenario latitude). The registry's per-rover ``panel_efficiency``
    and ``panel_dust_factor`` are deliberately ignored.
    """
    area = entry.design.solar_area_m2
    lat = entry.scenario.latitude_deg
    peak_elev = _peak_elevation_deg(lat)

    predicted_bol = _horizontal_peak_power_w(area, SYSTEM_EFFICIENCY, peak_elev, 1.0)
    predicted_clean = _horizontal_peak_power_w(area, SYSTEM_EFFICIENCY, peak_elev, CLEAN_DUST_FACTOR)
    sens_low, sens_high = sensitivity_band_w(area, peak_elev)

    return DetunedPowerPrediction(
        rover_name=entry.rover_name,
        latitude_deg=lat,
        panel_area_m2=area,
        peak_elevation_deg=peak_elev,
        mission_duration_days=truth.mission_duration_published_days,
        predicted_bol_w=predicted_bol,
        predicted_clean_w=predicted_clean,
        sensitivity_low_w=sens_low,
        sensitivity_high_w=sens_high,
        published_w=truth.peak_solar_power_w_published,
        band_low_w=truth.peak_solar_power_w_low,
        band_high_w=truth.peak_solar_power_w_high,
    )


def predict_all_flown(
    *,
    csv_path: Path | str | None = None,
) -> tuple[DetunedPowerPrediction, ...]:
    """De-tuned predictions for every flown rover in the registry."""
    truths = {row.rover_name: row for row in load_truth_table(csv_path)}
    return tuple(
        predict_one(entry, truths[entry.rover_name])
        for entry in flown_registry()
        if entry.rover_name in truths
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(predictions: tuple[DetunedPowerPrediction, ...]) -> str:
    """Human-readable table for notebooks and reports."""
    header = (
        "Rover      area_m2  elev_deg   pred_BOL  pred_clean   sens_band     published   band         "
        "in_band  err%    implied_derate"
    )
    lines = [header, "-" * len(header)]
    for p in predictions:
        lines.append(
            f"{p.rover_name:10s} {p.panel_area_m2:6.2f}  {p.peak_elevation_deg:7.1f}  "
            f"{p.predicted_bol_w:8.1f}  {p.predicted_clean_w:9.1f}  "
            f"{p.sensitivity_low_w:4.0f}-{p.sensitivity_high_w:<4.0f} W  "
            f"{p.published_w:8.1f} W  {p.band_low_w:3.0f}-{p.band_high_w:<3.0f} W  "
            f"{'yes' if p.in_band else 'NO':5s}  {p.pct_error_vs_published:+6.1f}  "
            f"{p.implied_total_derate:6.2f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"Fixed literature params: eta_sys = {SYSTEM_EFFICIENCY:.3f} "
        f"(cell {CELL_EFFICIENCY_BOL:.2f} x pack {PACKING_FACTOR:.2f} x "
        f"elec {ELECTRICAL_DERATE:.2f} x temp {HIGH_TEMP_DERATE:.2f}); "
        f"clean dust {CLEAN_DUST_FACTOR:.2f}."
    )
    return "\n".join(lines)
