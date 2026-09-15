"""Render the §5.4 rediscovery distance bar chart (Fig. 4).

Reproduces ``paper/figures/fig_rediscovery_distance.png`` from the two
committed CSV artifacts so the paper figure is regenerable rather than a
hand-made PNG:

- ``reports/rediscovery_loo_evaluator/summary.csv`` — per-rover nearest-
  Pareto design-space distance (the bars);
- ``reports/rediscovery_baseline/feasible_baseline.csv`` — the
  feasible-design null and each rover's distance to the feasible-region
  centroid (the overlays added for the §5.4 pre-submission item).

The chart shows, per rover:

- a horizontal bar = rediscovery distance (blue in-scope < 50 kg, grey
  out-of-scope Yutu-2, kept as a reference point);
- a marker = the rover's distance to the feasible-region centroid (the
  "typical feasible design");
- two vertical reference lines = the unit-cube random-pair null
  ($\approx$1.20) and the feasibility-restricted null ($\approx$1.17).

Usage
-----
::

    python scripts/make_rediscovery_distance_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import pandas as pd

from roverdevkit.tradespace.visualize import set_paper_rcparams

# Scope ceiling: the tool and the rediscovery check target sub-50 kg
# micro-rovers. Yutu-2 (~96 kg modelled / ~135 kg published) sits above
# it and is drawn greyed as an out-of-scope reference point.
_SCOPE_CEILING_KG: float = 50.0
_IN_SCOPE_COLOR = "#1f77b4"
_OUT_SCOPE_COLOR = "#b0b0b0"
_CENTROID_COLOR = "#d62728"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rediscovery-summary",
        type=Path,
        default=Path("reports/rediscovery_loo_evaluator/summary.csv"),
    )
    p.add_argument(
        "--feasible-baseline",
        type=Path,
        default=Path("reports/rediscovery_baseline/feasible_baseline.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("paper/figures/fig_rediscovery_distance.png"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    redis = pd.read_csv(args.rediscovery_summary)
    feas = pd.read_csv(args.feasible_baseline)
    df = redis.merge(
        feas[
            [
                "rover_name",
                "rover_to_centroid_distance",
                "feasible_random_pair_mean",
                "unit_cube_random_pair",
            ]
        ],
        on="rover_name",
        how="left",
    )

    df["out_of_scope"] = df["mass_modelled_kg"] > _SCOPE_CEILING_KG
    # Smallest distance at the bottom, largest (Yutu-2) at the top.
    df = df.sort_values("design_space_distance", ascending=True).reset_index(drop=True)

    in_scope = df[~df["out_of_scope"]]
    in_scope_median = float(median(in_scope["design_space_distance"]))
    unit_cube_null = float(df["unit_cube_random_pair"].iloc[0])
    feasible_null = float(df["feasible_random_pair_mean"].median())

    set_paper_rcparams()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = range(len(df))
    colors = [
        _OUT_SCOPE_COLOR if oos else _IN_SCOPE_COLOR for oos in df["out_of_scope"]
    ]
    ax.barh(list(y), df["design_space_distance"], color=colors, zorder=2)

    # Feasible-region centroid markers (the "typical feasible design").
    ax.scatter(
        df["rover_to_centroid_distance"],
        list(y),
        marker="D",
        s=34,
        facecolor="none",
        edgecolor=_CENTROID_COLOR,
        linewidths=1.4,
        zorder=4,
    )

    # Reference nulls.
    ax.axvline(unit_cube_null, ls="--", color="black", lw=1.2, zorder=3)
    ax.axvline(feasible_null, ls=":", color="#555555", lw=1.4, zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["rover_name"])
    ax.set_xlabel("design-space distance (normalised L2, 8-D)")
    ax.set_title(f"Rediscovery distance per rover (in-scope median = {in_scope_median:.2f})")
    ax.set_xlim(0, max(unit_cube_null, float(df["rover_to_centroid_distance"].max())) + 0.22)

    legend_handles = [
        Patch(facecolor=_IN_SCOPE_COLOR, label="rediscovery distance, in scope (< 50 kg)"),
        Patch(facecolor=_OUT_SCOPE_COLOR, label="rediscovery distance, out of scope (> 50 kg)"),
        Line2D(
            [0], [0], marker="D", linestyle="none", markerfacecolor="none",
            markeredgecolor=_CENTROID_COLOR, markersize=7,
            label="feasible-region centroid",
        ),
        Line2D([0], [0], color="black", ls="--", lw=1.2,
               label=f"unit-cube null ({unit_cube_null:.2f})"),
        Line2D([0], [0], color="#555555", ls=":", lw=1.4,
               label=f"feasible-design null ({feasible_null:.2f})"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Wrote {args.out}")
    print(
        f"  in-scope median={in_scope_median:.3f}, "
        f"unit-cube null={unit_cube_null:.3f}, feasible null={feasible_null:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
