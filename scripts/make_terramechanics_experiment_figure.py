"""Render the terramechanics validation figure (Fig. 8).

Reproduces ``paper/figures/fig_terramechanics_experiment.png``: the analytical
Bekker-Wong physics layer evaluated against measured single-wheel drawbar pull
and sinkage from three independent sources (Ding 2011, Wang & Han 2016 KLS-1,
Hurrell 2025 Rashid-1), smooth and grousered. Data and BW predictions come from
``roverdevkit.validation.terramechanics_experiment`` (digitised measurements in
``data/validation/single_wheel_experiments.csv``).

Usage
-----
::

    python scripts/make_terramechanics_experiment_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from roverdevkit.tradespace.visualize import set_paper_rcparams
from roverdevkit.validation.terramechanics_experiment import (
    compare_to_experiment,
    summarise,
)

SOURCE_LABELS: dict[str, str] = {
    "ding2011": "Ding et al. 2011",
    "wang_han_2016_kls1": "Wang & Han 2016 (KLS-1)",
    "hurrell2025_rashid1": "Hurrell et al. 2025 (Rashid-1)",
}

# (axis label, measured column, BW column, unit scale).
QUANTITIES = [
    ("drawbar pull (N)", "meas_drawbar_pull_n", "bw_drawbar_pull_n", 1.0),
    ("sinkage (mm)", "meas_sinkage_m", "bw_sinkage_m", 1000.0),
]
# (grouser height selector, marker, name, colour); None -> the grousered family.
FAMILIES = [(0.0, "o", "smooth", "#2166ac"), (None, "s", "grousered", "#b2182b")]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("paper/figures/fig_terramechanics_experiment.png"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    terra = compare_to_experiment()
    summary = summarise(terra)
    print(
        f"operating points: {summary['n_operating_points']} | "
        f"digitised: {summary['n_digitised']} | "
        f"pending: {summary['n_pending_digitisation']}"
    )
    if summary["n_digitised"]:
        print(
            f"BW median |%err|  DP: {summary['bw_dp_median_abs_pct_err']:.1f}%  "
            f"sinkage: {summary['bw_sinkage_median_abs_pct_err']:.1f}%"
        )

    # Plot sources that have measured data; fall back to all (model-only preview).
    with_data = [
        s for s in SOURCE_LABELS
        if terra[(terra["source"] == s) & terra["meas_drawbar_pull_n"].notna()].shape[0]
    ]
    plot_sources = with_data or [s for s in SOURCE_LABELS if (terra["source"] == s).any()]

    set_paper_rcparams()
    import matplotlib.pyplot as plt

    n_src = len(plot_sources)
    fig, axes = plt.subplots(
        n_src, len(QUANTITIES),
        figsize=(7.2, 3.1 * n_src), squeeze=False,
    )
    for row, source in enumerate(plot_sources):
        sub = terra[terra["source"] == source]
        hg_lug = sub.loc[sub["grouser_height_m"] > 0, "grouser_height_m"].max()
        for col, (ylab, meascol, bwcol, scale) in enumerate(QUANTITIES):
            ax = axes[row][col]
            for hg, marker, _name, color in FAMILIES:
                target_hg = hg_lug if hg is None else hg
                if target_hg != target_hg:  # NaN -> no grousered family
                    continue
                fam = sub[sub["grouser_height_m"] == target_hg].sort_values("slip")
                if fam.empty:
                    continue
                fam_label = (
                    "Smooth" if target_hg == 0
                    else f"Grousered ({target_hg * 1000:.0f} mm)"
                )
                ax.plot(
                    fam["slip"], fam[bwcol] * scale, "-", color=color, lw=1.6,
                    label=f"BW, {fam_label}",
                )
                meas = fam[fam[meascol].notna()]
                if not meas.empty:
                    ax.plot(
                        meas["slip"], meas[meascol] * scale, marker, color=color,
                        ms=8, mfc="none", mew=1.6, label=f"Measured, {fam_label}",
                    )
            if col == 0:
                ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
            ax.set_title(SOURCE_LABELS[source])
            ax.set_xlabel("slip ratio")
            ax.set_ylabel(ylab)
            ax.legend(loc="best")

    if summary["n_digitised"] == 0:
        fig.suptitle(
            "Measurements pending digitisation \u2014 BW curves shown",
            fontsize=9, y=1.01,
        )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
