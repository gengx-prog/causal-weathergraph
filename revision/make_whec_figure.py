"""Figure 8: pre-specified wind-humidity eddy convergence (WHEC) test, house style of make_paper_figures_v2.

Every plotted number is read from revision_outputs/whec_test/whec_band_endpoints.csv
(region-normalized frozen block gains, HAC64 standard errors).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.make_paper_figures_v2 import C, M, FIG, OUT, setup, style, caption, save  # noqa: E402

MANUSCRIPT_FIG = ROOT / "manuscript" / "revision_v2_20261002" / "figures"
SEGMENTS = [("eval_wb2", "2019–2023"), ("eval_cds", "2023–2025")]
BANDS = [("tropical", "Tropical\n0–30°"), ("midlatitude", "Midlatitude\n30–60°"), ("polar", "Polar\n60–90°")]
FAMILIES = [("A_local_whec", "Local WHEC"), ("B_remote_whec", "Remote WHEC"), ("C_placebo", "Placebo")]


def forest(ax, rows, ep):
    """rows: list of (label, family, target, statistic); two held-out segments per row."""
    for i, (label, fam, tv, stat) in enumerate(rows):
        for k, (seg, seg_label) in enumerate(SEGMENTS):
            r = ep[(ep.family == fam) & (ep.target_var == tv) & (ep.period == seg) & (ep.statistic == stat)].iloc[0]
            est, half = 100 * r.estimate, 196 * r.se_hac64
            ax.errorbar(est, i + (k - 0.5) * 0.28, xerr=half, color=C[[0, 5][k]], marker=M[k], ms=4.5, capsize=2, lw=1.0,
                        ls="none", label=seg_label if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)


def main():
    setup()
    ep = pd.read_csv(OUT / "whec_test" / "whec_band_endpoints.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), gridspec_kw={"wspace": 0.75})

    ax = axes[0]
    forest(ax, [(lab, "A_local_whec", "cloud_cover", b) for b, lab in BANDS], ep)
    ax.set_xlabel("MSE reduction (%)")
    ax.set_xlim(0, 0.8)
    ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.35)
    style(ax)
    caption(ax, "a", "Cloud, local WHEC", y=-0.36)

    ax = axes[1]
    forest(ax, [(lab, fam, "cloud_cover", "midlatitude") for fam, lab in FAMILIES], ep)
    ax.set_xlabel("MSE reduction (%)")
    style(ax)
    caption(ax, "b", "Cloud, midlatitude", y=-0.36)

    ax = axes[2]
    forest(ax, [(lab, "A_local_whec", "humidity", b) for b, lab in BANDS], ep)
    ax.set_xlabel("MSE reduction (%)")
    ax.set_xlim(0, 9)
    style(ax)
    caption(ax, "c", "Humidity, local WHEC", y=-0.36)

    save(fig, "fig08_whec")
    shutil.copyfile(FIG / "fig08_whec.pdf", MANUSCRIPT_FIG / "fig08_whec.pdf")
    print("saved", FIG / "fig08_whec.pdf")


if __name__ == "__main__":
    main()
