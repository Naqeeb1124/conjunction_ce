"""Figure of the scale-up (many conjunctions at once), drawn from results/scaleup.json.

The style (palette, sizes, fonts, title layout) is imported from ``conjce.figures``.  Every number
in the figure text is read from the results file.

    .venv\\Scripts\\python -m conjce.batch_figures
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from . import figures as F

LINES = (("a", "full joint", "$9^K$", F.BLUE),
         ("b", "exact-one joint", "$4^K$", F.ORANGE),
         ("c", "per conjunction", "$9K$", F.AQUA))


def _series(res, name):
    rows = [r for r in res["timing"] if r["formulation"] == name]
    return np.array([r["K"] for r in rows]), np.array([r["wall_clock_seconds"] for r in rows])


def make(res, path):
    F._setup()
    k_max = max(res["settings"]["k_grid"])
    cap = res["settings"]["size_cap"]
    bm = res["burden"][f"master_K{k_max}"]
    # sizes, not wall-clock times, in the headline: times belong to one machine
    n_var_c = [r["variables"] for r in res["timing"] if r["formulation"] == "c" and r["K"] == k_max][0]
    title = (f"Split by conjunction, {k_max} conjunctions need {n_var_c:,} variables, not $9^{{{k_max}}}$; "
             f"balancing the free\nchoices cuts the busiest satellite from {bm['max_burns_fixed_tie_break']} "
             f"to {bm['max_burns_balanced_deterministic']} burns")
    subtitle = (f"Batch of {k_max} conjunctions among {res['settings']['satellites']} satellites of "
                f"{res['settings']['operators']} operators over {res['settings']['window_hours']:.0f} hours; "
                f"{bm['n_free']} of them are free (either side may burn at equal cost).")

    fig = plt.figure(figsize=F.FIGSIZE, dpi=F.DPI)
    fig.text(0.035, 0.965, title, fontsize=F.FS_TITLE - 3, fontweight="semibold", color=F.INK,
             ha="left", va="top", linespacing=1.2)
    fig.text(0.035, 0.84, subtitle, fontsize=F.FS_SUB - 1, color=F.INK2, ha="left", va="top")

    # left: wall-clock against K
    ax = fig.add_axes([0.075, 0.12, 0.43, 0.56])
    for name, words, formula, color in LINES:
        k, t = _series(res, name)
        ax.plot(k, t, color=color, lw=F.LW, solid_capstyle="round", zorder=3)
        ax.plot(k, t, linestyle="none", marker="o", markersize=7, color=color, markeredgecolor=F.SURFACE,
                markeredgewidth=1.5, zorder=4)
        if name == "c":
            ax.text(k[-1], t[-1] * 1.9, f"({name}) {words}: {formula} variables", ha="right", va="bottom",
                    fontsize=F.FS_NOTE, color=F.INK)
        else:
            # the last K within the size cap: a ring marks where the formulation stops
            ax.plot([k[-1]], [t[-1]], linestyle="none", marker="o", markersize=15, markerfacecolor="none",
                    markeredgecolor=F.INK2, markeredgewidth=1.6, zorder=5)
            # (b) to the right of its ring; (a) in the empty area above and left of its ring, its leader
            # attached to the label's left end so that it crosses neither line (b) nor label (b)
            offset = (-30, 62) if name == "a" else (24, 10)
            ax.annotate(f"({name}) {words}: {formula} variables, stops at K = {k[-1]}", (k[-1], t[-1]),
                        xytext=offset, textcoords="offset points", ha="left", va="center",
                        fontsize=F.FS_NOTE, color=F.INK,
                        arrowprops=dict(arrowstyle="-", color=F.MUTED, lw=1.0, shrinkA=2, shrinkB=9,
                                        relpos=(0.0, 0.5)))
    ax.set_yscale("log")
    ax.set_xlim(0, k_max + 1)
    ax.set_ylim(3e-4, 5.0)
    ax.set_xlabel("K: conjunctions in the batch")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    F._unit_label(ax, f"wall-clock seconds, build plus solve, smallest of {res['settings']['timing_repeats']} "
                      f"(one machine)")
    ax.set_title(f"Solve time; rings: last K under the cap of {cap:,} variables", loc="left", pad=30,
                 fontsize=F.FS_PANEL - 1)

    # right: satellites by number of burns
    ax = fig.add_axes([0.62, 0.12, 0.35, 0.56])
    h_fix = np.array(bm["histogram_fixed_tie_break"])
    h_det = np.array(bm["histogram_balanced_deterministic"])
    n = max(h_fix.size, h_det.size)
    h_fix = np.pad(h_fix, (0, n - h_fix.size))
    h_det = np.pad(h_det, (0, n - h_det.size))
    x = np.arange(n)
    w = 0.38
    for off, h, color, label in ((-w / 2 - 0.02, h_fix, F.MUTED, "fixed tie-break (the project's fixed rule)"),
                                 (w / 2 + 0.02, h_det, F.INK, "balanced, deterministic")):
        # greys and ink only: the right panel uses no hue of the left panel
        ax.bar(x + off, h, width=w, color=color, label=label, zorder=3, linewidth=0)
        for xi, v in zip(x + off, h):
            if v > 0:
                ax.text(xi, v + 0.3, f"{v}", ha="center", va="bottom", fontsize=F.FS_NOTE - 1, color=F.INK)
    ax.set_xticks(x)
    ax.set_xlabel("burns of one satellite")
    ax.set_ylim(0, max(h_fix.max(), h_det.max()) * 1.3)
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.legend(loc="upper right", handlelength=1.2, fontsize=F.FS_NOTE - 1)
    F._unit_label(ax, "number of satellites")
    ax.set_title(f"Burns per satellite, all {k_max} conjunctions", loc="left", pad=30, fontsize=F.FS_PANEL - 1)
    F._save(fig, path)


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent.parent
    make(json.loads((root / "results" / "scaleup.json").read_text(encoding="utf-8")), root / "figures" / "scaleup.png")
    print("figure written")
