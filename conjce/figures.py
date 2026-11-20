"""Slide-ready figures (16:9, white background, large type) built from results/results.json.

One y-scale per panel (no dual axes), small multiples for measures with different units, a fixed
colour-blind-safe palette, text in ink colours with identity carried by the mark, direct labels
next to the lines.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402

# palette: BLUE, ORANGE, AQUA are categorical; RAMP2, RAMP3 are ordinal blue ramps
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP3 = ("#86b6ef", "#2a78d6", "#104281")           # light -> dark
RAMP2 = ("#86b6ef", "#104281")
ELLIPSE_FILL = ("#cde2fb", "#9ec5f4", "#6da7ec")     # 3-sigma, 2-sigma, 1-sigma

FIGSIZE = (13.333, 7.5)
DPI = 200
FS_TITLE, FS_SUB, FS_PANEL, FS_LABEL, FS_TICK, FS_NOTE = 25, 17, 18, 16, 15, 14
LW = 2.8


def _setup():
    family = "DejaVu Sans"
    for cand in ("Segoe UI", "Helvetica Neue", "Arial"):
        try:
            font_manager.findfont(cand, fallback_to_default=False)
            family = cand
            break
        except Exception:
            continue
    plt.rcParams.update({
        "font.family": family, "font.size": FS_TICK, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_PANEL, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": FS_NOTE,
        "mathtext.default": "regular",
    })


def _titles(fig, title, subtitle):
    fig.text(0.035, 0.955, title, fontsize=FS_TITLE, fontweight="semibold", color=INK, ha="left", va="top")
    fig.text(0.035, 0.895, subtitle, fontsize=FS_SUB, color=INK2, ha="left", va="top")


def _sci(v: float, digits: int = 1) -> str:
    """1.5e-3 -> mathtext '1.5×10^-3'."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10.0**exp
    if round(mant, digits) >= 10:
        mant, exp = mant / 10.0, exp + 1
    return rf"${mant:.{digits}f}\times10^{{{exp}}}$"


def _barh(ax, y, value, height, color):
    """Horizontal bar from zero with a rounded data end and a square baseline end."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    box = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    sx = (x1 - x0) / box.width                # data units per inch
    sy = abs(y1 - y0) / box.height
    r = min(0.065 * sx, 0.5 * value) if value > 0 else 0.0
    pad = 0.3 * sx                            # start left of the axis; clipping squares the baseline end
    patch = FancyBboxPatch((x0 - pad, y - height / 2), value - x0 + pad, height,
                           boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=sy / sx,
                           linewidth=0, facecolor=color, clip_on=True, zorder=3)
    ax.add_patch(patch)


def _save(fig, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def encounter_geometry(res, path):
    scn = res["scenario"]
    pc = np.array(res["baseline"]["pc_table"])
    xi = np.array(res["baseline"]["miss_xi_table"])
    sx, sz, zeta = scn["sigma_xi"], scn["sigma_zeta"], scn["miss_zeta"]
    outcomes = [("nobody burns", (0, 0)), ("one small burn", (1, 0)), ("two small burns", (1, 1)),
                ("one large burn", (2, 0)), ("small + large", (2, 1)), ("two large burns", (2, 2))]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    _titles(fig, "A burn slides the predicted pass point out of the uncertainty ellipse",
            "Encounter plane at closest approach. Burns are tangential, two orbits early: "
            f"1 cm/s moves the pass point {scn['gain_m_per_mps'] * 0.01:.0f} m.")
    ax = fig.add_axes([0.075, 0.13, 0.555, 0.66])
    for n_sig, fill in zip((3, 2, 1), ELLIPSE_FILL):
        ax.add_patch(Ellipse((0, 0), 2 * n_sig * sx, 2 * n_sig * sz, facecolor=fill, edgecolor=SURFACE,
                             linewidth=2.0, zorder=2))
        ax.text(-n_sig * sx + 14, 0, rf"{n_sig}$\sigma$", ha="left", va="center", fontsize=FS_NOTE, color=INK2, zorder=4)
    ax.plot([0], [0], marker="o", markersize=7, color=INK, zorder=6)
    ax.annotate("satellite 1 at the origin\n(hard-body radius 5 m)", (0, 0), xytext=(-400, -128),
                fontsize=FS_NOTE, color=INK2, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0, shrinkB=6))
    xs = [xi[a] for _, a in outcomes]
    ax.plot(xs, [zeta] * len(xs), color=AXIS, lw=1.6, zorder=3)
    ax.plot(xs, [zeta] * len(xs), linestyle="none", marker="o", markersize=12, color=ORANGE,
            markeredgecolor=SURFACE, markeredgewidth=2.2, zorder=5)
    for i, ((name, a), x) in enumerate(zip(outcomes, xs)):
        up = i % 2 == 0
        ax.annotate(name, (x, zeta), xytext=(x, zeta + (74 if up else -62)), ha="center",
                    va="bottom" if up else "top", fontsize=FS_NOTE, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0, shrinkA=2, shrinkB=8))
    ax.set_xlim(-420, 1790)
    ax.set_ylim(-175, 205)
    ax.set_xlabel(r"$\xi$: along the bisector of the two velocity directions (m)")
    ax.set_ylabel(r"$\zeta$: radial (m)")
    ax.set_title("Where satellite 2 is predicted to pass, for each joint action", loc="left", pad=12)
    ax.text(0.995, 0.975, "ellipses: 1, 2, 3 sigma of the combined position uncertainty (vertical scale stretched)",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_NOTE - 1, color=MUTED)

    ax2 = fig.add_axes([0.745, 0.13, 0.225, 0.66])
    floor = 1e-16
    names = [n for n, _ in outcomes][::-1]
    vals = [pc[a] for _, a in outcomes][::-1]
    ypos = np.arange(len(names))
    ax2.axvline(scn["pc_threshold"], color=INK2, lw=1.4, zorder=2)
    ax2.text(scn["pc_threshold"], len(names) - 0.42, r"$10^{-4}$ threshold ", ha="right", va="bottom",
             fontsize=FS_NOTE, color=INK2)
    for y, v in zip(ypos, vals):
        if v >= floor:
            ax2.plot([v], [y], marker="o", markersize=12, color=ORANGE, markeredgecolor=SURFACE,
                     markeredgewidth=2.2, zorder=5)
            if v > 1e-11:      # room on the left of the dot; never across the threshold line
                ax2.text(min(v * 0.18, scn["pc_threshold"] * 0.45), y, _sci(v), ha="right", va="center",
                         fontsize=FS_NOTE, color=INK)
            else:
                ax2.text(v * 5.5, y, _sci(v), ha="left", va="center", fontsize=FS_NOTE, color=INK)
        else:
            ax2.text(floor * 3, y, r"below $10^{-15}$", ha="left", va="center", fontsize=FS_NOTE, color=MUTED)
    ax2.set_xscale("log")
    ax2.set_xlim(floor, 3e-2)
    ax2.set_ylim(-0.6, len(names) - 0.1)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names)
    ax2.set_xticks([1e-14, 1e-10, 1e-6, 1e-2])
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("collision probability")
    ax2.set_title("Collision probability", loc="left", pad=12)
    _save(fig, path)


def concept_comparison(res, path):
    low, high = res["cost_uncertainty_low"], res["cost_uncertainty"]
    s_low, s_high = res["cost_uncertainty_low_settings"], res["cost_uncertainty_settings"]
    info = res["scheme_info"]
    rows = [("No coordinator\n(mixed Nash)", "nash_true_game"),
            ("Fixed rule\n(= best public rule)", "public"),
            ("Private,\nno margin", "private_m0"),
            (f"Private, uniform\nmargin {info['mu_uniform_max'] * 1e4:.1f}e-4", "private_uniform"),
            (f"Private, targeted\nmargin {info['mu_targeted_6'] * 1e4:.0f}e-4", "private_targeted_6")]

    def metric(block, key, name):
        if key == "nash_true_game":
            d = block[key]
            return {"violation": None, "dual": d["p_dual_inaction"], "pc": d["mean_pc"], "dv": d["delta_v"]}[name]
        d = block[key]
        return {"violation": d["violation_rate"], "dual": d["realised"]["p_dual_inaction"],
                "pc": d["realised"]["mean_pc"], "dv": d["realised"]["delta_v"]}[name]

    pct = lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"
    panels = [("violation", "A recommendation\nwould not be followed", "% of true games", 100.0, pct),
              ("dual", "Neither satellite\nmaneuvers", "% of events", 100.0, pct),
              ("pc", "Mean collision\nprobability", r"$\times10^{-5}$", 1e5, "{:.1f}".format),
              ("dv", "Mean total\ndelta-v", "cm/s", 100.0, "{:.2f}".format)]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    _titles(fig, "A private rule breaks without a margin and holds with a well-placed one",
            "10,000 sampled true games; operators who prefer to deviate do deviate. The rule itself is known to them.")
    left, width, gap, bottom, height = 0.185, 0.178, 0.022, 0.105, 0.60
    bar_h = 0.30
    for j, (name, title, unit, scale, fmt) in enumerate(panels):
        ax = fig.add_axes([left + j * (width + gap), bottom, width, height])
        vals = [[(metric(blk, key, name)) for blk in (low, high)] for _, key in rows]
        vmax = max(v * scale for pair in vals for v in pair if v is not None)
        ax.set_xlim(0, vmax * 1.42)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        fig.canvas.draw()
        for i, pair in enumerate(vals):
            for b, (v, color) in enumerate(zip(pair, RAMP2)):
                y = i + (b - 0.5) * (bar_h + 0.06)
                if v is None:
                    if b == 0:
                        ax.text(vmax * 0.03, i, "no recommendation", ha="left", va="center", fontsize=FS_NOTE, color=MUTED)
                    continue
                _barh(ax, y, v * scale, bar_h, color)
                ax.text(v * scale + vmax * 0.035, y, fmt(v * scale), ha="left", va="center",
                        fontsize=FS_NOTE, color=INK)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows] if j == 0 else [], fontsize=FS_LABEL, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(unit)
        ax.set_title(title, loc="left", pad=10, fontsize=FS_PANEL - 1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    handles = [matplotlib.patches.Patch(facecolor=RAMP2[0], label=f"costs known to about {int(s_low['sig_fuel'] * 100)}%"),
               matplotlib.patches.Patch(facecolor=RAMP2[1], label=f"costs known to about {int(s_high['sig_fuel'] * 100)}%")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.18, 0.862), ncol=2, fontsize=FS_LABEL,
               handlelength=1.2, columnspacing=2.0)
    _save(fig, path)


def _percent_log_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.3)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _unit_label(ax, text):
    """A horizontal unit line above the plot, instead of a rotated y-axis label."""
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_NOTE, color=INK2)


def margin_tradeoff(res, path):
    scan = res["margin_scan"]["targeted"]
    mu = np.array([r["mu"] for r in scan]) * 1e4
    mu_uniform = res["scheme_info"]["mu_uniform_max"] * 1e4
    public = res["nominal_comparison"]["public"]["coordinator_cost"] * 100
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    _titles(fig, "One margin on the incentive constraints does two jobs",
            "Private rule with slack on the recommendations to burn. Slack is in units of the operator's satellite value.")
    left, width, gap, bottom, height = 0.065, 0.265, 0.06, 0.14, 0.535
    xlab = r"margin ($10^{-4}$ of satellite value)"
    floor = 1e-4

    def uniform_marker(ax, y_text, va):
        ax.axvline(mu_uniform, color=AXIS, lw=1.4, zorder=1)
        ax.text(mu_uniform + 0.15, y_text, "largest margin if put\non every constraint", ha="left", va=va,
                fontsize=FS_NOTE - 1, color=MUTED)

    ax = fig.add_axes([left, bottom, width, height])
    uniform_marker(ax, 2.5, "bottom")
    ax.plot(mu, [r["nominal_cost"] * 100 for r in scan], color=BLUE, lw=LW, solid_capstyle="round", zorder=3)
    ax.axhline(public, color=INK2, lw=1.4, zorder=2)
    ax.text(mu[-1] + 0.3, public + 0.04, "fixed / public rule", ha="right", va="bottom", fontsize=FS_NOTE, color=INK2)
    ax.text(mu[len(mu) // 2] + 0.3, scan[len(mu) // 2]["nominal_cost"] * 100 - 0.08, "private rule", ha="left", va="top",
            fontsize=FS_NOTE, color=INK)
    ax.set_ylim(2.4, 4.2)
    ax.set_xlim(mu[0], mu[-1] + 0.3)

    def broken_line(ax, y, color, label):
        """Draw y on the log panel without the points below the floor (the line breaks there).

        A stretch that comes back above the floor is drawn as the data give it; a stretch of a
        single point gets a dot. Returns the index of the last point of the first stretch."""
        y = np.asarray(y, dtype=float)
        above = y >= floor
        ax.plot(mu, np.where(above, y, np.nan), color=color, lw=LW, solid_capstyle="round", label=label)
        for i in np.nonzero(above)[0]:
            alone = (i == 0 or not above[i - 1]) and (i == len(y) - 1 or not above[i + 1])
            if alone:
                ax.plot([mu[i]], [y[i]], marker="o", markersize=6, color=color, linestyle="none")
        below = np.nonzero(~above)[0]
        return int(below[0]) - 1 if below.size else len(y) - 1

    def end_label(ax, i, y, text):
        # keep the label clear of the bottom spine when the stretch ends on the floor
        ax.text(mu[i] + 0.25, max(y[i], 1.4 * floor), text, ha="left", va="center", fontsize=FS_NOTE, color=INK)
    ax.set_xlabel(xlab)
    _unit_label(ax, "coordinator cost (cm/s equivalent)")
    ax.set_title("What the margin costs\nif everything is known", loc="left", pad=34)

    ax = fig.add_axes([left + width + gap, bottom, width, height])
    for k, color in zip((15, 30, 100), RAMP3):
        y = np.array([r[f"not_followed_k{k}"] for r in scan])
        last = broken_line(ax, y, color, f"{k} past events")
        end_label(ax, last, y, f"{k}")
    _percent_log_axis(ax)
    ax.set_xlim(mu[0], mu[-1] + 0.3)
    ax.set_xlabel(xlab)
    _unit_label(ax, "share of recommendations not followed")
    ax.set_title("Job 1: operators who learned\nthe rule from past events", loc="left", pad=34)
    ax.legend(loc="upper right", title="rule learned from", title_fontsize=FS_NOTE, handlelength=1.4)

    ax = fig.add_axes([left + 2 * (width + gap), bottom, width, height])
    for key, label, color in (("violation_sig0.1", "costs known to about 10%", RAMP2[0]),
                              ("violation_sig0.3", "costs known to about 30%", RAMP2[1])):
        y = np.array([r[key] for r in scan])
        last = broken_line(ax, y, color, label)
        end_label(ax, last, y, label.split()[-1])
    _percent_log_axis(ax)
    ax.set_xlim(mu[0], mu[-1] + 1.3)
    ax.set_xlabel(xlab)
    _unit_label(ax, "games with a broken recommendation")
    ax.set_title("Job 2: a coordinator who\nmisjudges operators' costs", loc="left", pad=34)
    ax.legend(loc="upper right", handlelength=1.4)
    _save(fig, path)


def epsilon_tradeoff(res, path):
    cal = res["calibration_sigma_0.1"]
    rows = [r for r in cal["rows"] if r["feasible"]]
    eps = np.array([r["eps"] for r in rows])
    # the plotted points are costs on the test games, so the reference is the public rule's test-game cost
    public = cal["public_choice"]["realised_cost"] * 100
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    _titles(fig, "The violation budget is met, but a bigger budget costs more than it saves",
            f"Cheapest rule whose violation bound is within the budget, tested on {cal['n_test']:,} fresh games. "
            f"Costs known to about {int(cal['sig_cost'] * 100)}%.")
    left, width, gap, bottom, height = 0.065, 0.265, 0.06, 0.14, 0.535
    xticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    xlabels = ["1%", "2%", "5%", "10%", "20%", "50%"]

    def x_axis(ax):
        ax.set_xscale("log")
        ax.set_xlim(0.008, 0.62)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("violation budget (epsilon)")

    ax = fig.add_axes([left, bottom, width, height])
    kinds = list(dict.fromkeys(r["kind"] for r in rows))     # candidate labels as stored in the results
    chosen = (f"cheapest at every budget:\n{kinds[0]}" if len(kinds) == 1
              else "cheapest rule:\n" + "\n".join(kinds))
    ax.text(0.56, 0.0055, chosen, ha="right", va="bottom", fontsize=FS_NOTE - 1, color=INK2, linespacing=1.2)
    ax.plot([0.008, 0.62], [0.008, 0.62], color=INK2, lw=1.4)
    ax.text(0.26, 0.36, "budget", ha="right", va="bottom", fontsize=FS_NOTE, color=INK2)
    ax.plot(eps, [r["test_violation"] for r in rows], color=BLUE, lw=LW, marker="o", markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=1.8, solid_capstyle="round")
    ax.text(0.56, 0.085, "measured", ha="right", va="top", fontsize=FS_NOTE, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 0.8)
    ax.set_yticks([0.01, 0.1, 0.5])
    ax.set_yticklabels(["1%", "10%", "50%"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    x_axis(ax)
    _unit_label(ax, "games with a broken recommendation")
    ax.set_title("Violation on fresh games\nstays under the budget", loc="left", pad=34)

    specs = [("Coordinator cost: what the rule\npromises and what it delivers", "coordinator cost (cm/s equivalent)",
              "nominal_cost", "realised_cost", True),
             ("Total delta-v: fewer burns\nthan were recommended", "total delta-v (cm/s)",
              "nominal_delta_v", "realised_delta_v", False)]
    for j, (title, unit, key_nom, key_real, with_ref) in enumerate(specs):
        ax = fig.add_axes([left + (j + 1) * (width + gap), bottom, width, height])
        nominal = np.array([r[key_nom] for r in rows]) * 100
        realised = np.array([r[key_real] for r in rows]) * 100
        ax.plot(eps, nominal, color=ORANGE, lw=LW, marker="o", markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=1.8, solid_capstyle="round", label="promised (nominal costs)")
        ax.plot(eps, realised, color=BLUE, lw=LW, marker="o", markersize=9, markeredgecolor=SURFACE,
                markeredgewidth=1.8, solid_capstyle="round", label="delivered (true costs)")
        if with_ref:
            ax.axhline(public, color=INK2, lw=1.4)
            ax.text(0.58, public * 0.93, "fixed / public rule", ha="right", va="top", fontsize=FS_NOTE, color=INK2)
            ax.set_yscale("log")
            ax.set_ylim(2.2, 32)
            ax.set_yticks([3, 10, 30])
            ax.set_yticklabels(["3", "10", "30"])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.legend(loc="upper left", handlelength=1.4)
        else:
            ax.set_ylim(1.0, 1.75)
            ax.text(eps[-1], nominal[-1] + 0.025, "promised", ha="right", va="bottom", fontsize=FS_NOTE, color=INK)
            ax.text(eps[-1], realised[-1] - 0.03, "delivered", ha="right", va="top", fontsize=FS_NOTE, color=INK)
        x_axis(ax)
        _unit_label(ax, unit)
        ax.set_title(title, loc="left", pad=34)
    _save(fig, path)


def private_vs_public(res, path):
    curves = res["learning_curves"]
    best = res["best_margin_per_k"]["targeted"]
    nom = res["nominal_comparison"]
    public = nom["public"]["coordinator_cost"] * 100
    limit = nom["private_m0"]["coordinator_cost"] * 100
    ks = np.array([r["k"] for r in curves["private_m0"]])

    exact = res.get("exact_checks")
    note = None
    if exact is not None:
        # exact (sampling-free) crossover of the targeted family; operators start from observed frequencies
        k_cross = exact["crossover"]["targeted"]["observed"]["first_k"]
        y_c = ([r["cost"] * 100 for r in exact["best_per_k"]["targeted"]["observed"] if r["k"] == k_cross][0]
               if k_cross else None)
        cross_text = f"private wins from\n{k_cross} events"
        title = (f"With a targeted margin, private recommendations pay off after {k_cross} past events" if k_cross
                 else "With a targeted margin, private recommendations do not pay off in the events shown")
        uni = exact["crossover"]["targeted"]["uniform"]
        if uni["first_k"] == 1 and not uni["later_k_failing"]:
            ending = "the private rule wins at every k."
        elif uni["first_k"] is None:
            ending = "the private rule does not win at any k checked."
        elif not uni["later_k_failing"]:
            ending = f"the private rule wins from {uni['first_k']} events."
        else:
            ending = f"the private rule first wins at {uni['first_k']} events, not at every later k."
        note = ("Operators here believe only the\nfrequencies they have seen.\n"
                "If a never-seen recommendation is\ntreated as uninformative,\n" + ending)
    else:
        k_cross = res["crossover_k"]["targeted"]
        y_c = [r["coordinator_cost"] * 100 for r in best if r["k"] == k_cross][0] if k_cross else None
        cross_text = f"private wins from\nabout {k_cross} events"
        title = "Private recommendations pay off only after operators have seen enough events"

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    _titles(fig, title,
            "Realised coordinator cost when both operators learned the rule from k past events (costs known). Lower is better.")
    ax = fig.add_axes([0.075, 0.13, 0.535, 0.645])
    ax.axhline(public, color=INK2, lw=LW * 0.85, zorder=3, label="fixed / public rule (nothing to learn)")
    ax.plot(ks, [r["coordinator_cost"] * 100 for r in curves["private_m0"]], color=ORANGE, lw=LW,
            solid_capstyle="round", zorder=4, label="private, no margin (textbook optimum)")
    ax.plot(ks, [r["coordinator_cost"] * 100 for r in curves["private_uniform"]], color=AQUA, lw=LW,
            solid_capstyle="round", zorder=4, label="private, same margin on every constraint")
    kb = np.array([r["k"] for r in best])
    ax.plot(kb, [r["coordinator_cost"] * 100 for r in best], color=BLUE, lw=LW, solid_capstyle="round", zorder=5,
            label="private, targeted margin chosen for each k")
    ax.axhline(limit, color=AXIS, lw=1.4, zorder=2)
    ax.text(1.15, limit * 0.97, f"limit if the rule were known: {limit:.2f}", ha="left", va="top", fontsize=FS_NOTE, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(2.0, 130)
    ax.set_yticks([3, 10, 30, 100])
    ax.set_yticklabels(["3", "10", "30", "100"])
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.set_xlabel("k: past events each operator has seen")
    ax.set_ylabel("realised coordinator cost (cm/s equivalent)")
    if k_cross:
        # two lines in the empty space above the blue curve and left of the green curve's descent
        ax.annotate(cross_text, (k_cross, y_c), xytext=(2.0, 7.6),
                    fontsize=FS_NOTE + 1, color=INK, ha="left", va="center", linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2, shrinkA=4, shrinkB=7,
                                    relpos=(0.75, 0.0)))
        ax.plot([k_cross], [y_c], marker="o", markersize=11, color=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=2.2, zorder=6)
    if note:
        # right of the green curve's descent, between the legend and the flat curves
        ax.text(110, 13.5, note, ha="left", va="top", fontsize=FS_NOTE - 1, color=MUTED, linespacing=1.25,
                bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"), zorder=2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.60), handlelength=1.6, fontsize=FS_NOTE)
    ax.set_title("Learning the rule", loc="left", pad=12)

    # right: what privacy buys when everything is known
    specs = [("Total delta-v (cm/s)", "delta_v", 100.0, "{:.2f}"),
             (r"Collision probability ($\times10^{-5}$)", "mean_pc", 1e5, "{:.2f}"),
             ("Coordinator cost (cm/s equivalent)", "coordinator_cost", 100.0, "{:.2f}")]
    x0, w, h, gap = 0.735, 0.235, 0.125, 0.09
    top = 0.735
    for i, (title, key, scale, fmt) in enumerate(specs):
        axb = fig.add_axes([x0, top - (i + 1) * h - i * gap, w, h])
        vals = [nom["public"][key] * scale, nom["private_m0"][key] * scale]
        axb.set_xlim(0, max(vals) * 1.32)
        axb.set_ylim(1.6, -0.6)
        fig.canvas.draw()
        for y, v, color in zip((0, 1), vals, (MUTED, BLUE)):
            _barh(axb, y, v, 0.52, color)
            axb.text(v + max(vals) * 0.03, y, fmt.format(v), ha="left", va="center", fontsize=FS_NOTE, color=INK)
        axb.set_yticks([0, 1])
        axb.set_yticklabels(["public", "private"], fontsize=FS_NOTE, color=INK)
        axb.tick_params(axis="y", length=0)
        axb.tick_params(axis="x", labelsize=FS_NOTE - 1)
        axb.grid(axis="y", visible=False)
        axb.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axb.set_title(title, loc="left", pad=6, fontsize=FS_NOTE + 1, fontweight="semibold")
    fig.text(x0 - 0.065, 0.835, "If the rule and the costs are known", fontsize=FS_PANEL, fontweight="semibold",
             color=INK, ha="left", va="top")
    _save(fig, path)


def make_all(res, out_dir):
    _setup()
    import matplotlib.ticker  # noqa: F401  (used through matplotlib.ticker above)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    encounter_geometry(res, out_dir / "encounter_geometry.png")
    concept_comparison(res, out_dir / "concept_comparison.png")
    epsilon_tradeoff(res, out_dir / "epsilon_tradeoff.png")
    margin_tradeoff(res, out_dir / "margin_tradeoff.png")
    private_vs_public(res, out_dir / "private_vs_public.png")


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent.parent
    make_all(json.loads((root / "results" / "results.json").read_text(encoding="utf-8")), root / "figures")
    print("figures written")
