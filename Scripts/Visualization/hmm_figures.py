"""Result mock-ups for the hidden Markov model."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Patch

from common import C, FIGW, PHASE_COLORS, save, style

import matplotlib.pyplot as plt

RNG = np.random.default_rng(23)

# Same names as Scripts/HMM/phases.py.
PHASES = ["Approach", "Anticipation", "Yield", "Continue", "Conflict", "Resolution"]
TRANS = {
    "Approach": {"Approach": 0.86, "Anticipation": 0.14},
    "Anticipation": {"Anticipation": 0.71, "Yield": 0.13, "Continue": 0.11, "Conflict": 0.05},
    "Yield": {"Yield": 0.78, "Resolution": 0.22},
    "Continue": {"Continue": 0.81, "Resolution": 0.19},
    "Conflict": {"Conflict": 0.60, "Resolution": 0.40},
    "Resolution": {"Resolution": 1.00},
}
POS = {
    "Approach": (0.08, 0.50),
    "Anticipation": (0.33, 0.50),
    "Yield": (0.60, 0.80),
    "Continue": (0.60, 0.50),
    "Conflict": (0.60, 0.20),
    "Resolution": (0.88, 0.50),
}
FIG_H = 2.9
Y_SPAN = 1.20


def _self_loop(ax, x, y, p):
    d = 0.30
    w = d / FIGW
    h = d / FIG_H * Y_SPAN
    cy = y + 0.058 + h / 2
    ax.add_patch(Arc((x, cy), w, h, theta1=-35, theta2=245, color="0.45", lw=0.5 + 1.8 * p, zorder=2))
    ang = np.deg2rad(245)
    ax.plot(
        x + w / 2 * np.cos(ang), cy + h / 2 * np.sin(ang),
        marker=(3, 0, 245), ms=3.6 + 1.6 * p, color="0.45", zorder=2,
    )
    ax.text(x - w / 2 - 0.008, cy, f"{p:.2f}", ha="right", va="center", fontsize=6.4, color="0.4")


def transitions():
    fig, ax = plt.subplots(figsize=(FIGW, FIG_H))
    for src, targets in TRANS.items():
        x0, y0 = POS[src]
        for dst, p in targets.items():
            if src == dst:
                _self_loop(ax, x0, y0, p)
                continue
            x1, y1 = POS[dst]
            ax.add_patch(FancyArrowPatch(
                (x0 + 0.055, y0), (x1 - 0.055, y1),
                arrowstyle="-|>", mutation_scale=9, color="0.3", lw=0.5 + 3.6 * p,
                connectionstyle="arc3,rad=0.06", shrinkA=2, shrinkB=2, zorder=2,
            ))
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2 + 0.045, f"{p:.2f}", ha="center", fontsize=6.4, color="0.25",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
            )
    for name, (x, y) in POS.items():
        ax.add_patch(FancyBboxPatch(
            (x - 0.055, y - 0.052), 0.11, 0.104,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=PHASE_COLORS[name], edgecolor="0.35", linewidth=0.9, zorder=3,
        ))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.2, zorder=4)
    ax.text(
        0.5, 0.02,
        "arrow width proportional to the estimated transition probability; curved arrow = probability of staying",
        ha="center", fontsize=6.8, color="0.45", transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, -0.06 + Y_SPAN)
    ax.axis("off")
    ax.set_title("Estimated phase transitions within one merge", pad=2)
    save(fig, "05_hmm_transitions.png")


def _ramp(t, start, width):
    return 1.0 / (1.0 + np.exp(-(t - start) / width))


def single_trial():
    fig, axes = plt.subplots(
        3, 1, figsize=(FIGW, 4.1), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1, 0.85], "hspace": 0.16},
    )
    t = np.arange(-1.0, 6.0, 0.02)
    segments = [
        ("Approach", -1.0, 0.6), ("Anticipation", 0.6, 1.75),
        ("Yield", 1.75, 3.4), ("Resolution", 3.4, 6.0),
    ]
    for ax in axes:
        for name, t0, t1 in segments:
            ax.axvspan(t0, t1, color=PHASE_COLORS[name], alpha=0.55, lw=0, zorder=0)

    ax = axes[0]
    ax.plot(t, 24.0 - 3.2 * _ramp(t, 0.9, 0.45), color=C["A"], ls="--", label="A (AV)")
    ax.plot(t, 25.0 - 4.6 * _ramp(t, 2.0, 0.5), color=C["L"], label="L")
    ax.plot(t, 24.5 - 5.4 * _ramp(t, 2.5, 0.55), color=C["R"], label="R")
    ax.set_ylabel("speed  [m/s]")
    ax.legend(loc="lower left", ncol=3)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(t, 14.0 + 9.0 * _ramp(t, 2.2, 0.6), color="0.2", label="gap ahead of L  [m]")
    ax.set_ylabel("gap  [m]")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.plot(t, 3.0 + 5.5 * _ramp(t, 2.3, 0.65), color="#8a76b8", ls="-.", label="TTC  [s]")
    ax2.set_ylabel("TTC  [s]", color="#8a76b8")
    ax2.tick_params(axis="y", colors="#8a76b8")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#8a76b8")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right", ncol=2)
    for x, label, col in [(0.0, "AV cue", "0.25"), (1.42, "L responds", C["L"])]:
        ax.text(x + 0.08, 0.97, label, transform=ax.get_xaxis_transform(), fontsize=7, color=col, rotation=90, va="top")

    ax = axes[2]
    stress = 0.30 + 0.34 * _ramp(t, 1.1, 0.4) - 0.22 * _ramp(t, 3.6, 0.8)
    ax.plot(t, stress + RNG.normal(0, 0.012, t.size), color="#b5452f")
    ax.axhspan(0.26, 0.34, color="0.85", lw=0, zorder=1)
    ax.text(5.9, 0.30, "baseline", fontsize=6.4, color="0.45", ha="right", va="center")
    ax.set_ylabel("stress  [a.u.]")
    ax.set_xlabel("time relative to the AV cue  [s]")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    for ax in axes:
        ax.axvline(0, color="0.2", lw=1.0, ls=(0, (3, 3)))
        ax.axvline(1.42, color=C["L"], lw=1.0, ls=(0, (1, 2)))
    axes[0].annotate("", xy=(1.42, 25.85), xytext=(0.0, 25.85), arrowprops=dict(arrowstyle="<->", color="0.2", lw=0.9))
    axes[0].text(0.71, 26.1, r"$\mathrm{RT}_L = 1.42$ s", ha="center", fontsize=7, color="0.2")
    axes[0].set_ylim(18.5, 27.0)
    axes[0].set_xlim(-1.0, 6.0)

    handles = [Patch(facecolor=PHASE_COLORS[name], alpha=0.55, label=name) for name, _, _ in segments]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09), title="most likely phase", title_fontsize=7.5)
    fig.suptitle("One trial: the phase sequence read onto the measured signals", y=0.985, fontsize=9.5)
    save(fig, "06_single_trial_phases.png")


def phase_raster():
    fig, ax = plt.subplots(figsize=(FIGW, 3.6))
    n_part, n_exp, row = 4, 6, 0
    yticks, ylabels = [], []
    for p in range(n_part):
        for e in range(n_exp):
            t_cur = -1.0
            onset = 0.45 + 0.25 * RNG.random() - 0.04 * e
            decide = onset + 0.75 + 0.7 * RNG.random() - 0.07 * e
            resolve = decide + 1.3 + 0.8 * RNG.random()
            choice = ["Yield", "Continue", "Conflict"][int(RNG.choice(3, p=[0.55, 0.36, 0.09]))]
            for name, t1 in [("Approach", onset), ("Anticipation", decide), (choice, resolve), ("Resolution", 6.0)]:
                ax.barh(row, t1 - t_cur, left=t_cur, height=0.74, color=PHASE_COLORS[name], edgecolor="white", linewidth=0.3, zorder=2)
                t_cur = t1
            ax.plot(decide, row, "o", ms=2.6, mfc="white", mec="0.15", mew=0.6, zorder=3)
            yticks.append(row)
            ylabels.append(f"e{e + 1}")
            row += 1
        row += 0.9
    ax.axvline(0, color="0.15", lw=1.1, ls=(0, (3, 3)), zorder=4)
    ax.text(0.08, row - 0.4, "AV cue", fontsize=7, color="0.25")
    ax.set_yticks(yticks, ylabels, fontsize=6.2)
    ax.invert_yaxis()
    for p in range(n_part):
        centre = p * (n_exp + 0.9) + (n_exp - 1) / 2
        ax.text(-1.45, centre, f"P{p + 1:02d}", fontsize=7.5, va="center", ha="right", rotation=90)
    ax.set_xlim(-1.0, 6.0)
    ax.set_xlabel("time relative to the AV cue  [s]")
    ax.set_title("Phase sequences across participants and repeated exposures", pad=6)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=PHASE_COLORS[name], label=name) for name in PHASES]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.15", ms=4, label="decision onset"))
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.30))
    save(fig, "07_phase_raster.png")


def main() -> None:
    style()
    transitions()
    single_trial()
    phase_raster()


if __name__ == "__main__":
    main()
