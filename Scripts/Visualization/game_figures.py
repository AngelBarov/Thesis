"""Result mock-ups for the game model and the model comparison."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from common import C, FIGW, PHASE_COLORS, save, sigmoid, style

import matplotlib.pyplot as plt

RNG = np.random.default_rng(11)


def _car(ax, x, y, label, color, w=7.5, h=2.2, fc=None):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.5",
            facecolor=fc if fc else "white",
            edgecolor=color,
            linewidth=1.5,
            zorder=4,
        )
    )
    ax.text(
        x, y, label, ha="center", va="center", fontsize=10, color=color,
        fontweight="bold", zorder=5,
    )


def scenario():
    fig, ax = plt.subplots(figsize=(FIGW, 2.85))
    ax.add_patch(Rectangle((0, 4), 100, 4, facecolor="0.945", edgecolor="none"))
    ax.add_patch(Polygon([(0, 0), (76, 0), (90, 4), (0, 4)], facecolor="0.90", edgecolor="none"))
    ax.plot([0, 100], [8, 8], color="0.3", lw=1.6)
    ax.plot([0, 100], [4, 4], color="0.45", lw=0.9, ls=(0, (7, 7)))
    ax.plot([0, 76], [0, 0], color="0.3", lw=1.6)
    ax.plot([76, 90], [0, 4], color="0.3", lw=1.6)
    ax.text(2, 6.9, "target lane", fontsize=7.5, color="0.4")
    ax.text(2, 0.55, "ending lane", fontsize=7.5, color="0.4")
    ax.text(84, 1.1, "lane ends", fontsize=7, color="0.35", rotation=16)

    _car(ax, 46, 2, "A", C["A"], fc="#ededed")
    _car(ax, 34, 6, "L", C["L"])
    _car(ax, 22, 2, "R", C["R"])

    ax.annotate(
        "", xy=(55.5, 4.9), xytext=(50, 3.0),
        arrowprops=dict(arrowstyle="-|>", color=C["A"], lw=1.3, connectionstyle="arc3,rad=-0.12"),
    )
    ax.text(57, 4.9, "merge", fontsize=7, color=C["A"], va="center")

    for x, y, v, col in [(46, 2, r"$v_A$", C["A"]), (34, 6, r"$v_L$", C["L"]), (22, 2, r"$v_R$", C["R"])]:
        ax.annotate(
            "", xy=(x + 6.5, y - 1.55), xytext=(x - 3.5, y - 1.55),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9),
        )
        ax.text(x + 7.6, y - 1.55, v, fontsize=7.5, color=col, va="center")

    ax.annotate("", xy=(42.2, 2.9), xytext=(25.8, 2.9), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    ax.text(34, 3.15, "following gap", fontsize=7, color="0.3", ha="center")
    ax.annotate("", xy=(64, 6.8), xytext=(37.9, 6.8), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    ax.text(51, 7.15, "space ahead of L that A can use", fontsize=7, color="0.3", ha="center")
    ax.annotate("", xy=(86, -1.0), xytext=(49.8, -1.0), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    ax.text(68, -1.55, "distance to end of lane", fontsize=7, color="0.3", ha="center")

    ax.text(30, 8.9, "L: Yield / Continue", fontsize=8, color=C["L"], ha="center")
    ax.text(20, -1.45, "R: Brake / Maintain", fontsize=8, color=C["R"], ha="center")
    ax.text(74, 8.9, "A: signal, decelerate, merge", fontsize=8, color=C["A"], ha="center")
    ax.set_xlim(-1, 101)
    ax.set_ylim(-2.6, 9.9)
    ax.axis("off")
    ax.set_title("Scenario geometry and the measured quantities the costs are built from", pad=6)
    save(fig, "01_scenario_schematic.png", "Schematic only - final version should use measured lane geometry")


def _chip(ax, x, y, w, h, label, face, edge="0.35"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color="0.12", zorder=4)


def _outcome(ax, x, y, w, h, action, detail, cost, face, edge):
    """One HMM scenario, and the Resolution ending every scenario shares."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.text(x, y + 0.48, action, ha="center", va="center", fontsize=8.5, color="0.12", fontweight="bold", zorder=4)
    ax.text(x, y + 0.14, detail, ha="center", va="center", fontsize=6.9, color="0.25", zorder=4)
    ax.text(x, y - 0.20, "outcome: Resolution", ha="center", va="center", fontsize=7.2, color="#3d5c38", zorder=4)
    ax.text(x, y - 0.50, f"$C_L = {cost:.2f}$", ha="center", va="center", fontsize=7.1, color="0.2", zorder=4)


def game_tree():
    """Example tree for L covering every path in the HMM phase chain.

    Approach, then Anticipation, then Yield, Continue, or Conflict.
    Each of those three ends in Resolution.
    """
    fig, ax = plt.subplots(figsize=(FIGW, 4.55))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 8.75)

    approach_x, decide_x, mid_y = 1.45, 4.25, 4.35
    _chip(ax, approach_x, mid_y, 2.05, 0.85, "Approach", PHASE_COLORS["Approach"])
    _chip(ax, decide_x, mid_y, 2.45, 0.85, "Anticipation", PHASE_COLORS["Anticipation"], "#b08a2e")
    ax.annotate(
        "",
        xy=(decide_x - 1.22, mid_y),
        xytext=(approach_x + 1.02, mid_y),
        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1),
        zorder=1,
    )
    ax.text(
        (approach_x + decide_x) / 2,
        mid_y + 0.62,
        "AV cue",
        ha="center",
        va="center",
        fontsize=7,
        color="0.35",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"),
    )
    ax.text(decide_x, mid_y + 0.68, "L", ha="center", va="bottom", fontsize=9, color=C["L"], fontweight="bold")

    box_w, box_h, box_x = 4.35, 1.55, 9.7
    box_left = box_x - box_w / 2
    decide_right = decide_x + 1.22
    branches = [
        (6.85, "Yield", 0.52, 1.82, "AV merges into the opened gap", PHASE_COLORS["Yield"], "#2c6b9e"),
        (4.35, "Continue", 0.33, 2.41, "L holds while the AV waits", PHASE_COLORS["Continue"], "#c05a2c"),
        (1.85, "Conflict", 0.15, 3.35, "the gap is contested, then clears", PHASE_COLORS["Conflict"], "#a85a5a"),
    ]
    for y, action, probability, cost, detail, face, edge in branches:
        ax.annotate(
            "",
            xy=(box_left, y),
            xytext=(decide_right, mid_y),
            arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.1, shrinkA=2, shrinkB=4),
            zorder=1,
        )
        if y == mid_y:
            label_x = decide_right + 0.62 * (box_left - decide_right)
            label_y = mid_y + 0.42
        else:
            label_x = decide_right + 0.32 * (box_left - decide_right)
            label_y = mid_y + 0.32 * (y - mid_y)
            label_y += 0.32 if y > mid_y else -0.32
        ax.text(
            label_x,
            label_y,
            f"$P={probability:.2f}$",
            ha="center",
            va="center",
            fontsize=7.4,
            color=edge,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
        )
        _outcome(ax, box_x, y, box_w, box_h, action, detail, cost, face, edge)

    ax.text(
        6.1,
        0.38,
        "Approach  →  Anticipation  →  Yield, Continue, or Conflict  →  Resolution",
        ha="center",
        fontsize=6.6,
        color="0.45",
    )
    ax.axis("off")
    ax.set_title("Example for L: every scenario in the phase chain", pad=4)
    save(
        fig,
        "02_game_tree.png",
        "Layout mock-up, invented numbers. Paths follow the HMM mask; each scenario ends in Resolution.",
    )


def cost_decomposition():
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.55))
    panels = [
        (axes[0], "Driver L", ["Yield", "Continue"],
         {"safety": [0.42, 1.34], "efficiency": [0.88, 0.21], "comfort": [0.46, 0.12], "interaction": [0.06, 0.28]}),
        (axes[1], "Driver R", ["Brake", "Maintain"],
         {"safety": [0.31, 1.12], "efficiency": [0.74, 0.18], "comfort": [0.58, 0.09], "interaction": [0.00, 0.00]}),
    ]
    labels = {
        "safety": r"$w_s S_i(a)$",
        "efficiency": r"$w_e E_i(a)$",
        "comfort": r"$w_c K_i(a)$",
        "interaction": r"$w_I I_i(a)$  (optional)",
    }
    for ax, title, actions, comps in panels:
        y = np.arange(len(actions))[::-1]
        left = np.zeros(len(actions))
        for key, values in comps.items():
            values = np.array(values, dtype=float)
            ax.barh(
                y, values, left=left, height=0.5, color=C[key], edgecolor="white", linewidth=0.7,
                hatch="///" if key == "interaction" else None,
                label=labels[key] if ax is axes[0] else None,
            )
            left = left + values
        for yi, total in zip(y, left):
            ax.text(total + 0.05, yi, f"$C_i = {total:.2f}$", va="center", fontsize=7.5, color="0.2")
        ax.set_yticks(y, actions)
        ax.set_xlim(0, max(left) * 1.34)
        ax.set_xlabel("weighted cost contribution")
        ax.set_title(title)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        delta = left[1] - left[0]
        p = sigmoid(1.8 * delta)
        ax.text(
            0.98, 0.5,
            f"$\\Delta C = {delta:+.2f}$\n$P(\\mathrm{{{actions[0]}}}) = {p:.2f}$",
            transform=ax.transAxes, ha="right", va="center", fontsize=7.3, color="0.25",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="0.96", edgecolor="0.85", linewidth=0.6),
        )
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.11), columnspacing=1.3, handlelength=1.1)
    fig.suptitle("Which cost component drives each response, and by how much", y=1.04, fontsize=9.5)
    fig.tight_layout()
    save(fig, "03_cost_decomposition.png")


def decision_map():
    fig, ax = plt.subplots(figsize=(FIGW * 0.78, 3.0))
    gap = np.linspace(4, 46, 220)
    rel = np.linspace(-6, 6, 220)
    G, V = np.meshgrid(gap, rel)
    p_yield = sigmoid(1.35 * (2.45 - 0.105 * G - 0.20 * V))
    mesh = ax.pcolormesh(G, V, p_yield, cmap="RdBu", vmin=0, vmax=1, shading="auto", rasterized=True)
    cs = ax.contour(
        G, V, p_yield, levels=[0.25, 0.5, 0.75], colors=["0.35", "0.1", "0.35"],
        linewidths=[0.7, 1.4, 0.7], linestyles=["dashed", "solid", "dashed"],
    )
    ax.clabel(cs, fmt="%.2f", fontsize=6.5)
    n = 150
    g_obs = RNG.uniform(5, 45, n)
    v_obs = RNG.uniform(-5.5, 5.5, n)
    yielded = RNG.random(n) < sigmoid(1.35 * (2.45 - 0.105 * g_obs - 0.20 * v_obs))
    ax.plot(g_obs[yielded], v_obs[yielded], "o", ms=3.4, mfc="white", mec="0.15", mew=0.7, label="observed Yield")
    ax.plot(g_obs[~yielded], v_obs[~yielded], "x", ms=3.8, color="0.15", mew=0.9, label="observed Continue")
    ax.set_xlabel("space ahead of L available to A  [m]")
    ax.set_ylabel(r"relative speed $v_L - v_A$  [m/s]")
    ax.set_title("Predicted $P(\\mathrm{Yield})$ over the traffic situation", pad=5)
    ax.legend(loc="upper right", framealpha=0.9, frameon=True, fontsize=7)
    fig.colorbar(mesh, ax=ax, label="$P(\\mathrm{Yield})$", pad=0.02)
    fig.tight_layout()
    save(fig, "04_decision_map.png")


def model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.7), sharex=True)
    models = ["Statistical\nbaseline", "Pairwise\ngame", "Full A-L-R\ngame", "Full game\n+ profiles"]
    acc_l = np.array([0.63, 0.71, 0.74, 0.77])
    acc_r = np.array([0.60, 0.68, 0.73, 0.74])
    acc_err = np.array([0.045, 0.042, 0.040, 0.046])
    ll_l = np.array([0.64, 0.55, 0.50, 0.47])
    ll_r = np.array([0.66, 0.58, 0.52, 0.51])
    ll_err = np.array([0.035, 0.032, 0.030, 0.036])
    x = np.arange(len(models))
    w = 0.36
    ax = axes[0]
    ax.bar(x - w / 2, acc_l, w, yerr=acc_err, color=C["L"], alpha=0.85, capsize=2.5, error_kw=dict(lw=0.9), label="L: Yield / Continue")
    ax.bar(x + w / 2, acc_r, w, yerr=acc_err, color=C["R"], alpha=0.85, capsize=2.5, error_kw=dict(lw=0.9), label="R: Brake / Maintain")
    ax.axhline(0.5, color="0.4", lw=0.9, ls=(0, (4, 3)))
    ax.text(3.45, 0.512, "chance", fontsize=6.6, color="0.4", ha="right")
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Higher is better", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.bar(x - w / 2, ll_l, w, yerr=ll_err, color=C["L"], alpha=0.85, capsize=2.5, error_kw=dict(lw=0.9))
    ax.bar(x + w / 2, ll_r, w, yerr=ll_err, color=C["R"], alpha=0.85, capsize=2.5, error_kw=dict(lw=0.9))
    ax.set_ylim(0.3, 0.78)
    ax.set_ylabel("log loss")
    ax.set_title("Lower is better", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for ax in axes:
        ax.set_xticks(x, models, fontsize=7)
    fig.suptitle("Model progression, scored on held-out participants", y=1.04, fontsize=9.5)
    fig.tight_layout()
    save(fig, "11_model_comparison.png")


def main() -> None:
    style()
    scenario()
    game_tree()
    cost_decomposition()
    decision_map()
    model_comparison()


if __name__ == "__main__":
    main()
