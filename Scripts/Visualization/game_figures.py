"""Result mock-ups for the game model and the model comparison."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from common import C, FIGW, save, sigmoid, style

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


def game_tree():
    fig, ax = plt.subplots(figsize=(FIGW, 3.1))
    leaves = [
        ("Brake", 0.90, (1.82, 2.41), 0.34),
        ("Maintain", 0.66, (1.95, 2.68), 0.21),
        ("Brake", 0.34, (2.61, 2.12), 0.27),
        ("Maintain", 0.10, (2.88, 3.35), 0.18),
    ]
    ax.plot([0.10, 0.34], [0.5, 0.5], color="0.35", lw=1.2)
    ax.plot(0.10, 0.5, "s", color=C["A"], ms=7)
    ax.text(0.10, 0.60, "A", ha="center", fontsize=9.5, color=C["A"], fontweight="bold")
    ax.text(0.10, 0.40, "observed\nfirst move", ha="center", va="top", fontsize=7, color="0.4")
    ax.text(
        0.22, 0.52, "signal / decelerate / merge", ha="center", fontsize=7, color="0.35",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
    )
    ax.plot(0.34, 0.5, "o", color=C["L"], ms=7)
    ax.text(0.34, 0.60, "L", ha="center", fontsize=9.5, color=C["L"], fontweight="bold")

    for y_l, name, p_l in [(0.78, "Yield", 0.55), (0.22, "Continue", 0.45)]:
        ax.plot([0.34, 0.58], [0.5, y_l], color=C["L"], lw=1.2)
        ax.text(
            0.455, (0.5 + y_l) / 2, f"{name}\n$P={p_l:.2f}$", ha="center", va="center",
            fontsize=7.5, color=C["L"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
        )
        ax.plot(0.58, y_l, "o", color=C["R"], ms=7)
        ax.text(0.58, y_l + 0.075, "R", ha="center", fontsize=9.5, color=C["R"], fontweight="bold")

    for name, y_leaf, (cl, cr), p in leaves:
        y_l = 0.78 if y_leaf > 0.5 else 0.22
        ax.plot([0.58, 0.80], [y_l, y_leaf], color=C["R"], lw=1.2)
        ax.text(
            0.685, (y_l + y_leaf) / 2, name, ha="center", va="center", fontsize=7.5, color=C["R"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
        )
        ax.plot(0.80, y_leaf, "o", color="0.3", ms=3.5)
        ax.text(
            0.825, y_leaf,
            f"$(C_L, C_R) = ({cl:.2f},\\ {cr:.2f})$\njoint probability $= {p:.2f}$",
            va="center", fontsize=7.3, color="0.2",
        )

    ax.set_xlim(0.02, 1.16)
    ax.set_ylim(0.0, 1.02)
    ax.axis("off")
    ax.set_title("Sequential structure with estimated costs and response probabilities", pad=4)
    save(fig, "02_game_tree.png", "Layout mock-up - treat A as strategic only if several AV scripts exist")


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
