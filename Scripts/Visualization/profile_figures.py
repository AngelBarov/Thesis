"""Result mock-ups for the latent profile model and reaction time."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from common import C, FIGW, save, sigmoid, smoothstep, style

import matplotlib.pyplot as plt

RNG = np.random.default_rng(5)
PROFILES = ["Profile 1", "Profile 2", "Profile 3"]
PCOL = [C["p1"], C["p2"], C["p3"]]
P_TRANS = np.array([[0.68, 0.26, 0.06], [0.14, 0.66, 0.20], [0.05, 0.28, 0.67]])
N_PART, N_EXP = 24, 6


def _profiles():
    z = np.zeros((N_PART, N_EXP), dtype=int)
    z[:, 0] = RNG.choice(3, size=N_PART, p=[0.42, 0.38, 0.20])
    for e in range(1, N_EXP):
        for p in range(N_PART):
            z[p, e] = RNG.choice(3, p=P_TRANS[z[p, e - 1]])
    return z


def profile_flow(z):
    fig, ax = plt.subplots(figsize=(FIGW, 3.1))
    bar_w, gap, tops = 0.16, 0.10, {}
    for e in range(N_EXP):
        counts = np.array([(z[:, e] == k).sum() for k in range(3)])
        y = 0.0
        tops[e] = {}
        for k in range(3):
            ax.add_patch(FancyBboxPatch(
                (e - bar_w / 2, y), bar_w, counts[k], boxstyle="square,pad=0",
                facecolor=PCOL[k], edgecolor="white", linewidth=0.6, zorder=3,
            ))
            if counts[k] >= 2:
                ax.text(e, y + counts[k] / 2, str(counts[k]), ha="center", va="center", fontsize=6.6, color="white", zorder=4)
            tops[e][k] = [y, y]
            y += counts[k] + gap
    ease = smoothstep(60)
    for e in range(N_EXP - 1):
        for a in range(3):
            for b in range(3):
                n = int(((z[:, e] == a) & (z[:, e + 1] == b)).sum())
                if n == 0:
                    continue
                y0 = tops[e][a][1]
                y1 = tops[e + 1][b][0]
                tops[e][a][1] += n
                tops[e + 1][b][0] += n
                x = np.linspace(e + bar_w / 2, e + 1 - bar_w / 2, 60)
                lower = y0 + (y1 - y0) * ease
                ax.fill_between(x, lower, lower + n, color=PCOL[a], alpha=0.34, lw=0, zorder=2)
    ax.set_xticks(range(N_EXP), [f"exposure {e + 1}" for e in range(N_EXP)], fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(-0.4, N_EXP - 0.6)
    ax.set_ylabel(f"participants (n = {N_PART})")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Do participants move between behavioural profiles as exposure repeats?", pad=6)
    ax.legend(
        handles=[Patch(facecolor=PCOL[k], label=PROFILES[k]) for k in range(3)],
        loc="lower center", ncol=3, bbox_to_anchor=(0.66, -0.20),
    )
    save(fig, "08_profile_flow.png", "Illustrative synthetic data - profile names must follow from the fitted results")


def profile_parameters():
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.65), gridspec_kw={"width_ratios": [1.25, 1]})
    names = [r"$w_s$", r"$w_e$", r"$w_c$", r"$\alpha$"]
    vals = np.array([[1.62, 0.38, 0.52, 1.35], [1.18, 0.66, 0.44, 2.05], [0.74, 1.05, 0.31, 2.60]])
    errs = np.array([[0.22, 0.13, 0.14, 0.30], [0.18, 0.12, 0.11, 0.26], [0.20, 0.17, 0.12, 0.34]])
    ax = axes[0]
    x = np.arange(len(names))
    w = 0.26
    for k in range(3):
        ax.bar(x + (k - 1) * w, vals[k], w, yerr=errs[k], color=PCOL[k], alpha=0.88, capsize=2, error_kw=dict(lw=0.8), label=PROFILES[k])
    ax.set_xticks(x, names)
    ax.set_ylabel("estimate (95% CI)")
    ax.set_title("Profile-specific game parameters", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax = axes[1]
    gap = np.linspace(4, 46, 200)
    for k in range(3):
        dc = vals[k, 0] * (1.9 - 0.055 * gap) - vals[k, 1] * (0.35 + 0.012 * gap)
        ax.plot(gap, sigmoid(vals[k, 3] * dc), color=PCOL[k], label=PROFILES[k])
    ax.axhline(0.5, color="0.75", lw=0.7, ls=(0, (4, 4)))
    ax.set_xlabel("space ahead of L available to A  [m]")
    ax.set_ylabel(r"$P(\mathrm{Yield})$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("What the profiles mean behaviourally", pad=4)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.suptitle(r"Profile $\rightarrow$ $\{w_s, w_e, w_c, w_I, \alpha\}$ $\rightarrow$ action probability", y=1.04, fontsize=9.5)
    fig.tight_layout()
    save(fig, "09_profile_parameters.png")


def reaction_time():
    fig, axes = plt.subplots(1, 2, figsize=(FIGW, 2.75), gridspec_kw={"width_ratios": [1, 1.15]})
    groups = [
        ("L: Yield", 1.15, 0.26, C["yield"]),
        ("L: Continue", 1.62, 0.38, C["continue"]),
        ("R: Brake", 0.94, 0.22, C["brake"]),
        ("R: Maintain", 1.48, 0.35, C["maintain"]),
    ]
    ax = axes[0]
    data = [RNG.lognormal(np.log(m), s / m, 90) for _, m, s, _ in groups]
    parts = ax.violinplot(data, positions=range(4), widths=0.72, showextrema=False, showmedians=False)
    for body, (_, _, _, col) in zip(parts["bodies"], groups):
        body.set_facecolor(col)
        body.set_alpha(0.32)
        body.set_edgecolor(col)
        body.set_linewidth(0.9)
    for i, (d, (_, _, _, col)) in enumerate(zip(data, groups)):
        ax.plot(i + RNG.normal(0, 0.055, d.size), d, "o", ms=1.6, color=col, alpha=0.5)
        ax.plot(i, np.median(d), "_", ms=16, color="0.1", mew=1.6)
    ax.set_xticks(range(4), [g[0] for g in groups], fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("reaction time  [s]")
    ax.set_title("By chosen action", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    exposures = np.arange(1, N_EXP + 1)
    for label, base, col in [("L", 1.42, C["L"]), ("R", 1.22, C["R"])]:
        indiv = base - 0.075 * (exposures - 1) + RNG.normal(0, 0.16, (N_PART, 1)) + RNG.normal(0, 0.05, (N_PART, N_EXP))
        for p in range(N_PART):
            ax.plot(exposures, indiv[p], color=col, alpha=0.11, lw=0.6)
        mean = indiv.mean(axis=0)
        sem = indiv.std(axis=0, ddof=1) / np.sqrt(N_PART)
        ax.fill_between(exposures, mean - 1.96 * sem, mean + 1.96 * sem, color=col, alpha=0.22, lw=0)
        ax.plot(exposures, mean, "o-", color=col, ms=4, label=f"Driver {label}")
    ax.set_xticks(exposures)
    ax.set_xlabel("exposure number")
    ax.set_ylabel("reaction time  [s]")
    ax.set_title("Across repeated exposures", pad=4)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.text(0.03, 0.04, "thin lines = individual participants", transform=ax.transAxes, fontsize=6.4, color="0.45")
    fig.suptitle("Reaction time from the AV cue to the human response", y=1.04, fontsize=9.5)
    fig.tight_layout()
    save(fig, "10_reaction_time.png")


def main() -> None:
    style()
    z = _profiles()
    profile_flow(z)
    profile_parameters()
    reaction_time()


if __name__ == "__main__":
    main()
