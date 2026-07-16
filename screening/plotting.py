"""Effect plots (mu* vs sigma) reproducing Figures 1 and 3 of the paper."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sequential_ee import ScreeningResult


def plot_final_effects(res: ScreeningResult, log_scale: bool = True, ax=None):
    """Single mu* vs sigma scatter for the final classification (like Fig 1b)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    names = res.factor_names or [str(i) for i in range(res.k)]
    for i in range(res.k):
        if i not in res.mu_star:
            continue
        x, y = res.mu_star[i], res.sigma[i]
        color = "tab:red" if i in res.nonlinear_factors else "tab:blue"
        ax.scatter(x, y, color=color, zorder=3)
        ax.annotate(names[i], (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=9)

    # final sigma0 (from the last history entry)
    if res.history:
        s0 = res.history[-1].sigma0
        ax.axhline(s0, color="crimson", linestyle="--", linewidth=1,
                    label=r"$\sigma_0$ (final)")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(r"$\mu^*$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("Elementary effects screening (final state)")
    ax.legend(loc="lower right")
    return ax


def plot_iteration_grid(res: ScreeningResult, log_scale: bool = False, max_panels: int = 9):
    """Small multiples of the mu-sigma (or mu*-sigma) plot at each iteration,
    reproducing the style of Figure 3 in the paper. Draws a line from a
    factor's previous (mu*, sigma) to its current one, as in the paper.
    """
    hist = res.history
    idx = np.linspace(0, len(hist) - 1, min(max_panels, len(hist))).astype(int)
    idx = sorted(set(idx.tolist()))
    n = len(idx)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    names = res.factor_names or [str(i) for i in range(res.k)]
    prev_pos: dict[int, tuple[float, float]] = {}

    panel = 0
    for h_i, rec in enumerate(hist):
        # update prev_pos with this record's (mu*, sigma) after drawing, if selected panel
        if h_i in idx:
            ax = axes[panel // ncols][panel % ncols]
            for j in rec.active_factors:
                if j not in rec.mu_star:
                    continue
                x, y = rec.mu_star[j], rec.sigma[j]
                if j in prev_pos:
                    px, py = prev_pos[j]
                    ax.plot([px, x], [py, y], color="green", alpha=0.6, linewidth=1, zorder=1)
                ax.scatter(x, y, color="green", zorder=3)
                ax.annotate(names[j], (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=8)
            ax.axhline(rec.sigma0, color="crimson", linestyle="--", linewidth=1)
            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title(f"iteration R={rec.R}")
            ax.set_xlabel(r"$\mu^*$")
            ax.set_ylabel(r"$\sigma$")
            panel += 1

        for j in rec.active_factors:
            if j in rec.mu_star:
                prev_pos[j] = (rec.mu_star[j], rec.sigma[j])

    for extra in range(panel, nrows * ncols):
        axes[extra // ncols][extra % ncols].axis("off")

    fig.tight_layout()
    return fig, axes
