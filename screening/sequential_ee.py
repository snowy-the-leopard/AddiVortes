"""
Sequential Elementary Effects screening.

Implementation of the method in:
    Boukouvalas, A., Gosling, J.P., and Maruri-Aguilar, H. (2014).
    "An Efficient Screening Method for Computer Experiments."
    Technometrics, 56(4), 422-431.

This is a from-scratch implementation of Algorithm 1 (and the sigma0
threshold heuristic of Section 4.1 / Lemma 1) that treats the model
under study as a black-box function

    y = simulate(x)      x in [0, 1]^k

Any deterministic function can be screened this way, including a
fitted AddiVortes model's predict() (see addivortes_wrapper.py).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import numpy as np
from scipy.stats import chi2, qmc


# --------------------------------------------------------------------------
# Space-filling design + "farthest points first" ordering (Algorithm 1, A)
# --------------------------------------------------------------------------

def maximin_lhs(k: int, m: int, seed: int | np.random.Generator | None = None) -> np.ndarray:
    """Generate an M-point maximin Latin Hypercube design on [0,1]^k.

    Uses scipy's optimized LHS sampler (maximin criterion via random-cd
    optimization), matching the "good space-filling design" called for
    in Section 4 of the paper.
    """
    sampler = qmc.LatinHypercube(d=k, seed=seed, optimization="random-cd")
    return sampler.random(n=m)


def order_by_max_distance(points: np.ndarray) -> np.ndarray:
    """Reorder rows of `points` from farthest-apart to closest.

    Reproduces the preprocessing stage of Algorithm 1 / Example 1:
    pick the two points that are mutually farthest apart, then
    greedily add whichever remaining point maximizes its minimum
    distance to the points already chosen.
    """
    m = points.shape[0]
    if m <= 2:
        return points.copy()

    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)

    chosen = [i, j]
    remaining = set(range(m)) - {i, j}

    while remaining:
        best_pt, best_score = None, -np.inf
        for cand in remaining:
            score = min(dist[cand, c] for c in chosen)
            if score > best_score:
                best_score, best_pt = score, cand
        chosen.append(best_pt)
        remaining.remove(best_pt)

    return points[chosen]


# --------------------------------------------------------------------------
# OAT trajectory + elementary effects (Section 3, Eq. 1)
# --------------------------------------------------------------------------

def _oat_trajectory(x0: np.ndarray, factor_order: Sequence[int], delta: float):
    pts = [x0.copy()]
    steps = []
    current = x0.copy()
    for j in factor_order:
        if current[j] + delta <= 1.0 + 1e-9:
            step = delta
        elif current[j] - delta >= -1e-9:
            step = -delta
        else:
            # Neither full +/- delta step fits (only possible when delta > 0.5).
            # Take a partial step to whichever boundary is closer.
            step = (1.0 - current[j]) if (1.0 - current[j]) <= current[j] else -current[j]
        nxt = current.copy()
        nxt[j] = current[j] + step
        pts.append(nxt)
        steps.append(step)
        current = nxt
    return pts, steps


def _trajectory_elementary_effects(
    simulate: Callable[[np.ndarray], float],
    x0: np.ndarray,
    factor_order: Sequence[int],
    delta: float,
) -> dict[int, float]:
    """Run one OAT trajectory and return {factor_index: EE_i(x0)}."""
    pts, steps = _oat_trajectory(x0, factor_order, delta)
    y = [simulate(p) for p in pts]
    ee = {}
    for pos, j in enumerate(factor_order):
        ee[j] = (y[pos + 1] - y[pos]) / steps[pos]
    return ee


# --------------------------------------------------------------------------
# sigma0 threshold heuristic (Section 4.1, Lemma 1, Eq. 5)
# --------------------------------------------------------------------------

def sigma0_threshold(R: int, delta: float, gamma: float, quantile: float = 0.99) -> float:
    """EE-variance threshold sigma0 for R trajectories.

    sigma0 = sqrt( chi2_{quantile, R-1} * (2*gamma/delta^2) / (R-1) )

    gamma is the tolerated output-scale variance for a "near-linear"
    input (Eq. 3): Y(x_i) = a*x_i + b + eps_i, eps_i ~ N(0, gamma).
    """
    if R < 2:
        raise ValueError("Need at least R=2 trajectories to define sigma0.")
    sigma_eps2 = 2.0 * gamma / delta**2
    return float(np.sqrt(chi2.ppf(quantile, R - 1) * sigma_eps2 / (R - 1)))


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclasses.dataclass
class IterationRecord:
    R: int                     # number of design points processed so far
    active_factors: list       # factors still under investigation before this update
    mu: dict
    mu_star: dict
    sigma: dict
    sigma0: float
    eliminated_this_step: list # factors newly classified as nonlinear at this step


@dataclasses.dataclass
class ScreeningResult:
    k: int
    delta: float
    gamma: float
    quantile: float
    linear_factors: list        # C at termination: linear / negligible
    nonlinear_factors: list     # A at termination: nonlinear / interacting
    mu: dict
    mu_star: dict
    sigma: dict
    n_runs: int
    m: int                      # size of the space-filling design budget (batch method would use all m points)
    history: list               # list[IterationRecord], one per processed design point (from R=2 on)
    factor_names: list | None = None

    def summary(self) -> str:
        names = self.factor_names or [str(i) for i in range(self.k)]
        batch_runs = (self.k + 1) * self.m
        pct_saved = 100 * (1 - self.n_runs / batch_runs)
        lines = [f"Sequential EE screening: {self.n_runs} simulator runs "
                 f"(batch method would need {batch_runs}, a {pct_saved:.0f}% saving).",
                 "",
                 f"{'factor':<20}{'mu':>10}{'mu*':>10}{'sigma':>10}  class"]
        for i in range(self.k):
            cls = "nonlinear" if i in self.nonlinear_factors else "linear/negligible"
            lines.append(
                f"{names[i]:<20}{self.mu.get(i, float('nan')):>10.3g}"
                f"{self.mu_star.get(i, float('nan')):>10.3g}"
                f"{self.sigma.get(i, float('nan')):>10.3g}  {cls}"
            )
        return "\n".join(lines)

    def _all_R(self):
        return [rec.R for rec in self.history] or [0]


# --------------------------------------------------------------------------
# Main algorithm (Algorithm 1)
# --------------------------------------------------------------------------

def sequential_ee_screen(
    simulate: Callable[[np.ndarray], float],
    k: int,
    m: int,
    delta: float,
    gamma: float,
    quantile: float = 0.99,
    seed: int | np.random.Generator | None = None,
    factor_names: list | None = None,
    design_points: np.ndarray | None = None,
    progress_callback: Callable[[IterationRecord], None] | None = None,
) -> ScreeningResult:
    """Run the sequential elementary effects screening algorithm.

    Parameters
    ----------
    simulate : callable
        Takes a length-k numpy array in [0, 1]^k and returns a scalar
        output. Wrap your real simulator/model to operate on the unit
        cube (see addivortes_wrapper.py for a ready-made AddiVortes
        wrapper that rescales to the model's true input bounds).
    k : number of input factors.
    m : maximum number of design points (rows of the space-filling
        design). Worst case cost is (k+1)*m simulator runs.
    delta : Morris step size, e.g. delta = p / (2*(p-1)) for a grid
        with p levels.
    gamma : tolerated output-scale variance for a near-linear effect
        (see sigma0_threshold / Eq. 3 of the paper).
    quantile : chi-square quantile used to derive sigma0 (paper uses 0.99).
    seed : RNG seed / Generator, for both the LHS design and the random
        coordinate orderings of each OAT trajectory.
    factor_names : optional list of length k for reporting.
    design_points : optional (m, k) array of pre-generated space-filling
        points in [0,1]^k, if you want to supply your own design instead
        of an internally generated maximin LHS.
    progress_callback : optional callable receiving each completed iteration
        record, useful for reporting progress during a long screening run.
    """
    rng = np.random.default_rng(seed)

    if design_points is None:
        pts = maximin_lhs(k, m, seed=rng)
    else:
        pts = np.asarray(design_points, dtype=float)
        m = pts.shape[0]
    ordered_pts = order_by_max_distance(pts)

    C = list(range(k))   # currently under investigation ("linear so far")
    A: list[int] = []    # classified nonlinear

    ee_store: dict[int, list[float]] = {i: [] for i in range(k)}
    history: list[IterationRecord] = []
    n_runs = 0

    for R, x in enumerate(ordered_pts, start=1):
        if not C:
            break

        factor_order = rng.permutation(C).tolist()
        ee = _trajectory_elementary_effects(simulate, x, factor_order, delta)
        n_runs += len(factor_order) + 1
        for j, val in ee.items():
            ee_store[j].append(val)

        if R < 2:
            continue  # first point alone: gather data, no decision yet (Algorithm 1, Step B)

        mu, mu_star, sigma = {}, {}, {}
        for j in C:
            vals = np.asarray(ee_store[j])
            mu[j] = float(vals.mean())
            mu_star[j] = float(np.abs(vals).mean())
            sigma[j] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        s0 = sigma0_threshold(R, delta, gamma, quantile)

        newly_eliminated = [j for j in C if sigma[j] > s0]

        history.append(IterationRecord(
            R=R, active_factors=list(C), mu=mu, mu_star=mu_star, sigma=sigma,
            sigma0=s0, eliminated_this_step=newly_eliminated,
        ))

        if progress_callback is not None:
            progress_callback(history[-1])

        for j in newly_eliminated:
            C.remove(j)
            A.append(j)

        if not C or R == m:
            break

    # Final moments dict, pulling from the last history record that covered
    # each factor (a factor's final moments are computed the last time it
    # was updated, whether that's when it got eliminated or the final pass).
    mu_final, mu_star_final, sigma_final = {}, {}, {}
    for rec in history:
        for j in rec.active_factors:
            if j in rec.mu:
                mu_final[j] = rec.mu[j]
                mu_star_final[j] = rec.mu_star[j]
                sigma_final[j] = rec.sigma[j]

    return ScreeningResult(
        k=k, delta=delta, gamma=gamma, quantile=quantile,
        linear_factors=sorted(C), nonlinear_factors=sorted(A),
        mu=mu_final, mu_star=mu_star_final, sigma=sigma_final,
        n_runs=n_runs, m=m, history=history, factor_names=factor_names,
    )
