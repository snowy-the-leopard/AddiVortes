"""Helpers for screening a scalar objective over AddiVortes parameters.

The sequential EE screener needs a callable

    simulate(x) -> scalar        x in [0, 1]^k

where the screened inputs are the hyperparameters of the objective being
studied. This module provides the small glue needed to map a unit-cube point
into a dictionary of AddiVortes parameter values using the ranges from
parameters.tex.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def load_parameter_bounds_from_tex(tex_path: str | Path) -> dict[str, tuple[float, float]]:
    """Parse AddiVortes parameter search ranges from parameters.tex.

    The file uses LaTeX table rows of the form:
        Number of tessellations & $m$ & 200 & [20, 500]\n
    and
        Number of MCMC iterations & --- & 1200 & [500, 100000] \\
    """
    path = Path(tex_path)
    text = path.read_text(encoding="utf-8")

    bounds: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        if "[" not in line or "]" not in line:
            continue
        match = re.search(r"\$(?:\\)?(m|nu|q|omega|lambda_c|sigma_c)\$", line)
        if match:
            symbol = match.group(1)
            range_match = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", line)
            if range_match:
                lo = float(range_match.group(1))
                hi = float(range_match.group(2))
                bounds[symbol] = (lo, hi)
                continue

        if "Number of MCMC iterations" in line:
            range_match = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", line)
            if range_match:
                bounds["iter"] = (float(range_match.group(1)), float(range_match.group(2)))
        elif "Number of MCMC burn-ins" in line:
            range_match = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", line)
            if range_match:
                bounds["burnin"] = (float(range_match.group(1)), float(range_match.group(2)))

    return bounds


def make_parameter_simulator(
    objective: Callable[[dict[str, float]], float],
    parameter_names: Sequence[str],
    bounds: dict[str, tuple[float, float]],
    baseline: dict[str, float] | None = None,
) -> Callable[[np.ndarray], float]:
    """Build a simulator for screening AddiVortes hyperparameters.

    The returned callable accepts a length-k vector in [0, 1]^k and maps it
    to a dictionary of parameter values using the supplied bounds. The
    objective is then evaluated on that dictionary and its scalar output is
    returned, so sequential EE can screen the hyperparameters as if they were
    ordinary inputs.
    """
    if baseline is None:
        baseline = {name: bounds[name][0] for name in parameter_names}

    def simulate(x: np.ndarray) -> float:
        params = dict(baseline)
        for xi, name in zip(x, parameter_names):
            lo, hi = bounds[name]
            params[name] = lo + xi * (hi - lo)
        return float(objective(params))

    return simulate
