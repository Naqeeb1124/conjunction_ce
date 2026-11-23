from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import games as G

__all__ = ["n_campi_garatti", "n_calafiore_campi", "violation_confidence", "solve_scenario_private"]


def violation_confidence(n: int, eps: float, d: int) -> float:
    return float(stats.binom.cdf(d - 1, n, eps))


def n_campi_garatti(eps: float, beta: float, d: int) -> int:
    lo, hi = d, d
    while violation_confidence(hi, eps, d) > beta:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if violation_confidence(mid, eps, d) > beta:
            lo = mid
        else:
            hi = mid
    return hi


def n_calafiore_campi(eps: float, beta: float, d: int) -> int:
    return math.ceil((2.0 / eps) * math.log(1.0 / beta) + 2.0 * d + (2.0 * d / eps) * math.log(2.0 / eps))


def solve_scenario_private(c1s, c2s, objective, extra=None):
    rows = []
    for c1, c2 in zip(c1s, c2s):
        A, _, _ = G.obedience_system(c1, c2)
        rows.append(A)
    A_all = np.vstack(rows)
    objective = np.asarray(objective, dtype=float)
    xf, status = G._solve_lp(objective.ravel(), A_all, extra=extra)
    if xf is None:
        return {"x": None, "objective": np.inf, "status": status}
    return {"x": xf.reshape(objective.shape), "objective": float(objective.ravel() @ xf), "status": status}
