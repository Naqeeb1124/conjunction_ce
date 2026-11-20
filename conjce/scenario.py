"""Sample sizes for the scenario approach, and the scenario version of the coordinator's program.

A scenario program replaces "the constraint holds with probability at least 1 - eps" by "the
constraint holds for each of N sampled scenarios".  For a convex program with d decision
variables that is feasible for every sample, the solution violates the constraint for a new
scenario with probability more than eps only with probability at most

    beta = sum_{i=0}^{d-1} C(N, i) eps^i (1 - eps)^(N - i)

(Campi and Garatti, SIAM J. Optim. 19(3), 2008).  The earlier explicit sufficient condition of
Calafiore and Campi (IEEE Trans. Automatic Control 51(5), 2006) is

    N >= (2 / eps) ln(1 / beta) + 2 d + (2 d / eps) ln(2 / eps).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import games as G

__all__ = ["n_campi_garatti", "n_calafiore_campi", "violation_confidence", "solve_scenario_private"]


def violation_confidence(n: int, eps: float, d: int) -> float:
    """beta(N) = P(Binomial(N, eps) <= d - 1): the Campi-Garatti bound on the probability that the
    scenario solution has violation probability above eps."""
    return float(stats.binom.cdf(d - 1, n, eps))


def n_campi_garatti(eps: float, beta: float, d: int) -> int:
    """Smallest N with sum_{i<d} C(N,i) eps^i (1-eps)^(N-i) <= beta."""
    lo, hi = d, d
    while violation_confidence(hi, eps, d) > beta:
        hi *= 2
    while hi - lo > 1:                       # the tail is decreasing in N for N >= d
        mid = (lo + hi) // 2
        if violation_confidence(mid, eps, d) > beta:
            lo = mid
        else:
            hi = mid
    return hi


def n_calafiore_campi(eps: float, beta: float, d: int) -> int:
    """Explicit sample size of Calafiore and Campi (2006)."""
    return math.ceil((2.0 / eps) * math.log(1.0 / beta) + 2.0 * d + (2.0 * d / eps) * math.log(2.0 / eps))


def solve_scenario_private(c1s, c2s, objective, extra=None):
    """Private recommendations whose obedience constraints hold in every sampled game.

    ``c1s`` and ``c2s`` have shape (N, m1, m2): one cost table per scenario.  All 12 * N obedience
    rows are imposed at once.  Returns the same dict as ``games.solve_private``.
    """
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
