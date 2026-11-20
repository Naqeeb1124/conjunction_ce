"""Equilibria and coordinator programs for two-player games in cost form.

Conventions: ``c1[a1, a2]`` and ``c2[a1, a2]`` are the costs each player wants to minimise.
A recommendation rule is a probability table ``x[a1, a2]`` over joint actions.
"""
from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np

__all__ = [
    "pure_nash", "support_enumeration", "nash_regret", "obedience_system", "obedience_slack",
    "conditional_advantages", "belief_dependent_rows", "uncertain_belief_rows", "reveals_other",
    "solve_private", "solve_public", "solve_public_hull", "product_table", "point_table",
]


# Nash equilibria
def pure_nash(c1, c2, margin: float = 0.0, tol: float = 1e-12):
    """Pure Nash equilibria.  With ``margin`` > 0, every unilateral deviation must cost at least
    ``margin`` more than staying (a strict equilibrium with that much slack)."""
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    m1, m2 = c1.shape
    out = []
    for a1, a2 in itertools.product(range(m1), range(m2)):
        dev1 = np.delete(c1[:, a2], a1)
        dev2 = np.delete(c2[a1, :], a2)
        if np.all(c1[a1, a2] + margin <= dev1 + tol) and np.all(c2[a1, a2] + margin <= dev2 + tol):
            out.append((a1, a2))
    return out


def _indifference(mat):
    """Mixture z over the columns of ``mat`` (rows = the indifferent player's supported actions)
    such that mat @ z is the same for every row.  Returns (z, common value) or None."""
    k = mat.shape[0]
    lhs = np.zeros((k + 1, k + 1))
    lhs[:k, :k] = mat
    lhs[:k, k] = -1.0
    lhs[k, :k] = 1.0
    rhs = np.zeros(k + 1)
    rhs[k] = 1.0
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:k], sol[k]


def support_enumeration(c1, c2, tol: float = 1e-9):
    """All Nash equilibria of a non-degenerate bimatrix cost game, by support enumeration.

    For every pair of equal-size supports (I for player 1, J for player 2): find player 2's
    mixture q on J that makes player 1 indifferent across I, and player 1's mixture p on I that
    makes player 2 indifferent across J; keep the pair if both are probability vectors and no
    action outside the supports does strictly better.  Returns a list of (p, q).
    """
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    m1, m2 = c1.shape
    scale = max(np.max(np.abs(c1)), np.max(np.abs(c2)), 1e-300)
    found = []
    for size in range(1, min(m1, m2) + 1):
        for I in itertools.combinations(range(m1), size):
            for J in itertools.combinations(range(m2), size):
                rq = _indifference(c1[np.ix_(I, J)])          # rows I indifferent under q
                rp = _indifference(c2[np.ix_(I, J)].T)        # columns J indifferent under p
                if rq is None or rp is None:
                    continue
                (q_s, u), (p_s, w) = rq, rp
                if np.any(q_s < -tol) or np.any(p_s < -tol):
                    continue
                p = np.zeros(m1)
                q = np.zeros(m2)
                p[list(I)] = np.clip(p_s, 0.0, None)
                q[list(J)] = np.clip(q_s, 0.0, None)
                p /= p.sum()
                q /= q.sum()
                if np.any(c1 @ q < u - tol * scale) or np.any(p @ c2 < w - tol * scale):
                    continue                                   # a profitable deviation exists
                if not any(np.allclose(p, p0, atol=1e-7) and np.allclose(q, q0, atol=1e-7) for p0, q0 in found):
                    found.append((p, q))
    return found


def nash_regret(p, q, c1, c2):
    """What each player could save by its best unilateral deviation from the mixed profile (p, q).
    Both numbers are zero at a Nash equilibrium."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    return float(p @ c1 @ q - np.min(c1 @ q)), float(p @ c2 @ q - np.min(p @ c2))


def product_table(p, q) -> np.ndarray:
    """Joint-action table of independent mixed strategies."""
    return np.outer(np.asarray(p, dtype=float), np.asarray(q, dtype=float))


def point_table(profile, shape=(3, 3)) -> np.ndarray:
    """Joint-action table that puts probability one on a single profile.

    ``profile`` may be a tuple, a list or an array of action indices.  It is converted to a tuple
    first: indexing with a list would select whole rows instead of one cell."""
    x = np.zeros(shape)
    x[tuple(int(a) for a in profile)] = 1.0
    return x


# Obedience constraints
def obedience_system(c1, c2):
    """Linear form of the correlated-equilibrium (obedience) constraints.

    Returns ``(A, P, info)`` such that, with x flattened row-major (index a1 * m2 + a2):

    * ``A @ x <= 0`` are the obedience constraints, one row per (player, recommended action r,
      deviation d != r).  For player 1 the row is  sum_{a2} x[r, a2] * (c1[r, a2] - c1[d, a2]).
    * ``P @ x`` is, for each row, the probability that the player receives recommendation r.
    * ``info[j] = (player, r, d)``.

    A margin mu turns the constraints into ``(A + mu * P) @ x <= 0``: conditional on being told
    r, obeying must be cheaper than deviating to d by at least mu.
    """
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    m1, m2 = c1.shape
    rows_a, rows_p, info = [], [], []
    for r in range(m1):
        for d in range(m1):
            if d == r:
                continue
            a = np.zeros((m1, m2))
            p = np.zeros((m1, m2))
            a[r, :] = c1[r, :] - c1[d, :]
            p[r, :] = 1.0
            rows_a.append(a.ravel())
            rows_p.append(p.ravel())
            info.append((0, r, d))
    for r in range(m2):
        for d in range(m2):
            if d == r:
                continue
            a = np.zeros((m1, m2))
            p = np.zeros((m1, m2))
            a[:, r] = c2[:, r] - c2[:, d]
            p[:, r] = 1.0
            rows_a.append(a.ravel())
            rows_p.append(p.ravel())
            info.append((1, r, d))
    return np.array(rows_a), np.array(rows_p), info


def obedience_slack(x, c1, c2) -> np.ndarray:
    """Left-hand sides A @ x of the obedience constraints (<= 0 means obeyed)."""
    A, _, _ = obedience_system(c1, c2)
    return A @ np.asarray(x, dtype=float).ravel()


def conditional_advantages(x, c1, c2, min_prob: float = 1e-12):
    """For each recommendation that has positive probability: how much cheaper obeying is than
    each deviation, conditional on that recommendation.  Returns a list of dicts."""
    A, P, info = obedience_system(c1, c2)
    xf = np.asarray(x, dtype=float).ravel()
    out = []
    for a_row, p_row, (player, r, d) in zip(A, P, info):
        prob = float(p_row @ xf)
        if prob > min_prob:
            out.append({"player": player, "rec": r, "dev": d, "prob": prob,
                        "advantage": float(-(a_row @ xf) / prob)})
    return out


# Coordinator LPs
def _solve_lp(obj, A_ub, extra=None, zero_idx=None):
    """min obj @ x  s.t.  A_ub @ x <= 0,  extra rows (a, b) meaning a @ x <= b,  x in the simplex."""
    n = obj.size
    x = cp.Variable(n, nonneg=True)
    cons = [cp.sum(x) == 1]
    if A_ub is not None and len(A_ub):
        scale = np.max(np.abs(A_ub), axis=1, keepdims=True)     # row scaling: costs are ~1e-4
        scale[scale == 0] = 1.0
        cons.append((A_ub / scale) @ x <= 0)
    for a, b in (extra or []):
        cons.append(np.asarray(a, dtype=float).ravel() @ x <= float(b))
    if zero_idx is not None and len(zero_idx):
        cons.append(x[list(zero_idx)] == 0)
    prob = cp.Problem(cp.Minimize(obj @ x), cons)
    prob.solve(solver=cp.HIGHS)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        return None, prob.status
    val = np.clip(np.asarray(x.value, dtype=float), 0.0, None)
    val[val < 1e-12] = 0.0
    return val / val.sum(), prob.status


def belief_dependent_rows(A) -> np.ndarray:
    """Which obedience constraints depend on what the player believes about the other.

    Row j holds, for each action of the other player, the cost of obeying minus the cost of
    deviating.  If all entries are <= 0, obeying is at least as good whatever the other does, so
    no belief (and no estimate of the rule) can make the player deviate.  A margin for
    learnability is only needed on rows that have entries of both signs.
    """
    A = np.asarray(A, dtype=float)
    return (A.min(axis=1) < 0) & (A.max(axis=1) > 0)


def uncertain_belief_rows(x0, c1, c2, tol: float = 1e-9) -> np.ndarray:
    """The obedience constraints on which a player who has to learn the rule x0 can go wrong.

    A row (player, recommendation r, deviation d) is selected when both hold:

    * its sign depends on the other player's action (``belief_dependent_rows``), and
    * under x0 the recommendation r leaves the player uncertain: more than one action of the
      other player has positive probability given r.

    If r reveals the other's action, one past observation already gives the exact belief, so no
    estimation error is possible on that row.
    """
    x0 = np.asarray(x0, dtype=float)
    A, _, info = obedience_system(c1, c2)
    dependent = belief_dependent_rows(A)
    out = np.zeros(len(info), dtype=bool)
    for j, (player, r, _) in enumerate(info):
        given_r = x0[r, :] if player == 0 else x0[:, r]
        out[j] = dependent[j] and np.count_nonzero(given_r > tol) > 1
    return out


def reveals_other(x, tol: float = 1e-9) -> bool:
    """True if every recommendation that can be issued tells the player exactly what the other
    was told.  Such a rule is public in effect: nothing is left for the player to infer."""
    x = np.asarray(x, dtype=float)
    rows_ok = all(np.count_nonzero(x[r, :] > tol) <= 1 for r in range(x.shape[0]))
    cols_ok = all(np.count_nonzero(x[:, r] > tol) <= 1 for r in range(x.shape[1]))
    return rows_ok and cols_ok


def solve_private(c1, c2, objective, margin=0.0, extra=None, margin_rows="all"):
    """Private recommendations: the coordinator's best correlated equilibrium.

        minimise   sum_a x(a) * objective(a)
        subject to (A + mu * P) x <= 0   (obedience with margin mu),   x a probability table.

    ``margin`` is a scalar or a pair (mu1, mu2), in the units of the cost tables.
    ``margin_rows`` chooses where the margin applies:

    * ``"all"``: on every obedience constraint.  This is the form of the chance-constrained
      correlated equilibrium of Im, Topcu and Fridovich-Keil with an additive cost perturbation.
    * ``"belief"``: only on constraints whose sign depends on the other player's action
      (``belief_dependent_rows``).
    * a boolean array with one entry per constraint, for example ``uncertain_belief_rows``.

    ``extra`` is an optional list of (a, b) pairs adding constraints a . x <= b.
    Returns a dict with the table ``x`` (None if infeasible), the objective and the status.
    """
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    objective = np.asarray(objective, dtype=float)
    A, P, info = obedience_system(c1, c2)
    mu = np.broadcast_to(np.asarray(margin, dtype=float), (2,))
    mu_rows = np.array([mu[player] for player, _, _ in info])
    if isinstance(margin_rows, str):
        if margin_rows == "belief":
            mu_rows = mu_rows * belief_dependent_rows(A)
        elif margin_rows != "all":
            raise ValueError("margin_rows must be 'all', 'belief' or a boolean array")
    else:
        mask = np.asarray(margin_rows, dtype=bool)
        if mask.shape != (len(info),):
            raise ValueError("margin_rows mask must have one entry per obedience constraint")
        mu_rows = mu_rows * mask
    xf, status = _solve_lp(objective.ravel(), A + mu_rows[:, None] * P, extra=extra)
    if xf is None:
        return {"x": None, "objective": np.inf, "status": status}
    return {"x": xf.reshape(c1.shape), "objective": float(objective.ravel() @ xf), "status": status}


def solve_public(c1, c2, objective, margin: float = 0.0, extra=None):
    """Public recommendations: both players see the whole recommended joint action.

    A publicly announced joint action is followed only if it is a Nash equilibrium, so the
    coordinator can only randomise over pure Nash equilibria (here: those that hold with slack
    ``margin``).  Returns the same dict as ``solve_private``.
    """
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    objective = np.asarray(objective, dtype=float)
    ok = pure_nash(c1, c2, margin=margin)
    if not ok:
        return {"x": None, "objective": np.inf, "status": "no pure Nash equilibrium with this margin"}
    m2 = c1.shape[1]
    if not extra:
        # a linear objective over a set of profiles is minimised at one profile; ties are broken
        # by the first profile in row-major order so that the result is reproducible
        best = min(ok, key=lambda a: (round(float(objective[a]), 12), a))
        x = point_table(best, c1.shape)
        return {"x": x, "objective": float(objective[best]), "status": "optimal"}
    allowed = {a1 * m2 + a2 for a1, a2 in ok}
    banned = [i for i in range(c1.size) if i not in allowed]
    xf, status = _solve_lp(objective.ravel(), None, extra=extra, zero_idx=banned)
    if xf is None:
        return {"x": None, "objective": np.inf, "status": status}
    return {"x": xf.reshape(c1.shape), "objective": float(objective.ravel() @ xf), "status": status}


def solve_public_hull(c1, c2, objective):
    """Best mixture over all Nash equilibria (pure and mixed) of the game.

    This is what a public signal can achieve in general: after any public message the players
    must be at a Nash equilibrium, so outcomes are mixtures of Nash equilibria.
    Returns a dict with ``x``, ``objective`` and the list of equilibria with their weights.
    """
    objective = np.asarray(objective, dtype=float)
    eqs = support_enumeration(c1, c2)
    tables = [product_table(p, q) for p, q in eqs]
    vals = np.array([float(np.sum(t * objective)) for t in tables])
    best = int(np.argmin(vals))            # a linear objective is minimised at a single equilibrium
    weights = np.zeros(len(eqs))
    weights[best] = 1.0
    return {"x": tables[best], "objective": float(vals[best]), "equilibria": eqs,
            "values": vals, "weights": weights}
