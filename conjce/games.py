from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np

__all__ = [
    "pure_nash", "support_enumeration", "nash_regret", "obedience_system", "obedience_slack",
    "conditional_advantages", "belief_dependent_rows", "uncertain_belief_rows", "reveals_other",
    "solve_private", "solve_public", "solve_public_hull", "product_table", "point_table",
]


def pure_nash(c1, c2, margin: float = 0.0, tol: float = 1e-12):
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
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    m1, m2 = c1.shape
    scale = max(np.max(np.abs(c1)), np.max(np.abs(c2)), 1e-300)
    found = []
    for size in range(1, min(m1, m2) + 1):
        for I in itertools.combinations(range(m1), size):
            for J in itertools.combinations(range(m2), size):
                rq = _indifference(c1[np.ix_(I, J)])
                rp = _indifference(c2[np.ix_(I, J)].T)
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
                    continue
                if not any(np.allclose(p, p0, atol=1e-7) and np.allclose(q, q0, atol=1e-7) for p0, q0 in found):
                    found.append((p, q))
    return found


def nash_regret(p, q, c1, c2):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    return float(p @ c1 @ q - np.min(c1 @ q)), float(p @ c2 @ q - np.min(p @ c2))


def product_table(p, q) -> np.ndarray:
    return np.outer(np.asarray(p, dtype=float), np.asarray(q, dtype=float))


def point_table(profile, shape=(3, 3)) -> np.ndarray:
    x = np.zeros(shape)
    x[tuple(int(a) for a in profile)] = 1.0
    return x


def obedience_system(c1, c2):
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
    A, _, _ = obedience_system(c1, c2)
    return A @ np.asarray(x, dtype=float).ravel()


def conditional_advantages(x, c1, c2, min_prob: float = 1e-12):
    A, P, info = obedience_system(c1, c2)
    xf = np.asarray(x, dtype=float).ravel()
    out = []
    for a_row, p_row, (player, r, d) in zip(A, P, info):
        prob = float(p_row @ xf)
        if prob > min_prob:
            out.append({"player": player, "rec": r, "dev": d, "prob": prob,
                        "advantage": float(-(a_row @ xf) / prob)})
    return out


def _solve_lp(obj, A_ub, extra=None, zero_idx=None):
    n = obj.size
    x = cp.Variable(n, nonneg=True)
    cons = [cp.sum(x) == 1]
    if A_ub is not None and len(A_ub):
        scale = np.max(np.abs(A_ub), axis=1, keepdims=True)
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
    A = np.asarray(A, dtype=float)
    return (A.min(axis=1) < 0) & (A.max(axis=1) > 0)


def uncertain_belief_rows(x0, c1, c2, tol: float = 1e-9) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    A, _, info = obedience_system(c1, c2)
    dependent = belief_dependent_rows(A)
    out = np.zeros(len(info), dtype=bool)
    for j, (player, r, _) in enumerate(info):
        given_r = x0[r, :] if player == 0 else x0[:, r]
        out[j] = dependent[j] and np.count_nonzero(given_r > tol) > 1
    return out


def reveals_other(x, tol: float = 1e-9) -> bool:
    x = np.asarray(x, dtype=float)
    rows_ok = all(np.count_nonzero(x[r, :] > tol) <= 1 for r in range(x.shape[0]))
    cols_ok = all(np.count_nonzero(x[:, r] > tol) <= 1 for r in range(x.shape[1]))
    return rows_ok and cols_ok


def solve_private(c1, c2, objective, margin=0.0, extra=None, margin_rows="all"):
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
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    objective = np.asarray(objective, dtype=float)
    ok = pure_nash(c1, c2, margin=margin)
    if not ok:
        return {"x": None, "objective": np.inf, "status": "no pure Nash equilibrium with this margin"}
    m2 = c1.shape[1]
    if not extra:
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
    objective = np.asarray(objective, dtype=float)
    eqs = support_enumeration(c1, c2)
    tables = [product_table(p, q) for p, q in eqs]
    vals = np.array([float(np.sum(t * objective)) for t in tables])
    best = int(np.argmin(vals))
    weights = np.zeros(len(eqs))
    weights[best] = 1.0
    return {"x": tables[best], "objective": float(vals[best]), "equilibria": eqs,
            "values": vals, "weights": weights}
