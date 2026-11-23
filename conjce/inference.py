from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

__all__ = ["realised_joint", "exact_realised_joint", "n_histories", "hoeffding_disobedience_bound",
           "STARTS", "MAX_HISTORIES"]

STARTS = ("observed", "uniform")
MAX_HISTORIES = 5_000_000


def _check_start(start):
    if start not in STARTS:
        raise ValueError(f"start must be one of {STARTS}, got {start!r}")


def _default_tie_tol(c1, c2):
    return 1e-9 * max(float(np.max(np.abs(c1))), float(np.max(np.abs(c2))), 1e-300)


def _best_responses(counts1, counts2, c1, c2, tie_tol):
    m1, m2 = counts1.shape[1:]
    tot1 = counts1.sum(axis=2, keepdims=True)
    tot2 = counts2.sum(axis=1, keepdims=True)
    belief1 = np.where(tot1 > 0, counts1 / np.where(tot1 > 0, tot1, 1.0), 1.0 / m2)
    belief2 = np.where(tot2 > 0, counts2 / np.where(tot2 > 0, tot2, 1.0), 1.0 / m1)

    exp1 = np.einsum("nra,nda->nrd", belief1, c1)
    exp2 = np.einsum("nar,nad->nrd", belief2, c2)
    br1 = _best_response(exp1, m1, tie_tol)
    br2 = _best_response(exp2, m2, tie_tol)
    return br1, br2


def _best_response(expected, rec_axis_len, tie_tol):
    best = np.argmin(expected, axis=-1)
    min_val = np.min(expected, axis=-1)
    r_idx = np.arange(rec_axis_len)[None, :]
    own = np.take_along_axis(expected, np.broadcast_to(r_idx, best.shape)[..., None], axis=-1)[..., 0]
    return np.where(own <= min_val + tie_tol, np.broadcast_to(r_idx, best.shape), best)


def _init_counts(cond, n_rep, rng):
    R, K = cond.shape
    out = np.zeros((n_rep, R, K))
    for r in range(R):
        if cond[r].sum() <= 0:
            continue
        draw = rng.choice(K, size=n_rep, p=cond[r] / cond[r].sum())
        out[np.arange(n_rep), r, draw] = 1.0
    return out


def realised_joint(x, c1, c2, k=None, n_rep: int = 1, rng=None, tie_tol=None, start: str = "observed"):
    _check_start(start)
    x = np.asarray(x, dtype=float)
    m1, m2 = x.shape
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    n = max(c1.shape[0] if c1.ndim == 3 else 1, c2.shape[0] if c2.ndim == 3 else 1,
            n_rep if k is not None else 1)
    c1 = np.broadcast_to(c1, (n, m1, m2))
    c2 = np.broadcast_to(c2, (n, m1, m2))
    if tie_tol is None:
        tie_tol = _default_tie_tol(c1, c2)

    pi1 = x.sum(axis=1)
    pi2 = x.sum(axis=0)
    if k is None:
        counts1 = np.broadcast_to(x, (n, m1, m2)).copy()
        counts2 = counts1.copy()
    elif start == "observed":
        if k < 1:
            raise ValueError("k must be at least 1 (one initial observation per recommendation)")
        rng = np.random.default_rng() if rng is None else rng
        shared = rng.multinomial(int(k) - 1, x.ravel() / x.sum(), size=n).reshape(n, m1, m2).astype(float)
        cond1 = x
        cond2 = x.T
        counts1 = shared + _init_counts(cond1, n, rng)
        counts2 = shared + np.transpose(_init_counts(cond2, n, rng), (0, 2, 1))
    else:
        if k < 0:
            raise ValueError("k must be non-negative")
        rng = np.random.default_rng() if rng is None else rng
        counts1 = rng.multinomial(int(k), x.ravel() / x.sum(), size=n).reshape(n, m1, m2).astype(float)
        counts2 = counts1

    br1, br2 = _best_responses(counts1, counts2, c1, c2, tie_tol)

    y = np.zeros((n, m1, m2))
    rows = np.arange(n)
    obey = np.zeros(n)
    for r1 in range(m1):
        for r2 in range(m2):
            if x[r1, r2] <= 0:
                continue
            np.add.at(y, (rows, br1[:, r1], br2[:, r2]), x[r1, r2])
            obey += x[r1, r2] * ((br1[:, r1] == r1) & (br2[:, r2] == r2))
    obey_1 = (pi1[None, :] * (br1 == np.arange(m1)[None, :])).sum(axis=1)
    obey_2 = (pi2[None, :] * (br2 == np.arange(m2)[None, :])).sum(axis=1)
    return {"y": y, "obey": obey, "obey_1": obey_1, "obey_2": obey_2}


def _composition_blocks(n: int, s: int):
    if s == 1:
        yield np.array([[n]], dtype=np.int64)
    elif s == 2:
        a = np.arange(n + 1, dtype=np.int64)
        yield np.stack([a, n - a], axis=1)
    elif s == 3:
        i, j = np.meshgrid(np.arange(n + 1, dtype=np.int64), np.arange(n + 1, dtype=np.int64), indexing="ij")
        ok = i + j <= n
        i, j = i[ok], j[ok]
        yield np.stack([i, j, n - i - j], axis=1)
    else:
        for first in range(n + 1):
            for block in _composition_blocks(n - first, s - 1):
                yield np.column_stack([np.full(len(block), first, dtype=np.int64), block])


def _chunks(blocks, size):
    buf, filled = [], 0
    for b in blocks:
        buf.append(b)
        filled += len(b)
        if filled >= size:
            yield np.concatenate(buf)
            buf, filled = [], 0
    if buf:
        yield np.concatenate(buf)


def n_histories(x, k, start: str = "observed") -> int:
    _check_start(start)
    s = int(np.count_nonzero(np.asarray(x, dtype=float) > 0))
    n_events = int(k) - 1 if start == "observed" else int(k)
    return math.comb(n_events + s - 1, s - 1)


def exact_realised_joint(x, c1, c2, k: int, start: str = "observed", tie_tol=None,
                         max_histories: int = MAX_HISTORIES, chunk: int = 200_000) -> dict:
    _check_start(start)
    x = np.asarray(x, dtype=float)
    m1, m2 = x.shape
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    if c1.shape != (m1, m2) or c2.shape != (m1, m2):
        raise ValueError("exact_realised_joint takes one cost table per operator, shaped like x")
    if tie_tol is None:
        tie_tol = _default_tie_tol(c1, c2)
    k = int(k)
    if start == "observed" and k < 1:
        raise ValueError("k must be at least 1 (one initial observation per recommendation)")
    if k < 0:
        raise ValueError("k must be non-negative")
    n_hist = n_histories(x, k, start)
    if n_hist > max_histories:
        raise ValueError(f"exact enumeration needs {n_hist} count vectors (support of "
                         f"{np.count_nonzero(x > 0)} cells, k = {k}); the limit is {max_histories}")

    cells = np.argwhere(x > 0)
    s = len(cells)
    logp = np.log(x[cells[:, 0], cells[:, 1]] / x.sum())
    n_events = k - 1 if start == "observed" else k
    pi1 = x.sum(axis=1)
    pi2 = x.sum(axis=0)

    passes = []
    if start == "observed":
        def options(table):
            init, wt = [], []
            sup = [np.flatnonzero(table[r] > 0) for r in range(table.shape[0])]
            for j in range(max(len(a) for a in sup)):
                one = np.zeros_like(table)
                w = np.zeros(table.shape[0])
                for r, a in enumerate(sup):
                    if j < len(a):
                        one[r, a[j]] = 1.0
                        w[r] = table[r, a[j]] / table[r].sum()
                    elif j == 0:
                        w[r] = 1.0
                init.append(one)
                wt.append(w)
            return init, wt
        init1, wt1 = options(x)
        init2, wt2 = options(x.T)
        for j in range(max(len(init1), len(init2))):
            passes.append((init1[j] if j < len(init1) else np.zeros((m1, m2)),
                           wt1[j] if j < len(wt1) else np.zeros(m1),
                           (init2[j] if j < len(init2) else np.zeros((m2, m1))).T,
                           wt2[j] if j < len(wt2) else np.zeros(m2)))
    else:
        passes.append((np.zeros((m1, m2)), np.ones(m1), np.zeros((m1, m2)), np.ones(m2)))

    y = np.zeros((m1, m2))
    obey = obey_1 = obey_2 = total = 0.0
    lgk = gammaln(n_events + 1)
    for comp in _chunks(_composition_blocks(n_events, s), chunk):
        h = len(comp)
        w = np.exp(lgk - gammaln(comp + 1).sum(axis=1) + (comp * logp).sum(axis=1))
        shared = np.zeros((h, m1, m2))
        shared[:, cells[:, 0], cells[:, 1]] = comp
        c1b = np.broadcast_to(c1, (h, m1, m2))
        c2b = np.broadcast_to(c2, (h, m1, m2))
        d1 = np.zeros((h, m1, m1))
        d2 = np.zeros((h, m2, m2))
        hh = np.arange(h)[:, None]
        for init1_j, w1_j, init2_j, w2_j in passes:
            br1, br2 = _best_responses(shared + init1_j, shared + init2_j, c1b, c2b, tie_tol)
            d1[hh, np.arange(m1)[None, :], br1] += w1_j[None, :]
            d2[hh, np.arange(m2)[None, :], br2] += w2_j[None, :]
        t = np.einsum("hia,ij->hja", d1, x)
        y += np.einsum("h,hja,hjb->ab", w, t, d2)
        f1 = np.einsum("hii->hi", d1)
        f2 = np.einsum("hjj->hj", d2)
        obey += float(np.einsum("h,hi,ij,hj->", w, f1, x, f2))
        obey_1 += float(w @ (f1 @ pi1))
        obey_2 += float(w @ (f2 @ pi2))
        total += float(w.sum())
    return {"y": y, "obey": obey, "obey_1": obey_1, "obey_2": obey_2,
            "n_histories": n_hist, "total_probability": total}


def hoeffding_disobedience_bound(x, c1, c2, k: int) -> float:
    x = np.asarray(x, dtype=float)
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    total = 0.0
    for player, table, cost in ((0, x, c1), (1, x.T, c2.T)):
        for r in range(table.shape[0]):
            pi_r = table[r].sum()
            if pi_r <= 0:
                continue
            support = table[r] > 0
            cond = table[r] / pi_r
            for d in range(cost.shape[0]):
                if d == r:
                    continue
                g = cost[d] - cost[r]
                gamma = float(cond @ g)
                width = float(g[support].max() - g[support].min())
                if g[support].min() >= 0:
                    continue
                if gamma <= 0:
                    total += pi_r
                    continue
                s = 2.0 * gamma**2 / width**2
                total += pi_r * np.exp(-s) * (1.0 - pi_r * (1.0 - np.exp(-s))) ** (k - 1)
    return float(min(1.0, total))
