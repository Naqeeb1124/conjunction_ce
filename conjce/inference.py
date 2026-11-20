"""Operators who do not know the recommendation rule and learn it from past events.

Setting (the two-receiver analogue of the inference setting in Karabag et al., arXiv:2310.00468,
and Probine, Karabag, Topcu, arXiv:2510.01434):

* The coordinator commits to a fixed table x over joint recommendations and uses it for every event.
* Before the event of interest each operator has seen k past events.  After each past event the
  joint recommendation is disclosed, so operator i knows, for every recommendation r it received,
  how often the other operator was told each action.
* Operator i's belief about the other's action, given its own recommendation r, is the empirical
  frequency over those past events.  As in the two papers, one extra observation drawn from the
  true conditional distribution initialises the counts, so the belief is defined for every r:
  the counts are (k - 1) draws from x plus that one draw.
* Each operator then best-responds to its belief with its own true costs, and follows the
  recommendation when it is among the best responses.

That initialisation is a modelling convention.  ``start="uniform"`` gives the alternative: the
operator has seen exactly k events drawn from x and nothing else, and a recommendation that never
occurred in those k events leaves it with a uniform belief over the other operator's actions.

``exact_realised_joint`` computes the expectations that ``realised_joint`` estimates, without
sampling, by enumerating the multinomial counts of the past events.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

__all__ = ["realised_joint", "exact_realised_joint", "n_histories", "hoeffding_disobedience_bound",
           "STARTS", "MAX_HISTORIES"]

STARTS = ("observed", "uniform")
MAX_HISTORIES = 5_000_000      # largest number of count vectors exact_realised_joint will enumerate


def _check_start(start):
    if start not in STARTS:
        raise ValueError(f"start must be one of {STARTS}, got {start!r}")


def _default_tie_tol(c1, c2):
    return 1e-9 * max(float(np.max(np.abs(c1))), float(np.max(np.abs(c2))), 1e-300)


def _best_responses(counts1, counts2, c1, c2, tie_tol):
    """Each operator's action for every recommendation it can receive, given its counts.

    counts1[n, r1, a2]: how often operator 1 was told r1 while operator 2 was told a2;
    counts2[n, a1, r2]: the same for operator 2 (its recommendation is the column).
    A recommendation with no counts gives a uniform belief.  c1, c2 have shape (n, m1, m2).
    Returns br1 (n, m1) and br2 (n, m2)."""
    m1, m2 = counts1.shape[1:]
    # beliefs: operator 1 about a2 given r1 (normalise rows); operator 2 about a1 given r2 (columns)
    tot1 = counts1.sum(axis=2, keepdims=True)
    tot2 = counts2.sum(axis=1, keepdims=True)
    belief1 = np.where(tot1 > 0, counts1 / np.where(tot1 > 0, tot1, 1.0), 1.0 / m2)
    belief2 = np.where(tot2 > 0, counts2 / np.where(tot2 > 0, tot2, 1.0), 1.0 / m1)

    # expected cost of each own action d, given the recommendation
    exp1 = np.einsum("nra,nda->nrd", belief1, c1)          # (n, r1, d1)
    exp2 = np.einsum("nar,nad->nrd", belief2, c2)          # (n, r2, d2)
    br1 = _best_response(exp1, m1, tie_tol)                # (n, m1)
    br2 = _best_response(exp2, m2, tie_tol)                # (n, m2)
    return br1, br2


def _best_response(expected, rec_axis_len, tie_tol):
    """expected[n, r, d]: expected cost of playing d when told r.  Obey when r is within tie_tol
    of the minimum, otherwise play the minimiser."""
    best = np.argmin(expected, axis=-1)                          # (n, r)
    min_val = np.min(expected, axis=-1)
    r_idx = np.arange(rec_axis_len)[None, :]
    own = np.take_along_axis(expected, np.broadcast_to(r_idx, best.shape)[..., None], axis=-1)[..., 0]
    return np.where(own <= min_val + tie_tol, np.broadcast_to(r_idx, best.shape), best)


def _init_counts(cond, n_rep, rng):
    """One draw per recommendation from each row of ``cond`` (rows that sum to zero give no draw).
    cond: (R, K) conditional probabilities.  Returns one-hot counts of shape (n_rep, R, K)."""
    R, K = cond.shape
    out = np.zeros((n_rep, R, K))
    for r in range(R):
        if cond[r].sum() <= 0:
            continue
        draw = rng.choice(K, size=n_rep, p=cond[r] / cond[r].sum())
        out[np.arange(n_rep), r, draw] = 1.0
    return out


def realised_joint(x, c1, c2, k=None, n_rep: int = 1, rng=None, tie_tol=None, start: str = "observed"):
    """Joint actions that actually result when operators best-respond to their beliefs.

    Parameters
    ----------
    x : (m1, m2) recommendation table.
    c1, c2 : cost tables, shape (m1, m2) or (n_rep, m1, m2) for one true game per replicate.
    k : None for operators who know x exactly, or the number of past events they learned from.
    n_rep : number of independent learning histories (ignored when k is None and costs are fixed).
    start : how the counts begin (ignored when k is None).
      "observed": (k - 1) shared events plus, for every recommendation with positive probability,
                  one observation of the other's recommendation drawn from the true conditional
                  (k >= 1).  This is the convention used for every stored result.
      "uniform":  exactly k shared events and nothing else (k >= 0); a recommendation never seen
                  in them gives a uniform belief over the other operator's actions.

    Returns
    -------
    dict with
      ``y``    (n, m1, m2): realised joint-action distribution in each replicate (the expectation
               over the new event's recommendation is taken exactly);
      ``obey`` (n,): probability, in each replicate, that both operators follow the recommendation;
      ``obey_1``, ``obey_2`` (n,): the same for each operator alone.
    """
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

    pi1 = x.sum(axis=1)                                   # probability operator 1 is told r1
    pi2 = x.sum(axis=0)
    if k is None:
        counts1 = np.broadcast_to(x, (n, m1, m2)).copy()  # exact conditionals
        counts2 = counts1.copy()
    elif start == "observed":
        if k < 1:
            raise ValueError("k must be at least 1 (one initial observation per recommendation)")
        rng = np.random.default_rng() if rng is None else rng
        shared = rng.multinomial(int(k) - 1, x.ravel() / x.sum(), size=n).reshape(n, m1, m2).astype(float)
        cond1 = x                                           # rows: operator 1's recommendation
        cond2 = x.T                                         # rows: operator 2's recommendation
        counts1 = shared + _init_counts(cond1, n, rng)
        counts2 = shared + np.transpose(_init_counts(cond2, n, rng), (0, 2, 1))
    else:                                                   # start == "uniform"
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
    """All vectors of s non-negative integers summing to n, as a sequence of (h, s) int arrays."""
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
    """Number of count vectors exact_realised_joint enumerates for table x and k past events."""
    _check_start(start)
    s = int(np.count_nonzero(np.asarray(x, dtype=float) > 0))
    n_events = int(k) - 1 if start == "observed" else int(k)
    return math.comb(n_events + s - 1, s - 1)


def exact_realised_joint(x, c1, c2, k: int, start: str = "observed", tie_tol=None,
                         max_histories: int = MAX_HISTORIES, chunk: int = 200_000) -> dict:
    """The expectations that ``realised_joint`` estimates, computed without sampling.

    The shared past events are multinomial over the support cells of x (k - 1 events for
    start="observed", k for start="uniform"); every count vector is enumerated with its
    multinomial probability.  For start="observed" the initial observation of each operator and
    recommendation is independent of everything else and affects only that recommendation's
    belief, so each operator's response to recommendation r is averaged over its (at most m)
    possible initial observations, and the two operators' responses are independent given the
    shared counts.  Beliefs and best responses use the same code as ``realised_joint``.

    c1, c2 are single (m1, m2) cost tables.  Raises ValueError if more than ``max_histories``
    count vectors would be needed (see ``n_histories``).

    Returns a dict with ``y`` (m1, m2): the expected realised joint-action distribution;
    ``obey``, ``obey_1``, ``obey_2``: the probabilities that both operators, operator 1, operator 2
    follow the recommendation in the new event; ``n_histories``; ``total_probability`` (the sum of
    the enumerated history probabilities, 1 up to rounding).
    """
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

    # initial observations: pass j adds, for every recommendation r, the j-th possible observation
    # (weight = its conditional probability; 0 when r has fewer than j + 1 possible observations)
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
                        w[r] = 1.0              # recommendation never issued: no draw, weight 1
                init.append(one)
                wt.append(w)
            return init, wt
        init1, wt1 = options(x)                 # rows: operator 1's recommendation
        init2, wt2 = options(x.T)               # rows: operator 2's recommendation
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
        d1 = np.zeros((h, m1, m1))              # d1[h, r, d] = P(operator 1 plays d when told r)
        d2 = np.zeros((h, m2, m2))
        hh = np.arange(h)[:, None]
        for init1_j, w1_j, init2_j, w2_j in passes:
            br1, br2 = _best_responses(shared + init1_j, shared + init2_j, c1b, c2b, tie_tol)
            d1[hh, np.arange(m1)[None, :], br1] += w1_j[None, :]
            d2[hh, np.arange(m2)[None, :], br2] += w2_j[None, :]
        # new event: recommendation (r1, r2) ~ x, independent of the past
        t = np.einsum("hia,ij->hja", d1, x)                 # t[h, r2, a1]
        y += np.einsum("h,hja,hjb->ab", w, t, d2)
        f1 = np.einsum("hii->hi", d1)                       # P(operator 1 follows | told r)
        f2 = np.einsum("hjj->hj", d2)
        obey += float(np.einsum("h,hi,ij,hj->", w, f1, x, f2))
        obey_1 += float(w @ (f1 @ pi1))
        obey_2 += float(w @ (f2 @ pi2))
        total += float(w.sum())
    return {"y": y, "obey": obey, "obey_1": obey_1, "obey_2": obey_2,
            "n_histories": n_hist, "total_probability": total}


def hoeffding_disobedience_bound(x, c1, c2, k: int) -> float:
    """Upper bound on the probability that at least one operator does not follow its
    recommendation, when both learn x from k past events.

    For an operator told r, with deviation d, let g(a) = c(d, a) - c(r, a) be what obeying saves
    when the other operator plays a.  Its true conditional mean is the advantage gamma > 0 and it
    ranges over a width W on the conditional support.  With n observations of "told r", Hoeffding
    gives P(estimated saving < 0) <= exp(-2 n gamma^2 / W^2).  Here n = 1 + Binomial(k - 1, pi_r),
    and averaging over n gives  e^{-s} * (1 - pi_r * (1 - e^{-s}))^(k - 1)  with s = 2 gamma^2 / W^2.
    The bounds are summed over deviations and recommendations (union bound) and capped at 1.
    """
    x = np.asarray(x, dtype=float)
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    total = 0.0
    for player, table, cost in ((0, x, c1), (1, x.T, c2.T)):
        # table[r, a]: joint probability that this player is told r and the other is told a
        # cost[d, a]: this player's cost of playing d when the other plays a
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
                if g[support].min() >= 0:        # obeying is at least as good whatever is observed
                    continue
                if gamma <= 0:
                    total += pi_r
                    continue
                s = 2.0 * gamma**2 / width**2
                total += pi_r * np.exp(-s) * (1.0 - pi_r * (1.0 - np.exp(-s))) ** (k - 1)
    return float(min(1.0, total))
