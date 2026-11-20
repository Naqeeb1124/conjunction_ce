"""Many conjunctions at once: a 72-hour batch among 30 satellites of 4 operators.

Every parameter below was fixed before anything was run.  Words used here:

* A conjunction is one pair of satellites of two different operators.  Its game is one draw of
  ``experiments.random_instance``; side 1 plays the role of operator 1 of that game, side 2 the
  role of operator 2.  A slot is one side of one conjunction and belongs to that satellite's
  operator.
* A cell is the pair of actions in one conjunction, side 1 first.  Exact-one cells: SN, LN, NS, NL
  (exactly one satellite burns).
* A stable designation is an exact-one cell that is a pure Nash equilibrium of the conjunction's
  game; ``E_k`` is the list of stable designations of conjunction k.
* A free conjunction has one stable designation for each side, both with the same coordinator cost.

The batch game: the players are the operators; operator p's cost is the sum over its slots of that
conjunction's cost table; the coordinator's objective is the sum of the conjunctions' coordinator
cost tables.  Each operator is told the recommendations for all its slots at once.

Formulations (all solved with ``scipy.optimize.linprog(method="highs")``):

* (a) ``joint_program(conjs, "full")``: all 9^K joint profiles;
* (b) ``joint_program(conjs, "exact_one")``: the 4^K profiles in which every conjunction is in an
  exact-one cell;
* (c) ``per_conjunction_program(conjs)``: one stacked program with 9K variables;
* (c') ``closed_form(conjs)``: the sum over k of the cheapest element of E_k.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from . import experiments as E
from . import games as G
from . import model as M

# ----------------------------------------------------------------------------- fixed parameters
FLEETS = ((0, 12), (12, 20), (20, 26), (26, 30))          # operator p owns satellites lo..hi-1
N_SATELLITES = 30
N_OPERATORS = 4
OPERATOR_OF = np.concatenate([np.full(hi - lo, p) for p, (lo, hi) in enumerate(FLEETS)])
WINDOW_HOURS = 72.0
MASTER_SEED = 72
MASTER_SIZE = 50
K_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50)
SIZE_CAP = 100_000                                        # largest joint program that is built
REPEATS = 3                                               # timing: smallest of this many
AGREEMENT_TOL = 1e-6                                      # relative, between optimal values
FEASIBILITY_TOL = 1e-9                                    # on rows scaled to max |coefficient| = 1
FREE_TOL = 1e-9                                           # relative cost difference of a free conjunction
FURTHER_SEEDS = tuple(range(1000, 1200))
BURDEN_KS = (20, 50)
P1_KS = (2, 3)
P5_KS = (6, 8, 10)
BRUTE_FORCE_MAX_FREE = 12

# all satellite pairs of different operators, in lexicographic order
PAIRS = tuple((i, j) for i in range(N_SATELLITES) for j in range(i + 1, N_SATELLITES)
              if OPERATOR_OF[i] != OPERATOR_OF[j])

LABEL = M.ACTION_LABELS                                   # N, S, L
EXACT_ONE = ((1, 0), (2, 0), (0, 1), (0, 2))              # SN, LN, NS, NL
FULL = tuple((a1, a2) for a1 in range(3) for a2 in range(3))   # row-major, as table.ravel()
DEV = np.array([[1, 2], [0, 2], [0, 1]])                  # the two other actions, per own action
HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def cell_label(cell) -> str:
    return LABEL[cell[0]] + LABEL[cell[1]]


class CheckFailed(RuntimeError):
    """A check did not hold; the run writes no results file."""


# ----------------------------------------------------------------------------- the batch
@dataclass
class Conjunction:
    index: int                    # position in generation order
    sat1: int                     # satellite on side 1
    sat2: int                     # satellite on side 2
    time_h: float                 # time of closest approach, hours into the window
    scn: M.Scenario
    c1: np.ndarray                # side 1's cost table [a1, a2]
    c2: np.ndarray                # side 2's cost table [a1, a2]
    obj: np.ndarray               # coordinator cost table [a1, a2]
    stable: tuple                 # E_k: stable designations, in the order SN, LN, NS, NL
    n_draws: int                  # instance draws used for this conjunction (rejected ones included)

    @property
    def op1(self) -> int:
        return int(OPERATOR_OF[self.sat1])

    @property
    def op2(self) -> int:
        return int(OPERATOR_OF[self.sat2])

    @property
    def sats(self) -> tuple:
        return (self.sat1, self.sat2)


@dataclass
class Batch:
    seed: int
    conjunctions: list
    rejected: list = field(default_factory=list)   # (conjunction index, Scenario) of every rejected draw

    @property
    def n_draws(self) -> int:
        return sum(c.n_draws for c in self.conjunctions)


def stable_designations(c1, c2) -> tuple:
    """Exact-one cells that are pure Nash equilibria of the conjunction's game (``games.pure_nash``)."""
    nash = set(G.pure_nash(c1, c2))
    return tuple(cell for cell in EXACT_ONE if cell in nash)


def make_conjunction(index, sat1, sat2, time_h, scn, n_draws=1) -> Conjunction:
    c1, c2 = M.cost_tables(scn)
    return Conjunction(index=index, sat1=int(sat1), sat2=int(sat2), time_h=float(time_h), scn=scn,
                       c1=c1, c2=c2, obj=M.coordinator_cost_table(scn),
                       stable=stable_designations(c1, c2), n_draws=n_draws)


def generate_batch(seed: int, n: int = MASTER_SIZE) -> Batch:
    """Draw a batch of n conjunctions with one generator seeded with ``seed``.

    For each conjunction in turn: a pair uniformly among the pairs not yet used, a fair coin for
    which satellite is side 1, a time uniform on the window, then instances of the family of
    ``experiments.random_instance`` until one has at least one stable designation.  Because the
    draws are made conjunction by conjunction, the first K conjunctions of a batch of n are the
    batch of K drawn with the same seed.
    """
    rng = np.random.default_rng(seed)
    remaining = list(range(len(PAIRS)))
    conjs, rejected = [], []
    for k in range(n):
        i, j = PAIRS[remaining.pop(int(rng.integers(len(remaining))))]
        if int(rng.integers(2)) == 1:
            i, j = j, i
        t = float(rng.uniform(0.0, WINDOW_HOURS))
        draws = 0
        while True:
            scn = E.random_instance(rng)
            draws += 1
            c1, c2 = M.cost_tables(scn)
            if stable_designations(c1, c2):
                break
            rejected.append((k, scn))
        conjs.append(make_conjunction(k, i, j, t, scn, n_draws=draws))
    return Batch(seed=seed, conjunctions=conjs, rejected=rejected)


def slots_of(conjs) -> list:
    """For each operator, its slots as (conjunction position, side), side 0 = side 1."""
    out = [[] for _ in range(N_OPERATORS)]
    for k, c in enumerate(conjs):
        out[c.op1].append((k, 0))
        out[c.op2].append((k, 1))
    return out


def _side_cost(conj, side) -> np.ndarray:
    """Cost table of one side indexed [own action, other side's action]."""
    return conj.c1 if side == 0 else conj.c2.T


# ----------------------------------------------------------------------------- linear programs
def _scale_rows(A):
    """Divide every row by its largest absolute coefficient, as ``games.solve_private`` does."""
    A = sparse.csr_matrix(A)
    scale = np.asarray(abs(A).max(axis=1).todense()).ravel()
    scale[scale == 0] = 1.0
    return sparse.diags(1.0 / scale) @ A


def _clean(x) -> np.ndarray:
    """Same clean-up as ``games._solve_lp``: clip, zero entries below 1e-12, renormalise."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    x[x < 1e-12] = 0.0
    return x / x.sum()


def _lp(c, A_ub, A_eq, b_eq):
    """min c @ x  s.t.  A_ub x <= 0,  A_eq x = b_eq,  x >= 0 (HiGHS)."""
    return linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=b_eq,
                   bounds=(0, None), method="highs", options=HIGHS_OPTIONS)


def profiles(K: int, cells) -> np.ndarray:
    """All joint profiles as an integer array (N, K) of cell indices, conjunction 0 most significant
    (so a full-joint solution reshaped to (9,) * K has one axis per conjunction)."""
    m = len(cells)
    return np.indices((m,) * K, dtype=np.int64).reshape(K, -1).T


def joint_rows(conjs, cells, multi: bool = False):
    """Obedience rows of the batch game over the profiles built from ``cells``, unscaled.

    One row per (operator p, tuple r of recommendations to p's slots, slot s of p, other action d)
    when ``multi`` is False; with ``multi`` True one row per (p, r, alternative tuple d != r), the
    deviations in several slots at once.  Rows that no profile reaches are dropped.
    Returns (A as csr, objective vector, profile array).
    """
    K = len(conjs)
    cells = np.asarray(cells, dtype=np.int64)
    prof = profiles(K, cells)
    act = cells[prof]                                     # (N, K, 2): actions of side 1 and side 2
    N = prof.shape[0]
    cols = np.arange(N)
    rows, cs, vals = [], [], []
    off = 0
    for sl in slots_of(conjs):
        n = len(sl)
        if n == 0:
            continue
        own = np.stack([act[:, k, side] for k, side in sl], axis=1)
        oth = np.stack([act[:, k, 1 - side] for k, side in sl], axis=1)
        weights = 3 ** np.arange(n)
        tup = own @ weights
        tables = [_side_cost(conjs[k], side) for k, side in sl]
        if not multi:
            for j, C in enumerate(tables):
                base = C[own[:, j], oth[:, j]]
                for t in (0, 1):
                    d = DEV[own[:, j], t]
                    rows.append(off + (tup * n + j) * 2 + t)
                    cs.append(cols)
                    vals.append(base - C[d, oth[:, j]])
            off += 3 ** n * n * 2
        else:
            D = np.indices((3,) * n, dtype=np.int64).reshape(n, -1).T     # all tuples (M, n)
            d_idx = D @ weights
            Mn = D.shape[0]
            gain = np.zeros((N, Mn))
            for j, C in enumerate(tables):
                gain += C[own[:, j], oth[:, j]][:, None] - C[D[None, :, j], oth[:, j][:, None]]
            keep = d_idx[None, :] != tup[:, None]
            r = off + tup[:, None] * Mn + d_idx[None, :]
            rows.append(r[keep])
            cs.append(np.broadcast_to(cols[:, None], (N, Mn))[keep])
            vals.append(gain[keep])
            off += Mn * Mn
    rows = np.concatenate(rows)
    used, inv = np.unique(rows, return_inverse=True)
    A = sparse.csr_matrix((np.concatenate(vals), (inv.ravel(), np.concatenate(cs))), shape=(used.size, N))
    obj = np.zeros(N)
    for k, c in enumerate(conjs):
        obj += c.obj[act[:, k, 0], act[:, k, 1]]
    return A, obj, prof


def _summary(res, c, A_ub, n_eq):
    ok = res.status == 0
    x = _clean(res.x) if ok else None
    return {"x": x, "optimum": float(c @ x) if ok else np.inf, "status": int(res.status),
            "variables": int(c.size), "rows": int(A_ub.shape[0] + n_eq),
            "nonzeros": int(A_ub.nnz + c.size)}


def joint_program(conjs, kind: str = "full", multi: bool = False) -> dict:
    """(a) ``kind="full"``: all 9^K profiles.  (b) ``kind="exact_one"``: the 4^K exact-one profiles.

    Returns a dict with the cleaned optimum ``x`` (None if infeasible), ``optimum``, the HiGHS
    ``status`` (0 optimal, 2 infeasible), the sizes (variables, rows with the simplex row,
    nonzeros with the simplex row), the scaled rows ``A`` and the ``profiles``.
    """
    cells = FULL if kind == "full" else EXACT_ONE
    A, c, prof = joint_rows(conjs, cells, multi=multi)
    A = _scale_rows(A)
    A_eq = sparse.csr_matrix(np.ones((1, c.size)))
    out = _summary(_lp(c, A, A_eq, np.ones(1)), c, A, 1)
    out.update(A=A, profiles=prof, objective=c)
    return out


def per_conjunction_program(conjs) -> dict:
    """(c): one stacked program, a probability table per conjunction with its 12 obedience rows
    (``games.obedience_system``), each row scaled as in ``games.solve_private``."""
    K = len(conjs)
    blocks = [_scale_rows(G.obedience_system(c.c1, c.c2)[0]) for c in conjs]
    A = sparse.block_diag(blocks, format="csr")
    A_eq = sparse.kron(sparse.eye(K), np.ones((1, 9)), format="csr")
    c = np.concatenate([cj.obj.ravel() for cj in conjs])
    res = _lp(c, A, A_eq, np.ones(K))
    out = _summary(res, c, A, K)
    out["nonzeros"] = int(A.nnz + A_eq.nnz)
    if res.status == 0:
        tables = [_clean(res.x[9 * k:9 * k + 9]) for k in range(K)]
        out["x"] = np.concatenate(tables)
        out["tables"] = [t.reshape(3, 3) for t in tables]
        out["optimum"] = float(sum(float(cj.obj.ravel() @ t) for cj, t in zip(conjs, tables)))
    return out


def closed_form(conjs) -> float:
    """(c'): sum over k of the cheapest stable designation; infinite if some E_k is empty."""
    total = 0.0
    for c in conjs:
        if not c.stable:
            return np.inf
        total += min(float(c.obj[e]) for e in c.stable)
    return total


def timed(fn, *args, repeats: int = REPEATS, **kw):
    """Smallest wall-clock of ``repeats`` calls (build plus solve) and the last result."""
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return out, best


def rel_diff(u: float, v: float) -> float:
    if u == v:
        return 0.0
    return abs(u - v) / max(abs(u), abs(v), 1e-300)


def max_violation(A_scaled, x) -> float:
    """Largest left-hand side of the scaled rows A x <= 0 (0 if none is positive)."""
    if A_scaled.shape[0] == 0:
        return 0.0
    return float(max(0.0, np.max(A_scaled @ x)))


# ----------------------------------------------------------------------------- the checks
def marginals(x_full, K: int) -> list:
    """Per-conjunction 3x3 marginals of a distribution over the 9^K full profiles."""
    t = np.asarray(x_full).reshape((9,) * K)
    return [t.sum(axis=tuple(i for i in range(K) if i != k)).reshape(3, 3) for k in range(K)]


def product_distribution(tables) -> np.ndarray:
    """Product of per-conjunction tables as a distribution over the 9^K full profiles."""
    out = np.ones(1)
    for t in tables:
        out = np.kron(out, np.asarray(t).ravel())
    return out


def stable_mask(conjs, prof) -> np.ndarray:
    """For exact-one profiles: True where every conjunction is in one of its stable designations."""
    ok = np.ones(prof.shape[0], dtype=bool)
    for k, c in enumerate(conjs):
        allowed = np.array([cell in c.stable for cell in EXACT_ONE])
        ok &= allowed[prof[:, k]]
    return ok


def check_p1(conjs) -> dict:
    """P1: the program with every multi-slot deviation has the same optimum, and the single-slot
    optimum violates none of the multi-slot rows."""
    single = joint_program(conjs, "full")
    multi = joint_program(conjs, "full", multi=True)
    return {"K": len(conjs), "max_slots_of_an_operator": max(len(s) for s in slots_of(conjs)),
            "optimum_single_slot_rows": single["optimum"], "optimum_all_deviation_rows": multi["optimum"],
            "rel_diff": rel_diff(single["optimum"], multi["optimum"]),
            "violation_of_all_deviation_rows": max_violation(multi["A"], single["x"]),
            "rows_single": single["rows"], "rows_all_deviations": multi["rows"]}


def check_p2(conjs, full=None, per=None, private_sum=None) -> dict:
    """P2: optimum of (a) = optimum of (c) = sum of ``games.solve_private``; marginals of the (a)
    optimum obey each conjunction's 12 rows; the product of the (c) tables obeys the joint rows."""
    K = len(conjs)
    full = full if full is not None else joint_program(conjs, "full")
    per = per if per is not None else per_conjunction_program(conjs)
    if private_sum is None:
        private_sum = sum(G.solve_private(c.c1, c.c2, c.obj)["objective"] for c in conjs)
    marg_viol = max(max_violation(_scale_rows(G.obedience_system(c.c1, c.c2)[0]), m.ravel())
                    for c, m in zip(conjs, marginals(full["x"], K)))
    prod = product_distribution(per["tables"])
    return {"K": K, "optimum_full_joint": full["optimum"], "optimum_per_conjunction": per["optimum"],
            "sum_solve_private": float(private_sum),
            "rel_diff": max(rel_diff(full["optimum"], per["optimum"]), rel_diff(full["optimum"], private_sum),
                            rel_diff(per["optimum"], private_sum)),
            "marginal_violation": marg_viol,
            "product_violation": max_violation(full["A"], prod),
            "product_objective_rel_diff": rel_diff(float(full["objective"] @ prod), per["optimum"])}


def check_p3(conjs, ex=None) -> dict:
    """P3: optimum of (b) = closed form (c'); the (b) optimum has no mass outside the products of
    stable designations."""
    ex = ex if ex is not None else joint_program(conjs, "exact_one")
    cf = closed_form(conjs)
    mass = float(ex["x"][~stable_mask(conjs, ex["profiles"])].sum()) if ex["x"] is not None else np.nan
    return {"K": len(conjs), "optimum_exact_one_joint": ex["optimum"], "closed_form": cf,
            "rel_diff": rel_diff(ex["optimum"], cf), "mass_outside_stable_products": mass}


def infeasibility_batch(batch: Batch, size: int = 3):
    """A small batch with one conjunction whose E_k is empty: the first ``size`` conjunctions of
    ``batch``, with the game of conjunction min(j, size - 1) replaced by the first rejected draw
    (j = the conjunction at whose generation that draw was rejected).  If the batch had no rejected
    draw, the game is built by hand so that no burn is worth it for either side."""
    conjs = list(batch.conjunctions[:size])
    if batch.rejected:
        j, scn = batch.rejected[0]
        source = f"first rejected draw (while drawing conjunction {j} of seed {batch.seed})"
    else:
        j, scn = size - 1, M.Scenario(miss_xi=2000.0, miss_zeta=2000.0)
        source = "built by hand: miss far outside the uncertainty ellipse, so N is dominant for both sides"
    pos = min(j, size - 1)
    old = conjs[pos]
    conjs[pos] = make_conjunction(old.index, old.sat1, old.sat2, old.time_h, scn)
    return conjs, pos, source


def check_p3_infeasible(batch: Batch, size: int = 3) -> dict:
    conjs, pos, source = infeasibility_batch(batch, size)
    ex = joint_program(conjs, "exact_one")
    return {"K": len(conjs), "position_of_empty_conjunction": pos, "source": source,
            "empty_E_k": len(conjs[pos].stable) == 0, "status": ex["status"],
            "infeasible": ex["status"] == 2, "closed_form": closed_form(conjs)}


# ----------------------------------------------------------------------------- burden-sharing
def designation_info(conj) -> dict:
    """Stage 1 for one conjunction: its cost-optimal stable designations and whether it is free."""
    costs = {e: float(conj.obj[e]) for e in conj.stable}
    best = min(costs.values())
    optimal = tuple(e for e in conj.stable if rel_diff(costs[e], best) <= FREE_TOL)
    side1 = [e for e in conj.stable if e[0] > 0]
    side2 = [e for e in conj.stable if e[1] > 0]
    free = bool(len(side1) == 1 and len(side2) == 1
                and rel_diff(costs[side1[0]], costs[side2[0]]) <= FREE_TOL)
    return {"optimal": optimal, "free": free, "both_sides": bool(side1 and side2), "cost": best}


def fixed_rule_side(conj) -> int:
    """Side (0 or 1) that ``model.fixed_rule_profile`` designates to burn in this conjunction."""
    a = M.fixed_rule_profile(conj.scn)
    return 0 if a[0] > 0 else 1


def burner(cell) -> int:
    return 0 if cell[0] > 0 else 1


def _burden_setup(conjs):
    """Burns already fixed by stage 1, and for each free conjunction its two satellites."""
    base = np.zeros(N_SATELLITES)
    free = []
    for k, c in enumerate(conjs):
        info = designation_info(c)
        if info["free"]:
            free.append(k)
        else:
            base[c.sats[burner(info["optimal"][0])]] += 1
    return base, free


def _incidence(conjs, free):
    """B[i, f] = +1 if satellite i is side 1 of free conjunction f, -1 if side 2; s2[i] = number of
    free conjunctions with i on side 2.  Burns of i with z_f = 1 meaning side 1 burns: s2 + B z."""
    B = np.zeros((N_SATELLITES, len(free)))
    s2 = np.zeros(N_SATELLITES)
    for f, k in enumerate(free):
        B[conjs[k].sat1, f] += 1.0
        B[conjs[k].sat2, f] -= 1.0
        s2[conjs[k].sat2] += 1.0
    return B, s2


def burns_fixed_tie_break(conjs) -> np.ndarray:
    base, free = _burden_setup(conjs)
    for k in free:
        base[conjs[k].sats[fixed_rule_side(conjs[k])]] += 1
    return base


def burns_from_choice(conjs, z) -> np.ndarray:
    """Burns per satellite when free conjunction f (in order) has side 1 burn iff z[f] = 1."""
    base, free = _burden_setup(conjs)
    B, s2 = _incidence(conjs, free)
    return base + s2 + B @ np.asarray(z, dtype=float)


def balanced_randomised(conjs) -> dict:
    """P5 program: (free conjunctions + 1) variables q_f = P(side 1 burns in f) and t;
    minimise t subject to expected burns of every satellite <= t."""
    base, free = _burden_setup(conjs)
    B, s2 = _incidence(conjs, free)
    F = len(free)
    c = np.zeros(F + 1)
    c[-1] = 1.0
    A = np.hstack([B, -np.ones((N_SATELLITES, 1))])
    res = linprog(c, A_ub=A, b_ub=-(base + s2), bounds=[(0, 1)] * F + [(0, None)], method="highs",
                  options=HIGHS_OPTIONS)
    q = res.x[:F]
    return {"max_expected_burns": float(res.x[-1]), "q": q, "expected_burns": base + s2 + B @ q,
            "n_free": F, "variables": F + 1}


def balanced_randomised_joint(conjs) -> dict:
    """The same minimum over all distributions on the 2^F joint assignments of the free conjunctions."""
    base, free = _burden_setup(conjs)
    B, s2 = _incidence(conjs, free)
    F = len(free)
    Z = np.array(list(itertools.product((0, 1), repeat=F)), dtype=float).reshape(-1, F)
    burns = (base + s2)[:, None] + B @ Z.T                     # (satellites, assignments)
    n = Z.shape[0]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A_ub = np.hstack([burns, -np.ones((N_SATELLITES, 1))])
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(N_SATELLITES), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (n + 1), method="highs", options=HIGHS_OPTIONS)
    return {"max_expected_burns": float(res.x[-1]), "variables": n + 1, "n_free": F}


def balanced_deterministic(conjs) -> dict:
    """0/1 choice for every free conjunction minimising the largest number of burns
    (``scipy.optimize.milp``).  The optimal value is unique, the assignment found is one of possibly
    several."""
    base, free = _burden_setup(conjs)
    B, s2 = _incidence(conjs, free)
    F = len(free)
    c = np.zeros(F + 1)
    c[-1] = 1.0
    A = np.hstack([B, -np.ones((N_SATELLITES, 1))])
    res = milp(c, constraints=LinearConstraint(A, -np.inf, -(base + s2)),
               integrality=np.r_[np.ones(F), 0], bounds=Bounds(np.zeros(F + 1), np.r_[np.ones(F), np.inf]))
    z = np.round(res.x[:F])
    burns = base + s2 + B @ z
    return {"max_burns": int(round(burns.max())), "z": z.astype(int), "burns": burns, "n_free": F}


def brute_force_max_burns(conjs) -> int:
    base, free = _burden_setup(conjs)
    B, s2 = _incidence(conjs, free)
    F = len(free)
    Z = np.array(list(itertools.product((0, 1), repeat=F)), dtype=float).reshape(-1, F)
    burns = (base + s2)[:, None] + B @ Z.T
    return int(round(burns.max(axis=0).min()))


def burden(conjs) -> dict:
    """The three stage-2 variants for one batch; stage 1 gives every conjunction a cheapest stable
    designation, so the coordinator cost is the same in all three."""
    fixed = burns_fixed_tie_break(conjs)
    rnd = balanced_randomised(conjs)
    det = balanced_deterministic(conjs)
    out = {"n_conjunctions": len(conjs), "n_free": rnd["n_free"],
           "max_burns_fixed_tie_break": int(round(fixed.max())),
           "max_expected_burns_balanced_randomised": rnd["max_expected_burns"],
           "max_burns_balanced_deterministic": det["max_burns"],
           "burns_fixed_tie_break": fixed.astype(int), "burns_balanced_deterministic": det["burns"].astype(int),
           "expected_burns_balanced_randomised": rnd["expected_burns"]}
    if rnd["n_free"] <= BRUTE_FORCE_MAX_FREE:
        out["brute_force_max_burns"] = brute_force_max_burns(conjs)
    return out


def histogram(burns) -> list:
    """Number of satellites with 0, 1, 2, ... burns."""
    return np.bincount(np.asarray(burns, dtype=int), minlength=1).tolist()


# ----------------------------------------------------------------------------- carry-over
def designated_burners(conjs, variant: str) -> list:
    """Satellite designated to burn in each conjunction (``"fixed"`` or ``"deterministic"`` tie-break)."""
    _, free = _burden_setup(conjs)
    z = balanced_deterministic(conjs)["z"] if variant == "deterministic" else None
    out = []
    for k, c in enumerate(conjs):
        info = designation_info(c)
        if not info["free"]:
            side = burner(info["optimal"][0])
        elif variant == "fixed":
            side = fixed_rule_side(c)
        else:
            side = 0 if z[free.index(k)] == 1 else 1
        out.append(c.sats[side])
    return out


def carry_over(conjs) -> dict:
    """Ordered by time of closest approach: conjunctions with a satellite already designated to
    burn in an earlier conjunction, and conjunctions that share a satellite with another one."""
    order = sorted(range(len(conjs)), key=lambda k: conjs[k].time_h)
    out = {"n_conjunctions": len(conjs)}
    for variant in ("fixed", "deterministic"):
        who = designated_burners(conjs, variant)
        seen, count = set(), 0
        for k in order:
            if seen & set(conjs[k].sats):
                count += 1
            seen.add(who[k])
        key = "fixed_tie_break" if variant == "fixed" else "balanced_deterministic"
        out[f"after_earlier_burn_{key}"] = count
    uses = np.bincount([s for c in conjs for s in c.sats], minlength=N_SATELLITES)
    out["sharing_a_satellite"] = int(sum(1 for c in conjs if uses[c.sat1] > 1 or uses[c.sat2] > 1))
    return out


TIE_TOL = 1e-12          # the tolerance of games.pure_nash


def burn_size_tie_slots(conj) -> int:
    """Number of slots of this conjunction (0, 1 or 2) with a burn-size tie.

    Definition (the no-tie assumption of P3): a slot has a tie if
    the small and the large burn are both best replies to N on the other side, that is
    g_s(S, N) = g_s(L, N) <= g_s(N, N), with g_s the slot's cost table indexed [own, other].
    Tolerance: absolute 1e-12, the one ``games.pure_nash`` uses; a burn is a best reply to N if
    its cost is at most the smallest of the three costs against N plus 1e-12.
    """
    n = 0
    for side in (0, 1):
        g = _side_cost(conj, side)[:, 0]          # own action N, S, L against N on the other side
        best = g.min()
        n += bool(g[1] <= best + TIE_TOL and g[2] <= best + TIE_TOL)
    return n


def n_slots_with_burn_size_tie(conjs) -> int:
    """Number of slots with a burn-size tie (``burn_size_tie_slots``) over the given conjunctions."""
    return int(sum(burn_size_tie_slots(c) for c in conjs))


# ----------------------------------------------------------------------------- the run
def _family(batches) -> dict:
    kept = [c for b in batches for c in b.conjunctions]
    draws = sum(b.n_draws for b in batches)
    infos = [designation_info(c) for c in kept]
    n_sd = [len(c.stable) for c in kept]
    two = [i for i, n in zip(infos, n_sd) if n == 2]
    return {"n_batches": len(batches), "n_draws": int(draws), "n_kept": len(kept),
            "share_kept": len(kept) / draws,
            "share_one_stable_designation": float(np.mean([n == 1 for n in n_sd])),
            "share_two_stable_designations": float(np.mean([n == 2 for n in n_sd])),
            "n_more_than_two_stable_designations": int(sum(n > 2 for n in n_sd)),
            "n_two_on_the_same_side": int(sum(1 for c in kept if len(c.stable) == 2
                                              and burner(c.stable[0]) == burner(c.stable[1]))),
            "share_free_among_two": float(np.mean([i["free"] for i in two])) if two else 0.0,
            "n_slots_with_burn_size_tie": n_slots_with_burn_size_tie(kept)}


def _conj_record(c, private, public) -> dict:
    info = designation_info(c)
    return {"index": c.index, "sat1": c.sat1, "sat2": c.sat2, "op1": c.op1, "op2": c.op2,
            "time_h": c.time_h, "n_draws": c.n_draws,
            "stable_designations": [cell_label(e) for e in c.stable],
            "stable_costs": [float(c.obj[e]) for e in c.stable],
            "cheapest_stable_cost": info["cost"], "free": info["free"],
            "both_sides_stable": info["both_sides"], "fixed_rule_side": fixed_rule_side(c) + 1,
            "private_cost": private, "public_cost": public}


def strictly_cheaper(a: float, b: float) -> bool:
    """a < b by more than the relative threshold used throughout (1e-9)."""
    return a < b * (1 - FREE_TOL)


def best_public_direct(conj):
    """Best public rule computed directly: the cheapest pure Nash equilibrium of the conjunction's
    game, ties broken as in ``games.solve_public``.  Returns (cell, coordinator cost)."""
    nash = G.pure_nash(conj.c1, conj.c2)
    if not nash:
        return None, np.inf
    cell = min(nash, key=lambda a: (round(float(conj.obj[a]), 12), a))
    return cell, float(conj.obj[cell])


def private_costs_per_conjunction(conjs) -> list:
    """Private optimum (rule known, zero margin) of every conjunction, read from program (c)."""
    per = per_conjunction_program(conjs)
    return [float(c.obj.ravel() @ t.ravel()) for c, t in zip(conjs, per["tables"])]


def _totals(conjs, private, public) -> dict:
    p, u = float(np.sum(private)), float(np.sum(public))
    s = closed_form(conjs)
    stable = [designation_info(c)["cost"] for c in conjs]
    prv_cheaper = [strictly_cheaper(a, b) for a, b in zip(private, public)]
    beats = []
    for c, u_k, s_k in zip(conjs, public, stable):
        if strictly_cheaper(u_k, s_k):
            cell, _ = best_public_direct(c)
            beats.append({"index": c.index, "public_cell": cell_label(cell), "public_cost": float(u_k),
                          "cheapest_stable_cost": float(s_k)})
    return {"K": len(conjs), "sum_private": p, "sum_public": u, "sum_cheapest_stable": s,
            "rel_gap_private_vs_stable": (s - p) / s, "rel_gap_public_vs_stable": (s - u) / s,
            "rel_gap_private_vs_public": (u - p) / u,
            "n_public_strictly_cheaper": int(sum(1 for a, b in zip(public, stable) if a < b * (1 - FREE_TOL))),
            # private strictly cheaper than the cheapest stable designation (kept from the first run)
            "n_private_strictly_cheaper": int(sum(1 for a, b in zip(private, stable) if a < b * (1 - FREE_TOL))),
            "n_private_strictly_cheaper_than_public": int(sum(prv_cheaper)),
            "share_of_public_total_where_private_strictly_cheaper":
                float(sum(u_k for u_k, f in zip(public, prv_cheaper) if f)) / u,
            "public_strictly_cheaper_than_stable": beats}


GAP_KEYS = ("rel_gap_private_vs_stable", "rel_gap_public_vs_stable", "rel_gap_private_vs_public")
COUNT_KEYS = ("n_public_strictly_cheaper", "n_private_strictly_cheaper_than_public")


def _spread(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def totals_fast(conjs) -> dict:
    """Totals of one batch with the private sums from program (c) and the best public rule as the
    cheapest pure equilibrium computed directly (``best_public_direct``)."""
    private = private_costs_per_conjunction(conjs)
    public = [best_public_direct(c)[1] for c in conjs]
    stable = [designation_info(c)["cost"] for c in conjs]
    p, u, s = float(np.sum(private)), float(np.sum(public)), float(np.sum(stable))
    return {"K": len(conjs), "sum_private": p, "sum_public": u, "sum_cheapest_stable": s,
            "rel_gap_private_vs_stable": (s - p) / s, "rel_gap_public_vs_stable": (s - u) / s,
            "rel_gap_private_vs_public": (u - p) / u,
            "n_public_strictly_cheaper": int(sum(strictly_cheaper(a, b) for a, b in zip(public, stable))),
            "n_private_strictly_cheaper_than_public": int(sum(strictly_cheaper(a, b) for a, b in zip(private, public))),
            "private": private, "public": public, "stable": stable}


def family_private_vs_public(rows) -> dict:
    """Over all conjunctions of the batches whose ``totals_fast`` rows are given."""
    prv = np.concatenate([r["private"] for r in rows])
    pub = np.concatenate([r["public"] for r in rows])
    cheaper = prv < pub * (1 - FREE_TOL)
    saving = (pub - prv) / pub
    return {"n_conjunctions": int(prv.size), "share_private_strictly_cheaper": float(cheaper.mean()),
            "mean_relative_saving_all": float(saving.mean()),
            "median_relative_saving_where_positive": float(np.median(saving[cheaper])) if cheaper.any() else 0.0,
            "positive_means": "private strictly cheaper than public (1e-9 relative)"}


def rejected_breakdown(batches) -> dict:
    """Among rejected draws: nobody burning (NN) is a pure equilibrium; some pure equilibrium has
    both sides burning; no pure equilibrium at all.  The three need not be exclusive."""
    nn = both = none = total = 0
    for b in batches:
        for _, scn in b.rejected:
            c1, c2 = M.cost_tables(scn)
            nash = G.pure_nash(c1, c2)
            total += 1
            nn += (0, 0) in nash
            both += any(a1 > 0 and a2 > 0 for a1, a2 in nash)
            none += not nash
    return {"n_rejected": int(total), "n_nobody_burns_is_pure_equilibrium": int(nn),
            "n_pure_equilibrium_with_both_burning": int(both), "n_no_pure_equilibrium": int(none)}


def _burden_record(conjs) -> dict:
    b = burden(conjs)
    rec = {k: v for k, v in b.items() if not isinstance(v, np.ndarray)}
    rec["histogram_fixed_tie_break"] = histogram(b["burns_fixed_tie_break"])
    rec["histogram_balanced_deterministic"] = histogram(b["burns_balanced_deterministic"])
    rec["burns_fixed_tie_break"] = b["burns_fixed_tie_break"].tolist()
    rec["burns_balanced_deterministic"] = b["burns_balanced_deterministic"].tolist()
    return rec


def _stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def run(master_seed: int = MASTER_SEED, n_master: int = MASTER_SIZE, k_grid=K_GRID,
        further_seeds=FURTHER_SEEDS, size_cap: int = SIZE_CAP, repeats: int = REPEATS,
        p1_ks=None, p5_ks=P5_KS, burden_ks=BURDEN_KS, log=print) -> dict:
    """Everything the brief asks for; raises ``CheckFailed`` if a check does not hold.

    ``p1_ks=None`` runs the P1 check at every K of the grid within the full-joint size cap (the
    brief's K = 2 and 3 included); the tests run it at ``P1_KS``."""
    t_start = time.perf_counter()
    master = generate_batch(master_seed, n_master)
    conjs = master.conjunctions
    further = [generate_batch(s, MASTER_SIZE) for s in further_seeds]
    log(f"batches drawn ({len(further) + 1})")

    private = [G.solve_private(c.c1, c.c2, c.obj)["objective"] for c in conjs]
    public = [G.solve_public(c.c1, c.c2, c.obj)["objective"] for c in conjs]
    res = {"settings": {
        "satellites": N_SATELLITES, "operators": N_OPERATORS, "fleets": [list(f) for f in FLEETS],
        "n_pairs": len(PAIRS), "window_hours": WINDOW_HOURS, "master_seed": master_seed, "master_size": n_master,
        "k_grid": list(k_grid), "size_cap": size_cap, "timing_repeats": repeats,
        "agreement_tolerance": AGREEMENT_TOL, "feasibility_tolerance": FEASIBILITY_TOL, "free_tolerance": FREE_TOL,
        "further_seeds": [int(further_seeds[0]), int(further_seeds[-1])] if len(further_seeds) else [],
        "n_further_batches": len(further_seeds), "highs_options": HIGHS_OPTIONS,
        "fixed_tie_break": "the side that model.fixed_rule_profile designates for the conjunction's "
                           "scenario (cheaper=True) burns, using that side's stable designation"}}
    res["family"] = _family([master] + further)
    res["family_master_only"] = _family([master])
    res["family"]["rejected_breakdown"] = rejected_breakdown([master] + further)
    res["family_master_only"]["rejected_breakdown"] = rejected_breakdown([master])
    # spread over the further batches (added after the first run, step S1b)
    fast = [totals_fast(b.conjunctions) for b in further]
    res["totals_further_batches"] = {
        "n_batches": len(fast), "K": MASTER_SIZE,
        "private_from": "program (c)", "public_from": "cheapest pure equilibrium, computed directly",
        "batches": [{"seed": int(b.seed), **{k: v for k, v in r.items() if k not in ("private", "public", "stable")}}
                    for b, r in zip(further, fast)],
        "summary": {k: _spread([r[k] for r in fast]) for k in GAP_KEYS + COUNT_KEYS},
    }
    res["family_private_vs_public"] = family_private_vs_public(fast)
    res["master_batch"] = [_conj_record(c, p, u) for c, p, u in zip(conjs, private, public)]

    failures = []

    def need(ok, text):
        if not ok:
            failures.append(text)
            log("CHECK FAILED: " + text)

    # --- timing table and checks P2, P3 at every K within the size cap -------------------------
    timing, p2, p3 = [], [], []
    for K in k_grid:
        sub = conjs[:K]
        per, t_c = timed(per_conjunction_program, sub, repeats=repeats)
        timing.append({"K": K, "formulation": "c", "variables": per["variables"], "rows": per["rows"],
                       "nonzeros": per["nonzeros"], "optimum": per["optimum"], "wall_clock_seconds": t_c})
        cf, t_cf = timed(closed_form, sub, repeats=repeats)
        timing.append({"K": K, "formulation": "c_closed_form", "variables": 0, "rows": 0, "nonzeros": 0,
                       "optimum": cf, "wall_clock_seconds": t_cf})
        psum = float(np.sum(private[:K]))
        rec2 = {"K": K, "optimum_per_conjunction": per["optimum"], "sum_solve_private": psum,
                "rel_diff": rel_diff(per["optimum"], psum)}
        if 9 ** K <= size_cap:
            full, t_a = timed(joint_program, sub, "full", repeats=repeats)
            timing.append({"K": K, "formulation": "a", "variables": full["variables"], "rows": full["rows"],
                           "nonzeros": full["nonzeros"], "optimum": full["optimum"], "wall_clock_seconds": t_a})
            rec2 = check_p2(sub, full=full, per=per, private_sum=psum)
            log(f"K={K} (a) {full['variables']} variables, {t_a:.2f} s")
        p2.append(rec2)
        need(rec2["rel_diff"] <= AGREEMENT_TOL, f"P2 at K={K}: rel diff {rec2['rel_diff']:.3g}")
        for key in ("marginal_violation", "product_violation"):
            if key in rec2:
                need(rec2[key] <= FEASIBILITY_TOL, f"P2 {key} at K={K}: {rec2[key]:.3g}")
        if 4 ** K <= size_cap:
            ex, t_b = timed(joint_program, sub, "exact_one", repeats=repeats)
            timing.append({"K": K, "formulation": "b", "variables": ex["variables"], "rows": ex["rows"],
                           "nonzeros": ex["nonzeros"], "optimum": ex["optimum"], "wall_clock_seconds": t_b})
            rec3 = check_p3(sub, ex=ex)
            p3.append(rec3)
            need(rec3["rel_diff"] <= AGREEMENT_TOL, f"P3 at K={K}: (b) {ex['optimum']!r} vs (c') {rec3['closed_form']!r}")
            need(rec3["mass_outside_stable_products"] <= FEASIBILITY_TOL,
                 f"P3 mass outside at K={K}: {rec3['mass_outside_stable_products']:.3g}")
            log(f"K={K} (b) {ex['variables']} variables, {t_b:.2f} s")
    infeas = check_p3_infeasible(master)
    need(infeas["empty_E_k"] and infeas["infeasible"], f"P3 infeasibility: status {infeas['status']}")

    # --- P1 ------------------------------------------------------------------------------------
    if p1_ks is None:                       # every K of the grid within the full-joint size cap
        p1_ks = [K for K in k_grid if 9 ** K <= size_cap]
    p1 = [check_p1(conjs[:K]) for K in p1_ks]
    for r in p1:
        need(r["rel_diff"] <= AGREEMENT_TOL, f"P1 at K={r['K']}: rel diff {r['rel_diff']:.3g}")
        need(r["violation_of_all_deviation_rows"] <= FEASIBILITY_TOL,
             f"P1 violation at K={r['K']}: {r['violation_of_all_deviation_rows']:.3g}")

    # --- P5 ------------------------------------------------------------------------------------
    p5 = []
    for K in p5_ks:
        a, b = balanced_randomised(conjs[:K]), balanced_randomised_joint(conjs[:K])
        p5.append({"K": K, "n_free": a["n_free"], "marginal_program": a["max_expected_burns"],
                   "variables_marginal_program": a["variables"], "joint_program": b["max_expected_burns"],
                   "variables_joint_program": b["variables"],
                   "rel_diff": rel_diff(a["max_expected_burns"], b["max_expected_burns"])})
        need(p5[-1]["rel_diff"] <= AGREEMENT_TOL, f"P5 at K={K}: {a['max_expected_burns']!r} vs {b['max_expected_burns']!r}")

    def worst(rows, key="rel_diff"):
        r = max(rows, key=lambda r: r[key])
        return {"largest": r[key], "at_K": r["K"]}

    res["propositions"] = {
        "P1": {"checks": p1, "largest_rel_diff": worst(p1),
               "largest_violation": worst(p1, "violation_of_all_deviation_rows")},
        "P2": {"checks": p2, "largest_rel_diff": worst(p2),
               "largest_marginal_violation": worst([r for r in p2 if "marginal_violation" in r], "marginal_violation"),
               "largest_product_violation": worst([r for r in p2 if "product_violation" in r], "product_violation")},
        "P3": {"checks": p3, "largest_rel_diff": worst(p3),
               "largest_mass_outside": worst(p3, "mass_outside_stable_products"), "infeasible_batch": infeas},
        "P4": {f"K{K}": {"n_both_sides_stable": int(sum(designation_info(c)["both_sides"] for c in conjs[:K])),
                         "n_free": int(sum(designation_info(c)["free"] for c in conjs[:K]))}
               for K in burden_ks},
        "P5": {"checks": p5, "largest_rel_diff": worst(p5)},
    }
    res["timing"] = timing

    # --- totals, burden, carry-over ------------------------------------------------------------
    res["totals"] = {f"K{K}": _totals(conjs[:K], private[:K], public[:K]) for K in burden_ks}
    res["burden"] = {f"master_K{K}": _burden_record(conjs[:K]) for K in burden_ks}
    rows = [burden(b.conjunctions) for b in further]
    res["burden"]["further_batches"] = {
        "n_batches": len(rows), "K": MASTER_SIZE,
        "n_free": _stats([r["n_free"] for r in rows]),
        "max_burns_fixed_tie_break": _stats([r["max_burns_fixed_tie_break"] for r in rows]),
        "max_expected_burns_balanced_randomised": _stats([r["max_expected_burns_balanced_randomised"] for r in rows]),
        "max_burns_balanced_deterministic": _stats([r["max_burns_balanced_deterministic"] for r in rows]),
    }
    # brute force wherever at most 12 conjunctions are free: the burden batches and, since few of
    # those qualify, every master prefix of the K grid
    prefixes = [burden(conjs[:K]) for K in k_grid]
    brute = [(r["brute_force_max_burns"], r["max_burns_balanced_deterministic"])
             for r in rows + prefixes if "brute_force_max_burns" in r]
    res["burden"]["brute_force_agreement"] = {
        "n_checked": len(brute), "n_disagreeing": int(sum(a != b for a, b in brute)),
        "master_prefixes_checked": [r["n_conjunctions"] for r in prefixes if "brute_force_max_burns" in r],
        "further_batches_checked": int(sum("brute_force_max_burns" in r for r in rows))}
    need(all(a == b for a, b in brute), "deterministic burden program differs from brute force")
    n_free_master = res["burden"][f"master_K{n_master}"]["n_free"] if n_master in burden_ks else None
    res["burden"]["fewer_than_5_free_in_master"] = bool(n_free_master is not None and n_free_master < 5)
    res["carry_over"] = carry_over(conjs)

    res["checks_passed"] = not failures
    res["failures"] = failures
    res["runtime_seconds"] = time.perf_counter() - t_start
    if failures:
        raise CheckFailed("; ".join(failures))
    return res
