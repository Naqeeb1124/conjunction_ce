"""Exact (sampling-free) evaluation of the learning model in ``inference``.

Everything here uses ``inference.exact_realised_joint``, which enumerates the multinomial counts
of the past events, so no number below carries Monte Carlo noise (apart from the explicit
simulation-agreement check).  Both conventions for what an operator knows before it has seen a
recommendation are evaluated:

* "observed": k - 1 past events plus one observation per recommendation drawn from the true
  conditional (the convention of every stored learning result);
* "uniform": exactly k past events; a recommendation never seen leaves a uniform belief.

``run()`` returns a JSON-serialisable dict.
"""
from __future__ import annotations

import time

import numpy as np

from . import experiments as E
from . import inference as I
from . import model as M

__all__ = ["run", "K_GRID_EXACT", "k_grid", "family_best", "simulation_agreement", "two_parameter_tables",
           "agreement_cases"]

# k = 1..40 and a coarse tail for both families; the uniform family also gets every k from 41 to 130
K_GRID_EXACT = tuple(range(1, 41)) + (50, 70, 100, 150, 200, 300)
K_GRID_UNIFORM_EXTRA = tuple(range(41, 131))
GAIN = 0.01                     # crossover: realised cost at most (1 - GAIN) x the public rule's cost
TWO_PARAMETER_KS = (10, 15, 20, 30, 50, 100)
TWO_PARAMETER_GRID_N = 25
AGREEMENT_KS = (5, 50)
AGREEMENT_MARGINS = {"uniform": 1e-4, "targeted": 6e-4}
AGREEMENT_N_REP = 200_000


def k_grid(family_name: str) -> list:
    ks = set(K_GRID_EXACT)
    if family_name == "uniform":
        ks |= set(K_GRID_UNIFORM_EXTRA)
    return sorted(ks)


def _cost(x, c1, c2, obj, k, start):
    ex = I.exact_realised_joint(x, c1, c2, k, start=start)
    return float(np.sum(ex["y"] * obj)), ex["obey"]


def family_best(family, c1, c2, obj, k, start):
    """Exact minimum of the realised coordinator cost over a family's members at k."""
    best = None
    for f in family:
        cost, obey = _cost(f["x"], c1, c2, obj, k, start)
        if best is None or cost < best["cost"]:
            best = {"k": int(k), "cost": cost, "mu": float(f["mu"]), "obey": float(obey)}
    return best


def _crossover(rows, public_cost):
    thr = (1.0 - GAIN) * public_cost
    ok = [r["cost"] <= thr for r in rows]
    first = next((i for i, v in enumerate(ok) if v), None)
    return {"threshold": thr,
            "first_k": rows[first]["k"] if first is not None else None,
            "later_k_failing": [r["k"] for i, r in enumerate(rows) if first is not None and i > first and not ok[i]]}


def two_parameter_tables(scn, grid_n=TWO_PARAMETER_GRID_N):
    """The (alpha_1, alpha_2) grid of ``experiments.two_parameter_check``: the same function,
    ``experiments.two_parameter_grid``, with this module's default grid size."""
    return E.two_parameter_grid(scn, grid_n)


def _two_parameter_gap(scn, targeted, c1, c2, obj):
    tables, alphas, a_star = two_parameter_tables(scn)
    rows = []
    for k in TWO_PARAMETER_KS:
        costs = [_cost(x, c1, c2, obj, k, "observed")[0] for x in tables]
        j = int(np.argmin(costs))
        one = family_best(targeted, c1, c2, obj, k, "observed")
        rows.append({"k": int(k), "two_parameter_cost": costs[j], "alpha": [float(a) for a in alphas[j]],
                     "one_parameter_cost": one["cost"], "one_parameter_mu": one["mu"],
                     "relative_gap": (one["cost"] - costs[j]) / costs[j]})
    return {"start": "observed", "grid_n": TWO_PARAMETER_GRID_N, "alpha_thresholds": a_star.tolist(), "rows": rows}


def agreement_cases(scn=M.BASELINE):
    """(label, x, k, start) for the simulation-against-exact comparison."""
    _, families, _ = E.baseline_schemes(scn)
    cases = []
    for fam, mu in AGREEMENT_MARGINS.items():
        f = min(families[fam], key=lambda g: abs(g["mu"] - mu))
        for start in I.STARTS:
            for k in AGREEMENT_KS:
                cases.append((f"{fam} mu={f['mu']:.0e} {start} k={k}", f["x"], k, start))
    return cases


def simulation_agreement(scn=M.BASELINE, n_rep=AGREEMENT_N_REP, seed=11) -> dict:
    """Simulation (``realised_joint``) against exact enumeration: number of standard errors for
    P(both follow), P(operator i follows) and every cell of y, in every case of ``agreement_cases``.

    The standard error is the sample one.  When the sample standard deviation is at rounding level
    (below 1e-9: every replicate gave the same value, i.e. no replicate saw a recommendation not
    followed), it is replaced by sqrt((1 - P_both) / n), with P_both the exact probability that
    both operators follow.  This bounds the true standard error of every quantity compared: in
    each replicate, P(both follow), P(operator i follows) and every cell of y differ from their
    full-obedience values by at most the replicate's probability q that some recommendation is
    not followed, so their variance is at most E[q^2] <= E[q] = 1 - P_both."""
    c1, c2, _, _ = E.nominal(scn)
    out, worst = [], 0.0
    for i, (label, x, k, start) in enumerate(agreement_cases(scn)):
        ex = I.exact_realised_joint(x, c1, c2, k, start=start)
        sim = I.realised_joint(x, c1, c2, k=k, n_rep=n_rep, rng=np.random.default_rng(seed + i), start=start)
        zs = {}
        quantities = [("obey", sim["obey"], ex["obey"]), ("obey_1", sim["obey_1"], ex["obey_1"]),
                      ("obey_2", sim["obey_2"], ex["obey_2"])]
        quantities += [(f"y[{a},{b}]", sim["y"][:, a, b], ex["y"][a, b])
                       for a in range(x.shape[0]) for b in range(x.shape[1])]
        se_fallback = float(np.sqrt(max(1.0 - ex["obey"], 0.0) / n_rep))
        n_fallback = 0
        for name, v, p in quantities:
            diff = float(v.mean() - p)
            sd = float(v.std())
            if sd < 1e-9:
                se = se_fallback
                n_fallback += 1
            else:
                se = sd / np.sqrt(len(v))
            zs[name] = 0.0 if diff == 0.0 else (abs(diff) / se if se > 0 else float("inf"))
        case_max = max(zs.values())
        worst = max(worst, case_max)
        out.append({"case": label, "max_z": case_max, "z_obey": zs["obey"], "z": zs,
                    "n_quantities_with_bound_se": n_fallback})
    return {"n_rep": int(n_rep), "max_z": worst, "cases": out}


def run(scn=M.BASELINE) -> dict:
    t0 = time.time()
    c1, c2, obj, _ = E.nominal(scn)
    schemes, families, info = E.baseline_schemes(scn)
    public_cost = float(np.sum(schemes["public"] * obj))
    # the public rule is a pure Nash profile, so it is followed after any number of events;
    # record the largest difference between its learned and known-rule cost as a check
    pub_dev = max(abs(_cost(schemes["public"], c1, c2, obj, k, start)[0] - public_cost)
                  for start in I.STARTS for k in K_GRID_EXACT)

    best, cross = {}, {}
    for fam in ("uniform", "targeted"):
        best[fam], cross[fam] = {}, {}
        for start in I.STARTS:
            rows = [family_best(families[fam], c1, c2, obj, k, start) for k in k_grid(fam)]
            best[fam][start] = rows
            cross[fam][start] = _crossover(rows, public_cost)
    for fam in cross:
        cross[fam]["public_cost"] = public_cost

    out = {
        "public_cost": public_cost,
        "public_learned_max_abs_deviation": float(pub_dev),
        "family_sizes": {fam: len(families[fam]) for fam in families},
        "family_mu_max": {fam: float(families[fam][-1]["mu"]) for fam in families},
        "best_per_k": best,
        "crossover": cross,
        "two_parameter_gap": _two_parameter_gap(scn, families["targeted"], c1, c2, obj),
        "simulation_agreement": simulation_agreement(scn),
    }
    out["runtime_s"] = time.time() - t0
    return out


if __name__ == "__main__":       # pragma: no cover
    import json
    print(json.dumps(run(), indent=1))
