"""The experiments behind RESULTS.md.  ``run_all.py`` calls these and stores the numbers."""
from __future__ import annotations

import dataclasses
import time

import numpy as np
from scipy import stats

from . import games as G
from . import inference as I
from . import model as M
from . import scenario as SC

K_GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000, 5000, 10000)
STEP_UNIFORM = 1e-5         # margins are fractions of the satellite's value, like the cost tables
STEP_TARGETED = 2e-5


# ------------------------------------------------------------------------------- helpers
def nominal(scn: M.Scenario):
    c1, c2 = M.cost_tables(scn)
    return c1, c2, M.coordinator_cost_table(scn), M.pc_table(scn)


def select_mixed_nash(c1, c2):
    """The equilibrium used as the 'no coordinator' prediction: the one with the most mixing.
    With two asymmetric pure equilibria and no way to agree on one, this is the mixed one."""
    eqs = G.support_enumeration(c1, c2)
    return max(eqs, key=lambda e: (np.count_nonzero(e[0] > 1e-9) + np.count_nonzero(e[1] > 1e-9)))


def margin_family(c1, c2, obj, rows, step: float, mu_max: float = 2e-3):
    """Every feasible, genuinely private rule on a margin grid: a list of dicts (mu, x, objective).

    ``rows`` says where the margin applies (see ``games.solve_private``).  The family stops when
    the program becomes infeasible or when every recommendation reveals what the other operator
    was told.  Such a rule is public in effect, so it is not counted as a private one.
    """
    out = []
    for mu in np.arange(0.0, mu_max + 0.5 * step, step):
        res = G.solve_private(c1, c2, obj, margin=float(mu), margin_rows=rows)
        if res["x"] is None or G.reveals_other(res["x"]):
            break
        out.append({"mu": float(mu), "x": res["x"], "objective": res["objective"]})
    return out


def _nearest(family, mu):
    return min(family, key=lambda f: abs(f["mu"] - mu))


def baseline_schemes(scn: M.Scenario):
    """The recommendation rules compared throughout, all computed from nominal information.

    Two margin families:

    * "uniform": the same margin on every incentive constraint (the chance-constrained
      correlated-equilibrium form);
    * "targeted": the margin only on the constraints where an operator who has to learn the rule
      can go wrong (``games.uncertain_belief_rows``, evaluated on the zero-margin rule).
    """
    c1, c2, obj, _ = nominal(scn)
    p, q = select_mixed_nash(c1, c2)
    fam_uni = margin_family(c1, c2, obj, "all", STEP_UNIFORM)
    mask = G.uncertain_belief_rows(fam_uni[0]["x"], c1, c2)
    fam_tar = margin_family(c1, c2, obj, mask, STEP_TARGETED)
    t3, t6 = _nearest(fam_tar, 3e-4), _nearest(fam_tar, 6e-4)
    schemes = {
        "fixed_rule": G.point_table(M.fixed_rule_profile(scn), c1.shape),
        "nash_mixed": G.product_table(p, q),
        "public": G.solve_public(c1, c2, obj)["x"],
        "private_m0": fam_uni[0]["x"],
        "private_uniform": fam_uni[-1]["x"],
        "private_targeted_3": t3["x"],
        "private_targeted_6": t6["x"],
    }
    _, _, row_info = G.obedience_system(c1, c2)
    info = {"mu_uniform_max": fam_uni[-1]["mu"], "mu_targeted_max": fam_tar[-1]["mu"],
            "mu_targeted_3": t3["mu"], "mu_targeted_6": t6["mu"],
            "targeted_rows": [list(r) for r, m in zip(row_info, mask) if m],
            "mixed_nash": (p.tolist(), q.tolist())}
    return schemes, {"uniform": fam_uni, "targeted": fam_tar}, info


def _cost_stats(scn, y, pc=None):
    """Mean and standard error of the coordinator's cost over replicates, plus the usual metrics."""
    obj = M.coordinator_cost_table(scn, pc)
    per_rep = np.sum(y * obj, axis=(-2, -1))
    m = M.outcome_metrics(scn, y, pc)
    m["cost_se"] = float(np.std(per_rep) / np.sqrt(per_rep.size)) if per_rep.size > 1 else 0.0
    return m


# ------------------------------------------------------------------------------- E1
def nominal_comparison(scn: M.Scenario) -> dict:
    """Each rule's outcome if operators know the rule and the coordinator knows the costs."""
    schemes, _, _ = baseline_schemes(scn)
    c1, c2, _, _ = nominal(scn)
    out = {}
    for name, x in schemes.items():
        m = M.outcome_metrics(scn, x)
        m["max_obedience_slack"] = float(np.max(G.obedience_slack(x, c1, c2)))
        m["table"] = x.tolist()
        out[name] = m
    return out


# ------------------------------------------------------------------------------- E2
def learning_curve(scn, x, ks=K_GRID, n_rep=40000, seed=0) -> list:
    """Realised outcome of rule x when both operators learned it from k past events (costs known)."""
    c1, c2, _, _ = nominal(scn)
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        out = I.realised_joint(x, c1, c2, k=int(k), n_rep=n_rep, rng=rng)
        m = _cost_stats(scn, out["y"])
        m.update(k=int(k), obey=float(out["obey"].mean()),
                 hoeffding=I.hoeffding_disobedience_bound(x, c1, c2, int(k)))
        rows.append(m)
    return rows


def best_margin_per_k(scn, family, ks=K_GRID, n_select=20000, n_eval=100000, seed=0,
                      sig_cost=None, sig_k=0.1) -> list:
    """For each k: pick the margin with the lowest realised cost on one set of simulations, then
    re-evaluate that choice on an independent set, so the reported number is not biased by the
    selection.

    With ``sig_cost=None`` the costs are the nominal ones and only the learning histories are
    random.  Otherwise every replicate is also a sampled true game (cost log-sd ``sig_cost``,
    covariance-scale log-sd ``sig_k``).  ``k=None`` in ``ks`` means the rule is known.
    """
    if sig_cost is None:
        c1, c2, _, _ = nominal(scn)
        sel = {"c1": c1, "c2": c2, "pc": None}
        ev = sel
    else:
        sel = M.sample_true_games(scn, n_select, np.random.default_rng(seed + 1), sig_cost, sig_cost, sig_cost, sig_k)
        ev = M.sample_true_games(scn, n_eval, np.random.default_rng(seed + 2), sig_cost, sig_cost, sig_cost, sig_k)
    rng_sel = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 10_000)
    rows = []
    for k in ks:
        scores = [_cost_stats(scn, I.realised_joint(f["x"], sel["c1"], sel["c2"], k=k, n_rep=n_select,
                                                    rng=rng_sel)["y"], sel["pc"])["coordinator_cost"]
                  for f in family]
        j = int(np.argmin(scores))
        out = I.realised_joint(family[j]["x"], ev["c1"], ev["c2"], k=k, n_rep=n_eval, rng=rng_eval)
        m = _cost_stats(scn, out["y"], ev["pc"])
        m.update(k=k, mu=family[j]["mu"], obey=float(out["obey"].mean()),
                 known_rule_cost=family[j]["objective"], both_small=float(family[j]["x"][1, 1]))
        rows.append(m)
    return rows


def crossover(rows, reference_cost: float, n_sigma: float = 3.0, min_gain: float = 0.01):
    """Smallest k on the grid from which the rule stays below the reference cost by at least
    ``min_gain`` (relative) and by more than n_sigma standard errors, or None."""
    below = [r["coordinator_cost"] + n_sigma * r["cost_se"] < reference_cost * (1.0 - min_gain) for r in rows]
    for i, r in enumerate(rows):
        if all(below[i:]):
            return r["k"]
    return None


def margin_scan(scn, family, ks=(15, 30, 100, 300), n_rep=100000, seed=6, n_games=10000,
                uncertainty=((0.1, 0.05), (0.3, 0.1))) -> list:
    """For every margin in a family: what the margin costs when everything is known, how often a
    recommendation is not followed when the rule was learned from k events, and how often some
    recommendation would not be followed when costs are uncertain (rule known)."""
    c1, c2, _, _ = nominal(scn)
    rng = np.random.default_rng(seed)
    game_sets = [M.sample_true_games(scn, n_games, np.random.default_rng(seed + 1 + i), sc, sc, sc, sk)
                 for i, (sc, sk) in enumerate(uncertainty)]
    rows = []
    for f in family:
        row = {"mu": f["mu"], "nominal_cost": f["objective"], **M.outcome_metrics(scn, f["x"]),
               "both_small": float(f["x"][1, 1])}
        for k in ks:
            out = I.realised_joint(f["x"], c1, c2, k=int(k), n_rep=n_rep, rng=rng)
            row[f"not_followed_k{k}"] = float(1.0 - out["obey"].mean())
            row[f"cost_k{k}"] = _cost_stats(scn, out["y"])["coordinator_cost"]
        for (sc, sk), games in zip(uncertainty, game_sets):
            res = I.realised_joint(f["x"], games["c1"], games["c2"], k=None)
            row[f"violation_sig{sc}"] = float(np.mean(res["obey"] < 1.0 - 1e-9))
            row[f"cost_sig{sc}"] = _cost_stats(scn, res["y"], games["pc"])["coordinator_cost"]
        rows.append(row)
    return rows


def two_parameter_grid(scn, grid_n: int):
    """The (alpha_1, alpha_2) grid of three-action recommendation tables (DERIVATIONS.md 4.4).

    alpha_i is the probability, given that operator i is told to make a small burn, that the other
    operator was told not to burn.  Each alpha_i runs over ``grid_n`` points from the threshold
    a_star[i] to 0.97.  Returns (tables, alphas, a_star): a list of 3x3 tables, the matching list
    of (alpha_1, alpha_2) pairs, and the array of the two thresholds."""
    _, _, obj, pc = nominal(scn)
    kappa = M.maneuver_cost_fraction(scn)[:, 1]
    p0, p1, p2 = pc[0, 0], pc[1, 0], pc[1, 1]
    a_star = (kappa - (p1 - p2)) / ((p0 - p1) - (p1 - p2))

    def table(a1, a2):
        both = 1.0 / (1.0 + a1 / (1 - a1) + a2 / (1 - a2))
        x = np.zeros_like(obj)
        x[1, 1], x[1, 0], x[0, 1] = both, both * a1 / (1 - a1), both * a2 / (1 - a2)
        return x

    alphas = [(a1, a2) for a1 in np.linspace(a_star[0], 0.97, grid_n)
              for a2 in np.linspace(a_star[1], 0.97, grid_n)]
    return [table(a1, a2) for a1, a2 in alphas], alphas, a_star


def two_parameter_check(scn, targeted_family, ks=(10, 15, 20, 30, 50, 100, 300, 1000), grid_n=30,
                        n_select=8000, n_eval=100000, seed=9) -> dict:
    """How much does the one-parameter targeted family leave on the table?

    In the baseline every private rule uses three joint actions: both small, only operator 1,
    only operator 2.  Such a table is fixed by two numbers, alpha_1 and alpha_2: the probability,
    given that an operator is told to burn, that the other was told not to (DERIVATIONS.md 4.4).
    This scans a grid of (alpha_1, alpha_2), picks the best table for each k on one set of
    simulated histories, re-evaluates it on an independent set, and does the same for the
    one-parameter family.
    """
    c1, c2, _, _ = nominal(scn)
    x0 = targeted_family[0]["x"]
    support = {tuple(ix) for ix in np.argwhere(x0 > 1e-9)}
    if support != {(1, 1), (1, 0), (0, 1)}:
        raise ValueError("the two-parameter family assumes the baseline's three-action structure")
    grid, grid_alpha, a_star = two_parameter_grid(scn, grid_n)
    rows = []
    for k in ks:
        rng = np.random.default_rng(seed + k)
        rng_eval = np.random.default_rng(seed + 50_000 + k)
        cost = lambda x, n, r: _cost_stats(scn, I.realised_joint(x, c1, c2, k=int(k), n_rep=n, rng=r)["y"])
        j2 = int(np.argmin([cost(x, n_select, rng)["coordinator_cost"] for x in grid]))
        j1 = int(np.argmin([cost(f["x"], n_select, rng)["coordinator_cost"] for f in targeted_family]))
        two = cost(grid[j2], n_eval, rng_eval)
        one = cost(targeted_family[j1]["x"], n_eval, rng_eval)
        rows.append({"k": int(k),
                     "two_parameter_cost": two["coordinator_cost"], "two_parameter_se": two["cost_se"],
                     "alpha": list(grid_alpha[j2]), "alpha_minus_threshold": list(np.array(grid_alpha[j2]) - a_star),
                     "one_parameter_cost": one["coordinator_cost"], "one_parameter_se": one["cost_se"],
                     "one_parameter_mu": targeted_family[j1]["mu"]})
    return {"alpha_thresholds": a_star.tolist(), "grid_n": grid_n, "rows": rows}


# ------------------------------------------------------------------------------- E3
def cost_uncertainty_eval(scn, schemes: dict, games: dict) -> dict:
    """Known rule, unknown costs: the coordinator used nominal costs, the true game is a sample."""
    out = {}
    for name, x in schemes.items():
        res = I.realised_joint(x, games["c1"], games["c2"], k=None)
        realised = _cost_stats(scn, res["y"], games["pc"])
        as_rec = M.outcome_metrics(scn, np.broadcast_to(x, games["pc"].shape), games["pc"])
        out[name] = {
            "violation_rate": float(np.mean(res["obey"] < 1.0 - 1e-9)),
            "disobeyed_recommendations": float(1.0 - res["obey"].mean()),
            "realised": realised,
            "if_followed": as_rec,
        }
    return out


def true_game_nash(scn, games: dict, n_max: int = 2000) -> dict:
    """No coordinator: in each sampled true game the operators play its most-mixed equilibrium."""
    n = min(n_max, games["c1"].shape[0])
    ys = np.zeros((n, 3, 3))
    n_eq = np.zeros(n, dtype=int)
    for i in range(n):
        eqs = G.support_enumeration(games["c1"][i], games["c2"][i])
        n_eq[i] = len(eqs)
        p, q = max(eqs, key=lambda e: (np.count_nonzero(e[0] > 1e-9) + np.count_nonzero(e[1] > 1e-9)))
        ys[i] = G.product_table(p, q)
    m = _cost_stats(scn, ys, games["pc"][:n])
    m.update(n_games=int(n), share_with_three_equilibria=float(np.mean(n_eq == 3)),
             share_with_one_equilibrium=float(np.mean(n_eq == 1)))
    return m


def uncertainty_sweep(scn, schemes: dict, n=10000, seed=1,
                      cost_sigmas=(0.0, 0.05, 0.1, 0.2, 0.3), k_sigmas=(0.0, 0.05, 0.1, 0.2, 0.3)) -> dict:
    """Violation rate and realised cost of each rule as the coordinator's knowledge of costs and
    of the covariance degrades."""
    out = {"cost_sigmas": list(cost_sigmas), "k_sigmas": list(k_sigmas), "violation_rate": {}, "realised_cost": {}}
    for name in schemes:
        out["violation_rate"][name] = np.zeros((len(cost_sigmas), len(k_sigmas))).tolist()
        out["realised_cost"][name] = np.zeros((len(cost_sigmas), len(k_sigmas))).tolist()
    for i, sc in enumerate(cost_sigmas):
        for j, sk in enumerate(k_sigmas):
            rng = np.random.default_rng(seed + 100 * i + j)
            games = M.sample_true_games(scn, n, rng, sig_fuel=sc, sig_dis=sc, sig_val=sc, sig_k=sk)
            res = cost_uncertainty_eval(scn, schemes, games)
            for name in schemes:
                out["violation_rate"][name][i][j] = res[name]["violation_rate"]
                out["realised_cost"][name][i][j] = res[name]["realised"]["coordinator_cost"]
    return out


def clopper_pearson_upper(count: int, n: int, confidence: float) -> float:
    """Exact one-sided upper confidence bound for a binomial proportion."""
    return 1.0 if count >= n else float(stats.beta.ppf(confidence, count + 1, n - count))


def calibrate_margin(scn, families: dict, eps_list=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), n_cal=10000,
                     n_test=10000, seed=2, beta=1e-3, sig_cost=0.3, sig_k=0.1) -> dict:
    """Sample-based chance constraint over a finite list of candidate rules.

    Candidates: every rule in the margin families, plus each pure Nash equilibrium of the nominal
    game as a public rule.  For each candidate, count on n_cal sampled true games how often some
    recommendation would not be followed, and turn the count into an exact binomial upper
    confidence bound (Clopper-Pearson) at level 1 - beta / M, M being the number of candidates.
    With probability at least 1 - beta every bound holds at once (union bound).

    Two selections are reported and then tested on n_test fresh games:
    * "chance-constrained": the lowest nominal coordinator cost among candidates whose bound is
      at most eps (the usual reading of a chance constraint);
    * "realised-cost": the candidate with the lowest average realised coordinator cost on the
      calibration games, where operators who would deviate do deviate.
    """
    c1, c2, obj, _ = nominal(scn)
    cal = M.sample_true_games(scn, n_cal, np.random.default_rng(seed), sig_cost, sig_cost, sig_cost, sig_k)
    test = M.sample_true_games(scn, n_test, np.random.default_rng(seed + 1), sig_cost, sig_cost, sig_cost, sig_k)
    candidates = []
    seen_zero = False
    for name, fam in families.items():
        for f in fam:
            if f["mu"] == 0.0:
                if seen_zero:
                    continue                     # the zero-margin rule is the same in every family
                seen_zero = True
            candidates.append({"kind": f"private, {name} margin", "mu": f["mu"], "x": f["x"],
                               "objective": f["objective"]})
    for profile in G.pure_nash(c1, c2):
        x = G.point_table(profile, c1.shape)
        candidates.append({"kind": "public " + "".join(M.ACTION_LABELS[a] for a in profile), "mu": 0.0,
                           "x": x, "objective": float(np.sum(x * obj))})
    m_cand = len(candidates)
    for c in candidates:
        res = I.realised_joint(c["x"], cal["c1"], cal["c2"], k=None)
        count = int(np.sum(res["obey"] < 1 - 1e-9))
        c["cal_violation"] = count / n_cal
        c["cal_upper_bound"] = clopper_pearson_upper(count, n_cal, 1.0 - beta / m_cand)
        c["cal_realised_cost"] = _cost_stats(scn, res["y"], cal["pc"])["coordinator_cost"]

    def test_row(c):
        res = I.realised_joint(c["x"], test["c1"], test["c2"], k=None)
        realised = _cost_stats(scn, res["y"], test["pc"])
        return dict(kind=c["kind"], mu=c["mu"], nominal_cost=c["objective"],
                    nominal_delta_v=M.outcome_metrics(scn, c["x"])["delta_v"],
                    cal_violation=c["cal_violation"], cal_upper_bound=c["cal_upper_bound"],
                    test_violation=float(np.mean(res["obey"] < 1 - 1e-9)),
                    realised_cost=realised["coordinator_cost"], realised_cost_se=realised["cost_se"],
                    realised_delta_v=realised["delta_v"], realised_mean_pc=realised["mean_pc"],
                    realised_p_exceed=realised["p_exceed"],
                    realised_dual_inaction=realised["p_dual_inaction"], table=c["x"].tolist())

    out = {"n_candidates": m_cand, "n_cal": n_cal, "n_test": n_test, "beta": beta,
           "sig_cost": sig_cost, "sig_k": sig_k, "rows": []}
    for eps in eps_list:
        ok = [c for c in candidates if c["cal_upper_bound"] <= eps]
        row = {"eps": eps, "feasible": bool(ok)}
        if ok:
            row.update(test_row(min(ok, key=lambda c: c["objective"])))
        out["rows"].append(row)
    out["realised_cost_choice"] = test_row(min(candidates, key=lambda c: c["cal_realised_cost"]))
    public = [c for c in candidates if c["kind"].startswith("public")]
    out["public_choice"] = test_row(min(public, key=lambda c: c["cal_realised_cost"]))
    out["candidates"] = [{k: v for k, v in c.items() if k != "x"} for c in candidates]
    return out


def share_pc_below(scn, k, threshold, chunk: int = 10000) -> np.ndarray:
    """For each joint action, the share of covariance scales k with Pc(a; k) <= threshold.

    The collision-probability tables are built ``chunk`` scales at a time: the quadrature in
    ``pc_foster`` needs about 55 kB per scale, so all 200,000 scales at once would need about 11 GB.
    The counts are integers, so the result does not depend on the chunk size.
    """
    k = np.asarray(k, dtype=float)
    count = np.zeros((scn.n_actions, scn.n_actions), dtype=np.int64)
    for start in range(0, k.size, chunk):
        count += np.sum(M.pc_table(scn, k[start:start + chunk]) <= threshold, axis=0)
    return count / k.size


def safety_constraint_check(scn, sig_k_list=(0.1, 0.2, 0.3), delta=0.05, n=200000, seed=7,
                            chunk: int = 10000) -> dict:
    """A chance constraint on collision probability, added to the coordinator's program.

    For each joint action a, q(a) = P(Pc(a; k) <= threshold) over the uncertain covariance scale k
    (estimated from n draws).  The constraint is  sum_a x(a) * (1 - q(a)) <= delta : the
    probability, over both the coordinator's draw and the covariance scale, that the recommended
    joint action leaves Pc above the threshold is at most delta.  It is linear in x.
    The n draws are made at once (same stream as before) and evaluated ``chunk`` at a time.
    """
    c1, c2, obj, _ = nominal(scn)
    out = {"delta": delta, "threshold": scn.pc_threshold, "rows": []}
    for i, sk in enumerate(sig_k_list):
        k = np.exp(sk * np.random.default_rng(seed + i).standard_normal(n))
        q = share_pc_below(scn, k, scn.pc_threshold, chunk=chunk)
        unsafe = 1.0 - q
        row = {"sig_k": sk, "p_unsafe_by_action": unsafe.tolist()}
        for name, solver in (("private", G.solve_private), ("public", G.solve_public)):
            free = solver(c1, c2, obj)
            con = solver(c1, c2, obj, extra=[(unsafe, delta)])
            row[name] = {
                "unconstrained_cost": free["objective"],
                "unconstrained_p_unsafe": float(np.sum(free["x"] * unsafe)),
                "feasible": con["x"] is not None,
                "cost": con["objective"] if con["x"] is not None else None,
                "p_unsafe": float(np.sum(con["x"] * unsafe)) if con["x"] is not None else None,
                "table": con["x"].tolist() if con["x"] is not None else None,
            }
        out["rows"].append(row)
    return out


# ------------------------------------------------------------------------------- E4
def learning_with_cost_uncertainty(scn, families: dict, schemes: dict, ks=(15, 30, 100, 300, 1000, None),
                                   cost_sigmas=(0.1, 0.3), sig_k=0.1, n_select=20000, n_eval=50000,
                                   seed=3) -> dict:
    """Both imperfections at once: operators learned the rule from k events, and the true costs
    differ from the nominal ones the coordinator used.  ``k=None`` means the rule is known.

    For each cost-uncertainty level: the public rule, the largest uniform margin, and the best
    margin of the targeted family for each k (picked on one sample, evaluated on another).
    """
    out = {}
    for sc in cost_sigmas:
        ev = M.sample_true_games(scn, n_eval, np.random.default_rng(seed + 2), sc, sc, sc, sig_k)
        rng = np.random.default_rng(seed + 77)
        fixed = {}
        for name in ("public", "private_uniform"):
            rows = []
            for k in ks:
                res = I.realised_joint(schemes[name], ev["c1"], ev["c2"], k=k, n_rep=n_eval, rng=rng)
                m = _cost_stats(scn, res["y"], ev["pc"])
                m.update(k=k, obey=float(res["obey"].mean()))
                rows.append(m)
            fixed[name] = rows
        best = best_margin_per_k(scn, families["targeted"], ks=ks, n_select=n_select, n_eval=n_eval,
                                 seed=seed, sig_cost=sc, sig_k=sig_k)
        out[f"sigma_cost_{sc}"] = {**fixed, "private_targeted_best": best}
    return out


# ------------------------------------------------------------------------------- E5
def random_instance(rng: np.random.Generator) -> M.Scenario:
    """One draw from the instance family (declared before any result was looked at)."""
    sx = rng.uniform(80.0, 160.0)
    sz = rng.uniform(25.0, 60.0)
    log_u = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return M.Scenario(
        crossing_angle_deg=float(rng.uniform(40.0, 140.0)),
        lead_orbits=float(rng.choice([1.0, 2.0, 3.0])),
        sigma_xi=float(sx), sigma_zeta=float(sz), hbr=float(rng.uniform(3.0, 10.0)),
        miss_xi=float(rng.uniform(0.0, 1.5) * sx), miss_zeta=float(rng.uniform(0.0, 1.5) * sz),
        value=(log_u(30e6, 300e6), log_u(30e6, 300e6)),
        dv_budget=(float(rng.uniform(50.0, 200.0)), float(rng.uniform(50.0, 200.0))),
        disruption=(log_u(5e3, 100e3), log_u(5e3, 100e3)),
    )


def _saving_summary(rows, public_key: str, dv_key: str, pc_key: str, anti) -> dict:
    """Share of instances with a private gain over the public benchmark ``public_key`` and the
    size of that gain.  Instances without a finite benchmark are left out (counted separately)."""
    pub = np.array([np.nan if r[public_key] is None else r[public_key] for r in rows], dtype=float)
    prv = np.array([r["private_cost"] for r in rows], dtype=float)
    ok = np.isfinite(pub)
    sav = np.where(ok & (pub > 0), (pub - prv) / np.where(pub > 0, pub, 1.0), 0.0)[ok]
    anti = np.asarray(anti, dtype=bool)[ok]
    used = [r for r, o in zip(rows, ok) if o]
    gap = sav > 1e-6
    return {
        "n_instances_with_benchmark": int(ok.sum()),
        "share_with_private_gain": float(np.mean(gap)),
        "share_with_private_gain_among_anti_coordination": float(np.mean(gap[anti])) if anti.any() else 0.0,
        "relative_saving_quantiles_when_positive": (np.quantile(sav[gap], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
                                                    if gap.any() else []),
        "mean_relative_saving_all": float(np.mean(sav)),
        "delta_v_change_when_positive": float(np.mean([r["delta_v_private"] - r[dv_key] for r, g_ in zip(used, gap) if g_])) if gap.any() else 0.0,
        "pc_change_when_positive": float(np.mean([r["pc_private"] - r[pc_key] for r, g_ in zip(used, gap) if g_])) if gap.any() else 0.0,
    }


def private_vs_public_family(n_instances=300, seed=4) -> dict:
    """Nominal gap between the best private rule and the best public rule over random instances.

    The main benchmark is ``games.solve_public_hull``: the best mixture over all Nash equilibria,
    mixed ones included.  ``summary["pure_only_benchmark"]`` repeats the statistics against
    ``games.solve_public`` (pure equilibria only), the public rule used in the baseline
    experiments.  The extra benchmark draws no random numbers, so the instances are unchanged.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_instances):
        scn = random_instance(rng)
        c1, c2, obj, pc = nominal(scn)
        pure = G.pure_nash(c1, c2)
        hull = G.solve_public_hull(c1, c2, obj)
        pub_pure = G.solve_public(c1, c2, obj)
        prv = G.solve_private(c1, c2, obj)
        mp, mq = M.outcome_metrics(scn, prv["x"]), M.outcome_metrics(scn, hull["x"])
        mpp = M.outcome_metrics(scn, pub_pure["x"]) if pub_pure["x"] is not None else None
        best_action = np.unravel_index(np.argmin(obj), obj.shape)
        rows.append({
            "pc_no_burn": float(pc[0, 0]), "n_pure_nash": len(pure), "n_nash": len(hull["equilibria"]),
            "anti_coordination": bool(len(pure) == 2 and all(min(a) == 0 and max(a) > 0 for a in pure)),
            "coordinator_wants_a_burn": bool(best_action != (0, 0)),
            "large_burn_recommended": bool(prv["x"][2, :].sum() + prv["x"][:, 2].sum() > 1e-9),
            "public_cost": hull["objective"], "private_cost": prv["objective"],
            "saving": hull["objective"] - prv["objective"],
            "relative_saving": (hull["objective"] - prv["objective"]) / hull["objective"] if hull["objective"] > 0 else 0.0,
            "delta_v_private": mp["delta_v"], "delta_v_public": mq["delta_v"],
            "pc_private": mp["mean_pc"], "pc_public": mq["mean_pc"],
            "hull_best_is_pure": bool(np.count_nonzero(hull["x"] > 1e-9) == 1),
            "public_pure_cost": pub_pure["objective"] if mpp else None,
            "delta_v_public_pure": mpp["delta_v"] if mpp else None,
            "pc_public_pure": mpp["mean_pc"] if mpp else None,
        })
    sav = np.array([r["relative_saving"] for r in rows])
    gap = sav > 1e-6
    anti = np.array([r["anti_coordination"] for r in rows])
    summary = {
        "n_instances": n_instances,
        "share_coordinator_wants_a_burn": float(np.mean([r["coordinator_wants_a_burn"] for r in rows])),
        "share_anti_coordination": float(np.mean(anti)),
        "share_large_burn_recommended": float(np.mean([r["large_burn_recommended"] for r in rows])),
        "share_with_private_gain": float(np.mean(gap)),
        "share_with_private_gain_among_anti_coordination": float(np.mean(gap[anti])) if anti.any() else 0.0,
        "relative_saving_quantiles_when_positive": (np.quantile(sav[gap], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
                                                    if gap.any() else []),
        "mean_relative_saving_all": float(np.mean(sav)),
        "delta_v_change_when_positive": float(np.mean([r["delta_v_private"] - r["delta_v_public"] for r, g_ in zip(rows, gap) if g_])) if gap.any() else 0.0,
        "pc_change_when_positive": float(np.mean([r["pc_private"] - r["pc_public"] for r, g_ in zip(rows, gap) if g_])) if gap.any() else 0.0,
        "n_pure_nash_counts": {str(n): int(sum(1 for r in rows if r["n_pure_nash"] == n))
                               for n in sorted({r["n_pure_nash"] for r in rows})},
    }
    pure_only = _saving_summary(rows, "public_pure_cost", "delta_v_public_pure", "pc_public_pure", anti)
    pure_only["n_instances_hull_best_not_pure"] = int(sum(1 for r in rows if not r["hull_best_is_pure"]))
    pure_only["n_instances_without_pure_nash"] = int(sum(1 for r in rows if r["public_pure_cost"] is None))
    summary["pure_only_benchmark"] = pure_only
    return {"summary": summary, "rows": rows}


# ------------------------------------------------------------------------------- E6
def _obedience_feasible(A) -> bool:
    """Is there a probability table x with A @ x <= 0?  (Plain HiGHS feasibility solve.)"""
    from scipy.optimize import linprog
    A = np.asarray(A, dtype=float).reshape(-1, 9)
    if A.shape[0] == 0:
        return True
    scale = np.max(np.abs(A), axis=1, keepdims=True)          # same row scaling as games._solve_lp
    scale[scale == 0] = 1.0
    res = linprog(np.zeros(A.shape[1]), A_ub=A / scale, b_ub=np.zeros(A.shape[0]),
                  A_eq=np.ones((1, A.shape[1])), b_eq=[1.0], bounds=[(0, None)] * A.shape[1], method="highs")
    if res.status not in (0, 2):
        raise RuntimeError(f"feasibility solve ended with status {res.status}: {res.message}")
    return res.status == 0


def infeasible_subset(c1s, c2s, time_limit: float = 60.0):
    """An irreducible infeasible subset of the scenario program's obedience constraints.

    The program asks for one probability table x (x >= 0, sum 1) that satisfies the 12 obedience
    constraints of every sampled game.  If that is infeasible, a deletion filter finds a subset of
    constraints that is infeasible on its own while every proper subset is feasible:

    1. over games: drop blocks of games (halving the block size down to one game) whenever the
       rest stays infeasible;
    2. over constraints: within the remaining games, drop single constraints whenever the rest
       stays infeasible.

    Infeasibility is monotone (fewer constraints, larger feasible set), so a member kept in a
    single-deletion pass stays necessary after later deletions.  Returns None when the program
    is feasible, otherwise a dict with the members as (game, row) pairs (row indexes
    ``games.obedience_system``'s info list), and the result of re-checking both properties.
    """
    t0 = time.time()
    c1s, c2s = np.asarray(c1s, dtype=float), np.asarray(c2s, dtype=float)
    per_game = [G.obedience_system(a, b)[0] for a, b in zip(c1s, c2s)]

    def rows_of(games):
        return np.vstack([per_game[g] for g in games]) if games else np.zeros((0, 9))

    keep = list(range(len(per_game)))
    if _obedience_feasible(rows_of(keep)):
        return None
    block = max(1, len(keep) // 2)
    while True:
        i = 0
        while i < len(keep):
            if time.time() - t0 > time_limit:
                raise TimeoutError("deletion filter over games exceeded the time limit")
            trial = keep[:i] + keep[i + block:]
            if not _obedience_feasible(rows_of(trial)):
                keep = trial                     # this block is not needed for the conflict
            else:
                i += block
        if block == 1:
            break
        block = max(1, block // 2)
    members = [(g, j) for g in keep for j in range(per_game[g].shape[0])]

    def rows_for(mem):
        return np.array([per_game[g][j] for g, j in mem]).reshape(-1, 9)

    i = 0
    while i < len(members):
        if time.time() - t0 > time_limit:
            raise TimeoutError("deletion filter over constraints exceeded the time limit")
        trial = members[:i] + members[i + 1:]
        if not _obedience_feasible(rows_for(trial)):
            members = trial
        else:
            i += 1
    infeasible_alone = not _obedience_feasible(rows_for(members))
    each_removal_feasible = [bool(_obedience_feasible(rows_for(members[:i] + members[i + 1:])))
                             for i in range(len(members))]
    return {"members": members, "games": sorted({g for g, _ in members}),
            "verified_infeasible": bool(infeasible_alone),
            "verified_irreducible": bool(all(each_removal_feasible)),
            "seconds": time.time() - t0}


def describe_conflict(games: dict, subset: dict) -> dict:
    """The members of ``infeasible_subset`` with the sampled parameters of their games."""
    _, _, info = G.obedience_system(games["c1"][0], games["c2"][0])
    cons = []
    for g, j in subset["members"]:
        player, r, d = info[j]
        c = games["c1"][g] if player == 0 else games["c2"][g]
        diff = c[r, :] - c[d, :] if player == 0 else c[:, r] - c[:, d]
        cons.append({"game": int(g), "player": int(player), "recommended": int(r), "deviation": int(d),
                     "label": f"operator {player + 1} told {M.ACTION_LABELS[r]}, deviation {M.ACTION_LABELS[d]}",
                     "obey_minus_deviate_by_other_action": diff.tolist(),
                     "k": float(games["k"][g]),
                     "fuel_price": games["fuel_price"][g].tolist(),
                     "disruption": games["disruption"][g].tolist(),
                     "value": games["value"][g].tolist()})
    return {"n_constraints": len(cons), "games": subset["games"],
            "verified_infeasible": subset["verified_infeasible"],
            "verified_irreducible": subset["verified_irreducible"],
            "seconds": subset["seconds"], "constraints": cons}


def scenario_program_check(scn, eps=0.05, beta=1e-6, d=9, seed=5, sig_cost=0.3, sig_k=0.1, n_test=10000) -> dict:
    """The plain scenario program: impose obedience in every one of N sampled games."""
    obj = M.coordinator_cost_table(scn)
    out = {"eps": eps, "beta": beta, "d": d, "sig_cost": sig_cost, "sig_k": sig_k, "runs": []}
    test = M.sample_true_games(scn, n_test, np.random.default_rng(seed + 99), sig_cost, sig_cost, sig_cost, sig_k)
    for label, n in (("Campi-Garatti 2008", SC.n_campi_garatti(eps, beta, d)),
                     ("Calafiore-Campi 2006", SC.n_calafiore_campi(eps, beta, d))):
        g = M.sample_true_games(scn, n, np.random.default_rng(seed), sig_cost, sig_cost, sig_cost, sig_k)
        t0 = time.time()
        res = SC.solve_scenario_private(g["c1"], g["c2"], obj)
        run = {"bound": label, "n_scenarios": int(n), "feasible": res["x"] is not None,
               "solve_seconds": time.time() - t0, "status": str(res["status"])}
        if res["x"] is not None:
            ev = I.realised_joint(res["x"], test["c1"], test["c2"], k=None)
            run.update(table=res["x"].tolist(), nominal_cost=res["objective"],
                       test_violation=float(np.mean(ev["obey"] < 1 - 1e-9)))
        else:
            # why: an irreducible set of clashing obedience constraints (None if the plain
            # feasibility solve disagrees with the optimisation and finds the program feasible)
            sub = infeasible_subset(g["c1"], g["c2"])
            run["conflict"] = describe_conflict(g, sub) if sub is not None else None
        out["runs"].append(run)
    return out


# ------------------------------------------------------------------------------- E7
def published_pc_variant(scn, schemes: dict, n=10000, seed=8,
                         levels=((0.1, 0.05), (0.1, 0.2), (0.1, 0.3), (0.3, 0.1), (0.0, 0.3))) -> list:
    """Who knows the covariance?  Two readings of the same sampled true games.

    * "true scale": operators know the true covariance scale, so their incentives use the true
      collision probabilities (the assumption used everywhere else);
    * "published": operators act on the coordinator's published, nominal collision probabilities
      and only their cost weights differ from what the coordinator assumed.

    In both readings what actually happens is scored with the true collision probabilities.
    ``levels`` are (cost log-sd, covariance-scale log-sd) pairs.
    """
    pc_nominal = M.pc_table(scn)
    out = []
    for i, (sc, sk) in enumerate(levels):
        g = M.sample_true_games(scn, n, np.random.default_rng(seed + i), sc, sc, sc, sk)
        d1, d2 = M.cost_tables(scn, pc=pc_nominal, fuel_price=g["fuel_price"],
                               disruption=g["disruption"], value=g["value"])
        row = {"sig_cost": sc, "sig_k": sk, "rules": {}}
        for name, x in schemes.items():
            own = I.realised_joint(x, g["c1"], g["c2"], k=None)
            pub = I.realised_joint(x, d1, d2, k=None)
            row["rules"][name] = {
                "violation_true_scale": float(np.mean(own["obey"] < 1 - 1e-9)),
                "violation_published": float(np.mean(pub["obey"] < 1 - 1e-9)),
                "cost_true_scale": _cost_stats(scn, own["y"], g["pc"])["coordinator_cost"],
                "cost_published": _cost_stats(scn, pub["y"], g["pc"])["coordinator_cost"],
            }
        out.append(row)
    return out


def scenario_dict(scn: M.Scenario) -> dict:
    d = dataclasses.asdict(scn)
    d.update(period_s=scn.period, lead_time_s=scn.lead_time, gain_m_per_mps=scn.gain,
             fuel_price_usd_per_mps=list(scn.fuel_price))
    return d
