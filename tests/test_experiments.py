"""Tests for the helper logic behind the experiments (the experiments themselves are runs, not tests)."""
import numpy as np
import pytest
from scipy import stats

from conjce import experiments as E
from conjce import games as G
from conjce import inference as I
from conjce import model as M

S = M.BASELINE


def test_clopper_pearson_upper_bound_known_values():
    # zero violations in n trials: the bound solves (1 - u)^n = 1 - confidence
    assert E.clopper_pearson_upper(0, 100, 0.95) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-9)
    assert E.clopper_pearson_upper(100, 100, 0.95) == 1.0
    # the bound is above the observed rate and tightens with more data
    assert E.clopper_pearson_upper(5, 100, 0.99) > 0.05
    assert E.clopper_pearson_upper(50, 1000, 0.99) < E.clopper_pearson_upper(5, 100, 0.99)


def test_clopper_pearson_bound_covers_the_truth_at_the_stated_rate():
    rng = np.random.default_rng(0)
    truth, n, conf = 0.03, 400, 0.9
    counts = rng.binomial(n, truth, size=4000)
    covered = np.mean([E.clopper_pearson_upper(int(c), n, conf) >= truth for c in counts])
    assert covered >= conf - 0.01


def test_margin_families_are_private_ordered_and_feasible():
    c1, c2, obj, _ = E.nominal(S)
    fam_all = E.margin_family(c1, c2, obj, "all", E.STEP_UNIFORM)
    x0 = fam_all[0]["x"]                                                         # zero-margin private optimum
    mask = G.uncertain_belief_rows(x0, c1, c2)
    fam_bel = E.margin_family(c1, c2, obj, mask, E.STEP_TARGETED)
    for fam in (fam_all, fam_bel):
        assert fam[0]["mu"] == 0.0
        assert all(np.count_nonzero(f["x"] > 1e-9) > 1 for f in fam)            # never a single joint action
        assert all(b["objective"] >= a["objective"] - 1e-12 for a, b in zip(fam, fam[1:]))   # margin costs
        assert all(np.max(G.obedience_slack(f["x"], c1, c2)) <= 1e-12 for f in fam)
    assert fam_bel[-1]["mu"] > fam_all[-1]["mu"]                                 # targeted rows allow more slack
    assert fam_all[0]["objective"] == pytest.approx(fam_bel[0]["objective"])
    # every private rule in both families is cheaper for the coordinator than the best public rule
    public = G.solve_public(c1, c2, obj)["objective"]
    assert all(f["objective"] < public for f in fam_all + fam_bel)


def test_crossover_requires_a_real_and_lasting_improvement():
    rows = [{"k": k, "coordinator_cost": c, "cost_se": 0.001}
            for k, c in ((10, 0.9), (30, 0.2), (50, 0.0395), (70, 0.034), (100, 0.045), (300, 0.03), (1000, 0.029))]
    assert E.crossover(rows, 0.0391) == 300          # k = 70 is below, but k = 100 is above again
    assert E.crossover(rows[:4], 0.0391) == 70
    assert E.crossover(rows[:3], 0.0391) is None     # 0.0395 is not below the reference


def test_cost_uncertainty_eval_reports_no_violation_without_uncertainty():
    schemes, _, _ = E.baseline_schemes(S)
    games = M.sample_true_games(S, 50, np.random.default_rng(0), 0.0, 0.0, 0.0, 0.0)
    res = E.cost_uncertainty_eval(S, schemes, games)
    for name in ("public", "fixed_rule", "private_m0", "private_uniform", "private_targeted_3",
                 "private_targeted_6"):
        assert res[name]["violation_rate"] == 0.0
        assert res[name]["realised"]["coordinator_cost"] == pytest.approx(
            M.outcome_metrics(S, schemes[name])["coordinator_cost"])


def test_calibration_only_selects_candidates_whose_bound_is_within_eps():
    _, families, _ = E.baseline_schemes(S)
    cal = E.calibrate_margin(S, families, eps_list=(0.05, 0.3), n_cal=1500, n_test=1500, seed=11)
    for row in cal["rows"]:
        if row["feasible"]:
            assert row["cal_upper_bound"] <= row["eps"]
            assert row["cal_violation"] <= row["cal_upper_bound"]


def test_published_pc_reading_removes_the_effect_of_covariance_error_on_obedience():
    """With no cost uncertainty, operators who act on the published collision probabilities face
    exactly the nominal game, so every rule is followed whatever the true covariance scale is.
    Operators who know the true scale do deviate."""
    schemes, _, _ = E.baseline_schemes(S)
    keep = {k: schemes[k] for k in ("public", "private_m0", "private_uniform")}
    (row,) = E.published_pc_variant(S, keep, n=2000, seed=3, levels=((0.0, 0.3),))
    for name in keep:
        assert row["rules"][name]["violation_published"] == 0.0
    assert row["rules"]["private_uniform"]["violation_true_scale"] > 0.05
    # followed exactly, so the realised cost is the rule's table scored with the true probabilities
    games = M.sample_true_games(S, 2000, np.random.default_rng(3), 0.0, 0.0, 0.0, 0.3)
    expected = M.outcome_metrics(S, np.broadcast_to(keep["public"], games["pc"].shape), games["pc"])
    assert row["rules"]["public"]["cost_published"] == pytest.approx(expected["coordinator_cost"])


def test_mixed_nash_selection_picks_the_mixed_equilibrium_of_the_baseline():
    c1, c2, _, _ = E.nominal(S)
    p, q = E.select_mixed_nash(c1, c2)
    assert np.count_nonzero(p > 1e-9) == 2 and np.count_nonzero(q > 1e-9) == 2
    out = I.realised_joint(G.product_table(p, q), c1, c2, k=None)
    assert out["obey"][0] == pytest.approx(1.0)       # a Nash equilibrium is followed when it is known


# ------------------------------------------------------------------------------- added in step 1
def test_uncertain_belief_rows_of_the_baseline_are_told_small_burn_deviation_none():
    c1, c2, obj, _ = E.nominal(S)
    x0 = G.solve_private(c1, c2, obj)["x"]                 # zero-margin private optimum
    mask = G.uncertain_belief_rows(x0, c1, c2)
    _, _, info = G.obedience_system(c1, c2)
    assert mask.dtype == bool and mask.shape == (12,)
    assert sorted(tuple(r) for r, m in zip(info, mask) if m) == [(0, 1, 0), (1, 1, 0)]


def test_targeted_family_keeps_obedience_its_margin_and_stays_private():
    c1, c2, obj, _ = E.nominal(S)
    _, families, info = E.baseline_schemes(S)
    fam = families["targeted"]
    A, P, rows = G.obedience_system(c1, c2)
    targeted = [rows.index(tuple(r)) for r in info["targeted_rows"]]
    assert sorted(rows[j] for j in targeted) == [(0, 1, 0), (1, 1, 0)]
    assert len(fam) > 2
    for f in fam:
        xf = f["x"].ravel()
        assert np.max(A @ xf) <= 1e-9                                   # all 12 constraints, zero margin
        for j in targeted:                                              # the two targeted ones, margin mu
            assert (A[j] + f["mu"] * P[j]) @ xf <= 1e-9
        assert not G.reveals_other(f["x"])
        assert f["x"].sum() == pytest.approx(1.0) and np.all(f["x"] >= 0)
    assert all(b["objective"] >= a["objective"] - 1e-12 for a, b in zip(fam, fam[1:]))
    assert all(b["mu"] > a["mu"] for a, b in zip(fam, fam[1:]))
    assert fam[-1]["mu"] == info["mu_targeted_max"]


def test_crossover_needs_the_relative_gain_and_the_standard_error_test():
    one = lambda k, c, se: {"k": k, "coordinator_cost": c, "cost_se": se}
    # only 0.5 % better, with a negligible standard error: below the 1 % minimum gain
    assert E.crossover([one(10, 0.995, 1e-7)], 1.0, n_sigma=3.0, min_gain=0.01) is None
    # 5 % better, but 3 standard errors (0.06) are more than the gain
    assert E.crossover([one(10, 0.95, 0.02)], 1.0, n_sigma=3.0, min_gain=0.01) is None
    # the first k that passes both tests (and stays passing) is returned
    rows = [one(10, 0.995, 1e-7), one(20, 0.95, 0.02), one(30, 0.95, 0.001), one(50, 0.90, 0.001)]
    assert E.crossover(rows, 1.0, n_sigma=3.0, min_gain=0.01) == 30
    # both thresholds are parameters: with no minimum gain the 0.5 % row passes
    assert E.crossover([rows[0], rows[2], rows[3]], 1.0, n_sigma=3.0, min_gain=0.0) == 10
    # but a later row that fails the standard-error test resets it (the gain must last)
    assert E.crossover(rows, 1.0, n_sigma=3.0, min_gain=0.0) == 30


def test_crossover_counts_the_standard_error_and_needs_every_later_k():
    one = lambda k, c, se: {"k": k, "coordinator_cost": c, "cost_se": se}
    ref, gain, n_sigma = 2.0, 0.05, 2.0                        # threshold reference * (1 - gain) = 1.9
    # (i) cost 1.8 is below 1.9, but 1.8 + 2 * 0.06 = 1.92 is not: no crossover
    assert E.crossover([one(5, 1.8, 0.06)], ref, n_sigma=n_sigma, min_gain=gain) is None
    # the same cost with a smaller standard error (1.8 + 2 * 0.04 = 1.88) is one
    assert E.crossover([one(5, 1.8, 0.04)], ref, n_sigma=n_sigma, min_gain=gain) == 5
    # (ii) k = 5 and k = 20 pass, k = 10 fails in between: the early k is not returned
    rows = [one(5, 1.5, 0.01), one(10, 1.95, 0.01), one(20, 1.5, 0.01), one(40, 1.4, 0.01)]
    assert E.crossover(rows, ref, n_sigma=n_sigma, min_gain=gain) == 20
    # and a failure at the last k leaves no crossover at all
    assert E.crossover(rows + [one(80, 1.88, 0.02)], ref, n_sigma=n_sigma, min_gain=gain) is None


def test_calibration_picks_the_cheapest_candidate_with_a_certified_bound():
    _, families, _ = E.baseline_schemes(S)
    eps_list = (0.005, 0.05, 0.3)
    cal = E.calibrate_margin(S, families, eps_list=eps_list, n_cal=1500, n_test=1500, seed=11)
    m, n, beta = cal["n_candidates"], cal["n_cal"], cal["beta"]
    cands = cal["candidates"]
    assert len(cands) == m
    conf = 1.0 - beta / m
    for c in cands:                       # each bound is the exact Clopper-Pearson upper limit
        count = int(round(c["cal_violation"] * n))
        expected = 1.0 if count >= n else stats.beta.ppf(conf, count + 1, n - count)
        assert c["cal_upper_bound"] == pytest.approx(expected, rel=1e-12)
        assert c["cal_violation"] <= c["cal_upper_bound"]
    assert [r["eps"] for r in cal["rows"]] == list(eps_list)
    for row in cal["rows"]:
        ok = [c for c in cands if c["cal_upper_bound"] <= row["eps"]]
        assert row["feasible"] == bool(ok)
        if not ok:
            continue
        chosen = [c for c in cands if c["kind"] == row["kind"] and c["mu"] == row["mu"]]
        assert len(chosen) == 1
        assert row["cal_upper_bound"] == chosen[0]["cal_upper_bound"] <= row["eps"]
        assert row["nominal_cost"] == pytest.approx(min(c["objective"] for c in ok), rel=1e-12)
    assert any(r["feasible"] for r in cal["rows"]) and not all(r["feasible"] for r in cal["rows"])


def test_share_pc_below_does_not_depend_on_the_chunk_size():
    k = np.exp(0.3 * np.random.default_rng(0).standard_normal(2500))
    direct = np.mean(M.pc_table(S, k) <= S.pc_threshold, axis=0)
    for chunk in (7, 1000, 2500, 10000):
        assert np.array_equal(E.share_pc_below(S, k, S.pc_threshold, chunk=chunk), direct)


def test_pure_only_benchmark_is_never_cheaper_than_the_equilibrium_hull():
    res = E.private_vs_public_family(n_instances=12, seed=4)
    pure = res["summary"]["pure_only_benchmark"]
    assert pure["n_instances_with_benchmark"] == 12
    assert pure["n_instances_hull_best_not_pure"] == sum(not r["hull_best_is_pure"] for r in res["rows"])
    for r in res["rows"]:
        assert r["public_pure_cost"] >= r["public_cost"] - 1e-12        # pure equilibria are in the hull
        assert r["private_cost"] <= r["public_pure_cost"] + 1e-9
        if r["hull_best_is_pure"]:
            assert r["public_pure_cost"] == pytest.approx(r["public_cost"], rel=1e-12)
    assert pure["mean_relative_saving_all"] >= res["summary"]["mean_relative_saving_all"] - 1e-12


def _feasible_rows(rows):
    """Independent feasibility check: is there a probability table x with rows @ x <= 0?"""
    from scipy.optimize import linprog
    rows = np.asarray(rows, dtype=float).reshape(-1, 9)
    scale = np.max(np.abs(rows), axis=1, keepdims=True)
    lp = linprog(np.zeros(9), A_ub=rows / scale, b_ub=np.zeros(len(rows)), A_eq=np.ones((1, 9)), b_eq=[1.0],
                 bounds=[(0, None)] * 9, method="highs-ds")
    assert lp.status in (0, 2)
    return lp.status == 0


def test_infeasible_subset_of_the_scenario_program_is_irreducible():
    from conjce import scenario as SC
    games = M.sample_true_games(S, 608, np.random.default_rng(5), 0.3, 0.3, 0.3, 0.1)
    obj = M.coordinator_cost_table(S)
    assert SC.solve_scenario_private(games["c1"], games["c2"], obj)["x"] is None
    sub = E.infeasible_subset(games["c1"], games["c2"])
    assert sub is not None and sub["verified_infeasible"] and sub["verified_irreducible"]
    per_game = {g: G.obedience_system(games["c1"][g], games["c2"][g])[0] for g in sub["games"]}
    rows = [per_game[g][j] for g, j in sub["members"]]
    assert not _feasible_rows(rows)
    for i in range(len(rows)):
        assert _feasible_rows(rows[:i] + rows[i + 1:])
    # the games involved are infeasible together for the program itself (all 12 rows each)
    idx = sub["games"]
    assert SC.solve_scenario_private(games["c1"][idx], games["c2"][idx], obj)["x"] is None
    # and the description carries each member's game parameters and labels
    desc = E.describe_conflict(games, sub)
    assert desc["n_constraints"] == len(sub["members"])
    for c, (g, j) in zip(desc["constraints"], sub["members"]):
        assert c["game"] == g and c["k"] == games["k"][g]
        assert (c["player"], c["recommended"], c["deviation"]) == G.obedience_system(games["c1"][g], games["c2"][g])[2][j]


def test_infeasible_subset_is_none_for_a_feasible_program():
    c1, c2, _, _ = E.nominal(S)
    assert E.infeasible_subset(np.stack([c1] * 3), np.stack([c2] * 3)) is None
