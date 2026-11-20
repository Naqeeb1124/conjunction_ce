"""Tests for the baseline conjunction.

Several of these are design checks: they pin down the structure the baseline was built to have,
so that a later change of a parameter that
silently breaks that structure fails a test.
"""
import numpy as np
import pytest

from conjce import games as G
from conjce import model as M
from conjce.pc import pc_foster

S = M.BASELINE


def test_geometry_numbers():
    assert S.period == pytest.approx(5739.0, abs=1.0)
    assert S.gain == pytest.approx(3 * 2 * S.period * np.cos(np.pi / 4))
    assert S.gain * 0.01 == pytest.approx(243.5, abs=0.1)      # a 1 cm/s burn moves the miss by ~243 m


def test_pc_table_matches_direct_calls_and_decreases_with_total_burn():
    pc = M.pc_table(S)
    assert pc.shape == (3, 3)
    assert pc == pytest.approx(pc.T)                           # both operators have the same gain
    xi = M.miss_xi_table(S)
    assert pc[1, 2] == pytest.approx(pc_foster(xi[1, 2], S.miss_zeta, S.sigma_xi, S.sigma_zeta, S.hbr))
    total = M.burn_tables(S)[2]
    order = np.argsort(total.ravel())
    assert np.all(np.diff(pc.ravel()[order]) <= 1e-18)         # more total burn never raises Pc here


def test_pc_table_with_an_array_of_covariance_scales():
    ks = np.array([0.8, 1.0, 1.3])
    pcs = M.pc_table(S, ks)
    assert pcs.shape == (3, 3, 3)
    for i, k in enumerate(ks):
        assert pcs[i] == pytest.approx(M.pc_table(S, k), rel=1e-12)


def test_baseline_risk_levels():
    pc = M.pc_table(S)
    assert 1e-3 < pc[0, 0] < 2e-3                              # clearly actionable if nobody moves
    assert 1e-5 < pc[1, 0] < S.pc_threshold                    # one small burn: below 1e-4, not by much
    assert pc[1, 1] < 1e-7 and pc[2, 0] < 1e-7                 # two small burns or one large: negligible


def test_costs_are_fractions_of_value_and_scale_free():
    c1, c2 = M.cost_tables(S)
    pc = M.pc_table(S)
    assert c1[0, :] == pytest.approx(pc[0, :])                 # no burn: only the collision term
    assert c1[1, 0] - pc[1, 0] == pytest.approx((2e6 * 0.01 + 60e3) / 200e6)
    assert c2[0, 2] - pc[0, 2] == pytest.approx((0.8e6 * 0.03 + 16e3) / 80e6)
    # multiplying one operator's value, propellant price and disruption cost by the same factor
    # changes nothing
    fuel = np.array(S.fuel_price) * [3.0, 1.0]
    dis = np.array(S.disruption) * [3.0, 1.0]
    val = np.array(S.value) * [3.0, 1.0]
    d1, d2 = M.cost_tables(S, fuel_price=fuel, disruption=dis, value=val)
    assert d1 == pytest.approx(c1) and d2 == pytest.approx(c2)


def test_baseline_game_is_a_game_of_chicken():
    c1, c2 = M.cost_tables(S)
    assert G.pure_nash(c1, c2) == [(0, 1), (1, 0)]             # exactly one operator does a small burn
    eqs = G.support_enumeration(c1, c2)
    mixed = [(p, q) for p, q in eqs if np.count_nonzero(p > 1e-9) > 1]
    assert len(eqs) == 3 and len(mixed) == 1
    p, q = mixed[0]
    assert p[2] == pytest.approx(0.0, abs=1e-12) and q[2] == pytest.approx(0.0, abs=1e-12)
    assert 0 < p[0] < 0.5 and 0 < q[0] < 0.5                   # each waits with positive probability


def test_large_burn_is_dominated_in_the_nominal_game():
    c1, c2 = M.cost_tables(S)
    assert np.all(c1[1, :] < c1[2, :])                         # operator 1: small beats large whatever 2 does
    assert np.all(c2[:, 1] < c2[:, 2])


def test_coordinator_prefers_two_small_burns_and_values_safety_more_than_operators():
    cost = M.coordinator_cost_table(S)
    assert np.unravel_index(np.argmin(cost), cost.shape) == (1, 1)
    assert cost[1, 1] < cost[1, 0] and cost[1, 1] < cost[2, 0]
    # an operator pays for a small burn only if it removes more than its cost fraction (3e-4 to 4e-4);
    # the coordinator would spend the same 1 cm/s for 1e-5
    frac = M.maneuver_cost_fraction(S)
    assert 0.01 / S.lam < frac[0, 1] and 0.01 / S.lam < frac[1, 1]


def test_baseline_private_rule_matches_the_closed_form():
    """The solver's table against the hand-derived one."""
    c1, c2 = M.cost_tables(S)
    pc = M.pc_table(S)
    p0, p1, p2 = pc[0, 0], pc[1, 0], pc[1, 1]
    kappa = M.maneuver_cost_fraction(S)[:, 1]
    alpha = (kappa - (p1 - p2)) / ((p0 - p1) - (p1 - p2))
    both = 1.0 / (1.0 + alpha[0] / (1 - alpha[0]) + alpha[1] / (1 - alpha[1]))
    x = G.solve_private(c1, c2, M.coordinator_cost_table(S))["x"]
    assert x[1, 1] == pytest.approx(both, abs=1e-6)
    assert x[1, 0] == pytest.approx(both * alpha[0] / (1 - alpha[0]), abs=1e-6)
    assert x[0, 1] == pytest.approx(both * alpha[1] / (1 - alpha[1]), abs=1e-6)
    # the same alphas are the waiting probabilities of the mixed Nash equilibrium
    p, q = [e for e in G.support_enumeration(c1, c2) if np.count_nonzero(e[0] > 1e-9) > 1][0]
    assert q[0] == pytest.approx(alpha[0], abs=1e-9) and p[0] == pytest.approx(alpha[1], abs=1e-9)


@pytest.mark.parametrize("rows,margins", [("all", (0.0, 5e-5, 1e-4, 1.8e-4)), ("belief", (1e-4, 2e-4, 2.7e-4))])
def test_baseline_private_rules_satisfy_their_incentive_constraints(rows, margins):
    c1, c2 = M.cost_tables(S)
    A, P, _ = G.obedience_system(c1, c2)
    needs_margin = G.belief_dependent_rows(A) if rows == "belief" else np.ones(len(A), dtype=bool)
    scale = np.max(np.abs(A))
    for mu in margins:
        x = G.solve_private(c1, c2, M.coordinator_cost_table(S), margin=mu, margin_rows=rows)["x"]
        assert x is not None
        slack = A @ x.ravel()                     # <= 0 means the recommendation is obeyed
        prob = P @ x.ravel()                      # probability of the recommendation behind each row
        assert np.max(slack) <= 1e-9 * scale
        used = (prob > 1e-9) & needs_margin
        assert np.all(-slack[used] / prob[used] >= mu - 1e-9 * scale)       # conditional advantage >= margin


def test_fixed_rule_designates_the_cheaper_operator_with_a_small_burn():
    assert M.fixed_rule_profile(S) == (0, 1)
    assert M.fixed_rule_profile(S, cheaper=False) == (1, 0)


def test_sampled_true_games_have_the_right_shapes_and_medians():
    rng = np.random.default_rng(1)
    g = M.sample_true_games(S, 20000, rng)
    assert g["c1"].shape == (20000, 3, 3) and g["pc"].shape == (20000, 3, 3)
    assert np.median(g["k"]) == pytest.approx(1.0, abs=0.01)
    assert np.median(g["value"], axis=0) == pytest.approx(S.value, rel=0.02)
    assert np.median(g["disruption"], axis=0) == pytest.approx(S.disruption, rel=0.02)
    assert np.std(np.log(g["k"])) == pytest.approx(0.1, rel=0.05)
    # each sampled cost table is consistent with its own parameters
    i = 123
    c1, c2 = M.cost_tables(S, pc=g["pc"][i], fuel_price=g["fuel_price"][i],
                           disruption=g["disruption"][i], value=g["value"][i])
    assert g["c1"][i] == pytest.approx(c1) and g["c2"][i] == pytest.approx(c2)


def test_outcome_metrics_on_simple_tables():
    m = M.outcome_metrics(S, G.point_table((0, 0)))
    assert m["p_dual_inaction"] == 1.0 and m["delta_v"] == 0.0 and m["p_exceed"] == 1.0
    m = M.outcome_metrics(S, G.point_table((1, 1)))
    assert m["delta_v"] == pytest.approx(0.02) and m["p_exceed"] == 0.0
    assert m["coordinator_cost"] == pytest.approx(0.02 + S.lam * M.pc_table(S)[1, 1])


# --------------------------------------------------------------------------- added in step 1
@pytest.mark.parametrize("bad", [1.5, 2.25, 0.0, -1.0, float("nan"), float("inf")])
def test_scenario_rejects_a_lead_time_that_is_not_a_whole_number_of_orbits(bad):
    import dataclasses
    with pytest.raises(ValueError):
        M.Scenario(lead_orbits=bad)
    with pytest.raises(ValueError):
        dataclasses.replace(S, lead_orbits=bad)


@pytest.mark.parametrize("good", [1, 2.0, 3.0, 5])
def test_scenario_accepts_whole_orbits_and_the_gain_scales_with_them(good):
    scn = M.Scenario(lead_orbits=good)
    assert scn.gain == pytest.approx(S.gain * good / S.lead_orbits)


def test_sampled_true_games_without_uncertainty_equal_the_nominal_game():
    g = M.sample_true_games(S, 50, np.random.default_rng(4), 0.0, 0.0, 0.0, 0.0)
    c1, c2 = M.cost_tables(S)
    pc = M.pc_table(S)
    assert np.all(g["k"] == 1.0)
    for i in range(50):
        np.testing.assert_allclose(g["c1"][i], c1, rtol=1e-12, atol=0)
        np.testing.assert_allclose(g["c2"][i], c2, rtol=1e-12, atol=0)
        np.testing.assert_allclose(g["pc"][i], pc, rtol=1e-12, atol=0)


def test_covariance_uncertainty_alone_changes_collision_probabilities_but_not_maneuver_costs():
    n = 50
    g = M.sample_true_games(S, n, np.random.default_rng(4), 0.0, 0.0, 0.0, 0.2)
    assert len(np.unique(g["k"])) == n
    # collision-probability tables differ from sample to sample
    assert all(not np.allclose(g["pc"][i], g["pc"][0], rtol=1e-6, atol=0) for i in range(1, n))
    # maneuver costs (cost table minus collision probability) are the nominal ones in every sample
    frac = M.maneuver_cost_fraction(S)
    for name, value in (("fuel_price", S.fuel_price), ("disruption", S.disruption), ("value", S.value)):
        assert np.all(g[name] == np.asarray(value))
    for i in range(n):
        np.testing.assert_allclose(g["c1"][i] - g["pc"][i], np.repeat(frac[0][:, None], 3, axis=1), rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(g["c2"][i] - g["pc"][i], np.repeat(frac[1][None, :], 3, axis=0), rtol=1e-9, atol=1e-15)
