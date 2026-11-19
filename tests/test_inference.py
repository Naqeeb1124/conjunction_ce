"""Tests for operators who learn the recommendation rule from k past events."""
import itertools

import numpy as np
import pytest

from conjce import games as G
from conjce import inference as I
from conjce import model as M

S = M.BASELINE
C1, C2 = M.cost_tables(S)
OBJ = M.coordinator_cost_table(S)


def _scheme(margin):
    res = G.solve_private(C1, C2, OBJ, margin=margin)
    assert res["x"] is not None
    return res["x"]


def test_known_rule_is_followed_exactly():
    for margin in (0.0, 5e-5, 1e-4):
        x = _scheme(margin)
        out = I.realised_joint(x, C1, C2, k=None)
        assert out["obey"][0] == pytest.approx(1.0)
        assert out["y"][0] == pytest.approx(x, abs=1e-12)


def test_public_pure_recommendation_needs_no_learning():
    x = G.point_table((0, 1))
    rng = np.random.default_rng(0)
    for k in (1, 2, 10):
        out = I.realised_joint(x, C1, C2, k=k, n_rep=200, rng=rng)
        assert np.all(out["obey"] == 1.0)


def test_zero_margin_rule_is_not_learnable():
    """With binding obedience constraints an operator told to burn is exactly indifferent, so the
    sign of its estimated saving is a coin flip however many events it has seen."""
    x = _scheme(0.0)
    assert x[1, 1] > 0.3                                    # the rule mixes in "both burn small"
    rng = np.random.default_rng(1)
    for k in (30, 300, 3000):
        out = I.realised_joint(x, C1, C2, k=k, n_rep=20000, rng=rng)
        told_s_1 = x[1, :].sum()
        obey_when_told_s = (out["obey_1"].mean() - (1 - told_s_1)) / told_s_1
        assert 0.35 < obey_when_told_s < 0.65
        assert out["obey"].mean() < 0.6


def test_margin_makes_the_rule_learnable():
    x = _scheme(1.5e-4)
    rng = np.random.default_rng(2)
    rates = [I.realised_joint(x, C1, C2, k=k, n_rep=20000, rng=rng)["obey"].mean() for k in (10, 100, 1000)]
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > 0.999


def test_hoeffding_bound_is_an_upper_bound_on_disobedience():
    rng = np.random.default_rng(3)
    for margin in (5e-5, 1e-4, 1.8e-4):
        x = _scheme(margin)
        for k in (5, 50, 500, 5000):
            sim = 1.0 - I.realised_joint(x, C1, C2, k=k, n_rep=40000, rng=rng)["obey"].mean()
            bound = I.hoeffding_disobedience_bound(x, C1, C2, k)
            assert sim <= bound + 4 * np.sqrt(max(sim, 1e-6) / 40000)     # allow Monte Carlo noise


def test_hoeffding_bound_is_above_the_exact_disobedience_probability():
    """For start="observed" (the convention the bound is derived for): the bound is at least the
    exact probability that some recommendation is not followed, for every rule of both margin
    families and every k from 1 to 60."""
    from conjce import experiments as E
    _, families, _ = E.baseline_schemes(S)
    n_nontrivial = 0
    for name, fam in families.items():
        for f in fam:
            for k in range(1, 61):
                ex = I.exact_realised_joint(f["x"], C1, C2, k, start="observed")
                bound = I.hoeffding_disobedience_bound(f["x"], C1, C2, k)
                assert bound >= 1.0 - ex["obey"] - 1e-12, (name, f["mu"], k, bound, 1.0 - ex["obey"])
                n_nontrivial += bound < 1.0
    assert n_nontrivial > 0                 # the comparison is not only against the trivial bound 1


def test_hoeffding_bound_closed_form_on_a_hand_made_game():
    """Chicken-like 2x2 game for operator 1, zero costs for operator 2 (nothing to gain from
    deviating, so it contributes no term).  Rule x = [[0.4, 0.3], [0.3, 0]].

    Operator 1 told 0 (probability 0.7): the other was told 0 or 1 with probabilities 4/7, 3/7.
    Deviating to 1 changes its cost by g = c1[1] - c1[0] = (-1, +2), so obeying saves
    gamma = 4/7 * (-1) + 3/7 * 2 = 2/7 on average, and g ranges over a width W = 2 - (-1) = 3.
    Operator 1 told 1 always faces 0, where obeying saves c1[0, 0] - c1[1, 0] = 1 > 0: no term.
    Bound = 0.7 * e^{-s} * (1 - 0.7 * (1 - e^{-s}))^(k - 1) with s = 2 gamma^2 / W^2 = 8/441."""
    c1 = -np.array([[6.0, 2.0], [7.0, 0.0]])
    c2 = np.zeros((2, 2))
    x = np.array([[0.4, 0.3], [0.3, 0.0]])
    gamma, width = 2.0 / 7.0, 3.0
    s = 2.0 * gamma**2 / width**2
    assert s == pytest.approx(8.0 / 441.0)
    for k in (1, 2, 10, 100, 1000):
        hand = 0.7 * np.exp(-s) * (1.0 - 0.7 * (1.0 - np.exp(-s))) ** (k - 1)
        assert I.hoeffding_disobedience_bound(x, c1, c2, k) == pytest.approx(hand, rel=1e-12)
    for k in range(1, 61):
        ex = I.exact_realised_joint(x, c1, c2, k, start="observed")
        assert ex["obey_2"] == pytest.approx(1.0)
        assert I.hoeffding_disobedience_bound(x, c1, c2, k) >= 1.0 - ex["obey"] - 1e-12


def test_uniform_start_matches_observed_stream_and_validates():
    """start="observed" is the default and start="uniform" accepts k = 0 (every belief uniform)."""
    x = _scheme(1e-4)
    a = I.realised_joint(x, C1, C2, k=7, n_rep=500, rng=np.random.default_rng(9))
    b = I.realised_joint(x, C1, C2, k=7, n_rep=500, rng=np.random.default_rng(9), start="observed")
    for key in a:
        assert np.array_equal(a[key], b[key])
    out = I.realised_joint(x, C1, C2, k=0, n_rep=10, rng=np.random.default_rng(1), start="uniform")
    ex = I.exact_realised_joint(x, C1, C2, 0, start="uniform")
    assert np.allclose(out["y"], ex["y"][None])
    with pytest.raises(ValueError):
        I.realised_joint(x, C1, C2, k=0, n_rep=10, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        I.realised_joint(x, C1, C2, k=3, n_rep=10, start="neither")


def test_hoeffding_bound_is_trivial_without_margin_and_zero_for_public_rules():
    assert I.hoeffding_disobedience_bound(_scheme(0.0), C1, C2, 1000) == 1.0
    assert I.hoeffding_disobedience_bound(G.point_table((1, 0)), C1, C2, 1) == 0.0


def _exact_realised(x, c1, c2, k):
    """Exact realised joint distribution for a 2x2 rule by enumerating every learning history."""
    m = 2
    cells = list(itertools.product(range(m), range(m)))
    probs = np.array([x[c] for c in cells])
    y = np.zeros((m, m))

    def best(belief, cost_of, r):
        exp = [sum(belief[a] * cost_of(d, a) for a in range(m)) for d in range(m)]
        return r if exp[r] <= min(exp) + 1e-12 else int(np.argmin(exp))

    for hist in itertools.product(range(len(cells)), repeat=k - 1):
        p_hist = np.prod(probs[list(hist)]) if hist else 1.0
        if p_hist == 0:
            continue
        shared = np.zeros((m, m))
        for h in hist:
            shared[cells[h]] += 1
        # initial observations: one per recommendation, for each operator
        for i1 in itertools.product(range(m), repeat=m):          # i1[r1] = a2 observed by operator 1
            for i2 in itertools.product(range(m), repeat=m):      # i2[r2] = a1 observed by operator 2
                p_init = 1.0
                for r in range(m):
                    p_init *= x[r, i1[r]] / x[r, :].sum()
                    p_init *= x[i2[r], r] / x[:, r].sum()
                if p_init == 0:
                    continue
                br1, br2 = [], []
                for r in range(m):
                    cnt = shared[r, :].copy()
                    cnt[i1[r]] += 1
                    br1.append(best(cnt / cnt.sum(), lambda d, a: c1[d, a], r))
                    cnt = shared[:, r].copy()
                    cnt[i2[r]] += 1
                    br2.append(best(cnt / cnt.sum(), lambda d, a: c2[a, d], r))
                for r1, r2 in cells:
                    y[br1[r1], br2[r2]] += p_hist * p_init * x[r1, r2]
    return y


def test_simulation_matches_exact_enumeration_on_a_small_game():
    # chicken in cost form, with a rule that has slack and full support on three profiles
    c1 = -np.array([[6.0, 2.0], [7.0, 0.0]])
    c2 = -np.array([[6.0, 7.0], [2.0, 0.0]])
    x = np.array([[0.4, 0.3], [0.3, 0.0]])
    assert np.max(G.obedience_slack(x, c1, c2)) < 0
    rng = np.random.default_rng(4)
    for k in (1, 2, 4):
        exact = _exact_realised(x, c1, c2, k)
        assert exact.sum() == pytest.approx(1.0)
        n = 200000
        sim = I.realised_joint(x, c1, c2, k=k, n_rep=n, rng=rng)["y"].mean(axis=0)
        assert sim == pytest.approx(exact, abs=5.0 / np.sqrt(n))


def test_per_replicate_costs_are_used():
    """An operator whose true maneuver cost is far above nominal does not follow a 'burn' instruction."""
    x = G.point_table((1, 0))                                 # operator 1 told to do the small burn
    c1_hi = C1.copy()
    c1_hi[1:, :] += 1.0                                        # burning costs it its whole satellite value
    stacked = np.stack([C1, c1_hi])
    out = I.realised_joint(x, stacked, C2, k=None)
    assert out["obey"].tolist() == [1.0, 0.0]
    assert out["y"][1, 0, 0] == 1.0                            # neither maneuvers in the second replicate
