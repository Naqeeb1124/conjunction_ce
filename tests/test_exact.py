"""Exact enumeration of the learning model (inference.exact_realised_joint) and its checks."""
import math

import numpy as np
import pytest

from conjce import exact_checks as X
from conjce import games as G
from conjce import inference as I
from conjce import model as M

S = M.BASELINE
C1, C2 = M.cost_tables(S)
OBJ = M.coordinator_cost_table(S)
TIE = 1e-9 * max(np.abs(C1).max(), np.abs(C2).max())


# ------------------------------------------------------------------ simulation against exact
def test_simulation_agrees_with_exact_enumeration_on_the_baseline_rules():
    """Both conventions; uniform-margin rule at 1e-4 and targeted rule at 6e-4; k = 5 and 50;
    P(both follow), P(each follows) and every cell of y within 4 standard errors."""
    res = X.simulation_agreement(S)
    assert res["n_rep"] >= 200_000
    labels = [c["case"] for c in res["cases"]]
    assert len(labels) == 8
    for start in I.STARTS:
        for k in (5, 50):
            assert any(f"{start} k={k}" in lab and lab.startswith("uniform mu=1e-04") for lab in labels)
            assert any(f"{start} k={k}" in lab and lab.startswith("targeted mu=6e-04") for lab in labels)
    for case in res["cases"]:
        assert len(case["z"]) == 3 + 9
        assert case["max_z"] <= 4.0, case


def test_simulation_agrees_with_exact_on_a_full_support_table():
    """Nine support cells (the general composition code), both conventions, small k."""
    rng_x = np.random.default_rng(21)
    x = rng_x.dirichlet(np.ones(9)).reshape(3, 3)
    n = 200_000
    for i, start in enumerate(I.STARTS):
        for k in (1, 3):
            ex = I.exact_realised_joint(x, C1, C2, k, start=start)
            sim = I.realised_joint(x, C1, C2, k=k, n_rep=n, rng=np.random.default_rng(100 + 10 * i + k), start=start)
            for key in ("obey", "obey_1", "obey_2"):
                v = sim[key]
                assert abs(v.mean() - ex[key]) <= 4 * v.std() / math.sqrt(n) + 1e-12, (start, k, key)
            m, sd = sim["y"].mean(axis=0), sim["y"].std(axis=0)
            assert np.all(np.abs(m - ex["y"]) <= 4 * sd / math.sqrt(n) + 1e-12), (start, k)


# ------------------------------------------------------------------ hand-checkable cases
@pytest.mark.parametrize("start", I.STARTS)
def test_point_table_on_a_nash_profile_is_always_followed(start):
    profiles = G.pure_nash(C1, C2)
    assert (0, 1) in profiles and (1, 0) in profiles
    for prof in profiles:
        x = G.point_table(prof)
        for k in list(range(1, 31)) + [100, 1000]:
            ex = I.exact_realised_joint(x, C1, C2, k, start=start)
            assert ex["n_histories"] == 1
            assert ex["obey"] == pytest.approx(1.0, abs=1e-12)
            assert ex["obey_1"] == pytest.approx(1.0, abs=1e-12)
            assert ex["obey_2"] == pytest.approx(1.0, abs=1e-12)
            assert ex["y"] == pytest.approx(x, abs=1e-12)


def _br_point(cost_of, r, a):
    """Best response of an operator told r that is certain the other plays a; cost_of(d, a)."""
    e = [cost_of(d, a) for d in range(3)]
    return r if e[r] <= min(e) + TIE else int(np.argmin(e))


def _br_uniform(cost_of, r):
    e = [np.mean([cost_of(d, a) for a in range(3)]) for d in range(3)]
    return r if e[r] <= min(e) + TIE else int(np.argmin(e))


def test_uniform_start_one_event_two_cell_table_by_hand():
    """x puts p on (S, N) and 1 - p on (N, S); start="uniform", k = 1.

    The single past event is (S, N) with probability p or (N, S) with probability 1 - p.  After
    (S, N) operator 1 knows "told S -> other told N" and has never been told N (uniform belief);
    operator 2 knows "told N -> other told S" and has never been told S.  After (N, S) the roles
    of the two recommendations swap.  The new event is drawn from x independently."""
    p = 0.35
    x = np.zeros((3, 3))
    x[1, 0], x[0, 1] = p, 1 - p
    c1 = lambda d, a: C1[d, a]          # noqa: E731  operator 1 plays d, operator 2 plays a
    c2 = lambda d, a: C2[a, d]          # noqa: E731  operator 2 plays d, operator 1 plays a
    # responses after each possible past event, from the cost tables
    after_SN = {"1S": _br_point(c1, 1, 0), "1N": _br_uniform(c1, 0),
                "2N": _br_point(c2, 0, 1), "2S": _br_uniform(c2, 1)}
    after_NS = {"1N": _br_point(c1, 0, 1), "1S": _br_uniform(c1, 1),
                "2S": _br_point(c2, 1, 0), "2N": _br_uniform(c2, 0)}
    # in the baseline: seen recommendations are followed; an operator with a uniform belief burns small
    assert after_SN == {"1S": 1, "1N": 1, "2N": 0, "2S": 1}
    assert after_NS == {"1N": 0, "1S": 1, "2S": 1, "2N": 1}
    y_hand = np.zeros((3, 3))
    for w_hist, br in ((p, after_SN), (1 - p, after_NS)):
        y_hand[br["1S"], br["2N"]] += w_hist * p          # new event (S, N)
        y_hand[br["1N"], br["2S"]] += w_hist * (1 - p)    # new event (N, S)
    # which is: p^2 (S,N) + 2 p (1-p) (S,S) + (1-p)^2 (N,S)
    closed = np.zeros((3, 3))
    closed[1, 0], closed[1, 1], closed[0, 1] = p**2, 2 * p * (1 - p), (1 - p) ** 2
    assert y_hand == pytest.approx(closed, abs=1e-15)

    ex = I.exact_realised_joint(x, C1, C2, 1, start="uniform")
    assert ex["y"] == pytest.approx(closed, abs=1e-14)
    assert ex["obey"] == pytest.approx(p**2 + (1 - p) ** 2, abs=1e-14)
    assert ex["obey_1"] == pytest.approx(p * p + (1 - p), abs=1e-14)
    assert ex["obey_2"] == pytest.approx(p + (1 - p) ** 2, abs=1e-14)
    cost_hand = p**2 * OBJ[1, 0] + 2 * p * (1 - p) * OBJ[1, 1] + (1 - p) ** 2 * OBJ[0, 1]
    assert float(np.sum(ex["y"] * OBJ)) == pytest.approx(cost_hand, rel=1e-12)

    # start="observed", k = 1: no shared event, one observation per recommendation from the true
    # conditional, which is a point mass here, so both operators know the rule and follow it
    ex_obs = I.exact_realised_joint(x, C1, C2, 1, start="observed")
    assert ex_obs["obey"] == pytest.approx(1.0, abs=1e-14)
    assert ex_obs["y"] == pytest.approx(x, abs=1e-14)


@pytest.mark.parametrize("start", I.STARTS)
def test_probabilities_are_consistent(start):
    rng = np.random.default_rng(5)
    tables = [rng.dirichlet(np.ones(9)).reshape(3, 3) for _ in range(3)]
    _, families, _ = X.E.baseline_schemes(S)
    tables += [families["uniform"][5]["x"], families["targeted"][20]["x"]]
    for x in tables:
        for k in (1, 2, 4):
            ex = I.exact_realised_joint(x, C1, C2, k, start=start)
            assert ex["total_probability"] == pytest.approx(1.0, abs=1e-12)
            assert ex["y"].sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(ex["y"] >= 0)
            assert 0.0 <= ex["obey"] <= min(ex["obey_1"], ex["obey_2"]) + 1e-12
            assert ex["obey"] >= ex["obey_1"] + ex["obey_2"] - 1.0 - 1e-12
            assert max(ex["obey_1"], ex["obey_2"]) <= 1.0 + 1e-12


def test_known_rule_limit():
    """With many events the exact result approaches the known-rule outcome for a rule with slack."""
    _, families, _ = X.E.baseline_schemes(S)
    x = families["targeted"][-1]["x"]
    for start in I.STARTS:
        ex = I.exact_realised_joint(x, C1, C2, 300, start=start)
        assert ex["obey"] > 0.999
        assert ex["y"] == pytest.approx(x, abs=1e-3)


def test_history_count_and_limit():
    x = np.full((3, 3), 1 / 9)
    assert I.n_histories(x, 4, "observed") == math.comb(3 + 8, 8)
    assert I.n_histories(x, 4, "uniform") == math.comb(4 + 8, 8)
    with pytest.raises(ValueError, match="count vectors"):
        I.exact_realised_joint(x, C1, C2, 60)
    with pytest.raises(ValueError, match="start"):
        I.exact_realised_joint(x, C1, C2, 3, start="other")
    with pytest.raises(ValueError):
        I.exact_realised_joint(x, C1, C2, 0, start="observed")
    for n, s in ((7, 4), (5, 5), (12, 3), (9, 2)):
        comp = np.concatenate(list(I._composition_blocks(n, s)))
        assert comp.shape == (math.comb(n + s - 1, s - 1), s)
        assert np.all(comp.sum(axis=1) == n) and np.all(comp >= 0)
        assert len({tuple(r) for r in comp}) == len(comp)
    # four support cells and k = 300 stay within the limit
    x4 = np.zeros((3, 3))
    x4[0, 1], x4[1, 0], x4[1, 1], x4[0, 0] = 0.3, 0.3, 0.3, 0.1
    assert I.n_histories(x4, 300, "uniform") <= I.MAX_HISTORIES


# ------------------------------------------------------------------ exact_checks.run pieces
def test_two_parameter_grid_matches_the_experiment_construction():
    tables, alphas, a_star = X.two_parameter_tables(S, grid_n=5)
    assert len(tables) == 25
    for x, (a1, a2) in zip(tables, alphas):
        assert x.sum() == pytest.approx(1.0)
        assert x[1, 0] / (x[1, 0] + x[1, 1]) == pytest.approx(a1)      # P(other told N | told S), operator 1
        assert x[0, 1] / (x[0, 1] + x[1, 1]) == pytest.approx(a2)
    assert alphas[0] == pytest.approx(tuple(a_star))


def test_crossover_lists_later_failures():
    rows = [{"k": 1, "cost": 1.0}, {"k": 2, "cost": 0.5}, {"k": 3, "cost": 0.995}, {"k": 4, "cost": 0.9}]
    c = X._crossover(rows, 1.0)
    assert c["first_k"] == 2 and c["later_k_failing"] == [3]
    assert X._crossover(rows[:1], 1.0)["first_k"] is None
