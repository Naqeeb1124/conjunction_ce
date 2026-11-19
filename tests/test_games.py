"""Tests for the equilibrium solvers and the coordinator's linear programs.

Known answers used:
* The textbook game of chicken with payoffs (6,6), (2,7), (7,2), (0,0): pure equilibria
  (Chicken, Dare) and (Dare, Chicken); mixed equilibrium with probability 2/3 on Chicken; the
  correlated equilibrium with the largest total payoff puts 1/2 on (Chicken, Chicken) and 1/4 on
  each asymmetric profile, for a total payoff of 10.5 (Aumann 1974; the worked example in most
  game-theory texts).
* Matching pennies: unique equilibrium, both players mix 1/2-1/2.
* Random 3x3 games: cross-checked against the independent ``nashpy`` library.
"""
import contextlib
import warnings

import numpy as np
import pytest

from conjce import games as G


@contextlib.contextmanager
def _quiet():
    """Silence nashpy's own 'even number of equilibria' warning; the test accounts for it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# chicken, as utilities u[a1, a2] with action 0 = Chicken, 1 = Dare
U1 = np.array([[6.0, 2.0], [7.0, 0.0]])
U2 = np.array([[6.0, 7.0], [2.0, 0.0]])
C1, C2 = -U1, -U2          # the package works with costs


def test_pure_nash_of_chicken():
    assert G.pure_nash(C1, C2) == [(0, 1), (1, 0)]


def test_pure_nash_margin_removes_equilibria_with_less_slack():
    # at (Chicken, Dare): player 1 loses 2 by switching to Dare, player 2 loses 1 by switching to Chicken
    assert (0, 1) in G.pure_nash(C1, C2, margin=1.0)
    assert (0, 1) not in G.pure_nash(C1, C2, margin=1.5)


def test_support_enumeration_of_chicken():
    eqs = G.support_enumeration(C1, C2)
    assert len(eqs) == 3
    mixed = [e for e in eqs if np.count_nonzero(e[0] > 1e-9) == 2]
    assert len(mixed) == 1
    p, q = mixed[0]
    assert p == pytest.approx([2 / 3, 1 / 3])
    assert q == pytest.approx([2 / 3, 1 / 3])


def test_support_enumeration_of_matching_pennies():
    a = np.array([[1.0, -1.0], [-1.0, 1.0]])
    eqs = G.support_enumeration(-a, a)
    assert len(eqs) == 1
    assert eqs[0][0] == pytest.approx([0.5, 0.5])
    assert eqs[0][1] == pytest.approx([0.5, 0.5])


def test_support_enumeration_returns_exact_equilibria_and_an_odd_number_of_them():
    """Checked against the definition, with no reference library: at a Nash equilibrium neither
    player can save anything by a unilateral deviation, and a non-degenerate game has an odd
    number of equilibria (Wilson's oddness theorem)."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        c1, c2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
        eqs = G.support_enumeration(c1, c2)
        assert len(eqs) % 2 == 1
        for p, q in eqs:
            assert max(G.nash_regret(p, q, c1, c2)) <= 1e-12


def test_support_enumeration_against_nashpy_on_random_games():
    """Cross-check against the independent ``nashpy`` library.

    History: the first version of this test required equal counts and failed.  In 7 of these 40
    games nashpy returns an even number of equilibria (twice it returns none), which cannot be
    the full set, and it warns about this itself; every equilibrium it misses satisfies the
    definition to 1e-16 (previous test).  So the comparison is: nothing nashpy finds is missing
    here, and the two sets coincide whenever nashpy's count is odd.
    """
    nash = pytest.importorskip("nashpy")
    rng = np.random.default_rng(7)
    full_agreement = 0
    for _ in range(40):
        c1, c2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
        mine = G.support_enumeration(c1, c2)
        with _quiet():
            theirs = list(nash.Game(-c1, -c2).support_enumeration())
        for tp, tq in theirs:
            assert any(np.allclose(p, tp, atol=1e-7) and np.allclose(q, tq, atol=1e-7) for p, q in mine)
        if len(theirs) % 2 == 1:
            assert len(mine) == len(theirs)
            full_agreement += 1
    assert full_agreement >= 30


def test_obedience_system_matches_the_definition():
    rng = np.random.default_rng(3)
    c1, c2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    x = rng.dirichlet(np.ones(9)).reshape(3, 3)
    A, P, info = G.obedience_system(c1, c2)
    assert A.shape == (12, 9) and len(info) == 12
    for row, prow, (player, r, d) in zip(A, P, info):
        if player == 0:
            direct = sum(x[r, a2] * (c1[r, a2] - c1[d, a2]) for a2 in range(3))
            prob = x[r, :].sum()
        else:
            direct = sum(x[a1, r] * (c2[a1, r] - c2[a1, d]) for a1 in range(3))
            prob = x[:, r].sum()
        assert row @ x.ravel() == pytest.approx(direct)
        assert prow @ x.ravel() == pytest.approx(prob)


def test_best_correlated_equilibrium_of_chicken_is_the_textbook_one():
    res = G.solve_private(C1, C2, objective=-(U1 + U2))
    assert res["x"] == pytest.approx(np.array([[0.5, 0.25], [0.25, 0.0]]), abs=1e-7)
    assert res["objective"] == pytest.approx(-10.5, abs=1e-7)
    assert np.max(G.obedience_slack(res["x"], C1, C2)) <= 1e-8


def test_nash_equilibria_satisfy_obedience_and_private_is_at_least_as_good():
    rng = np.random.default_rng(11)
    for _ in range(25):
        c1, c2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
        obj = rng.normal(size=(3, 3))
        best_ne = np.inf
        for p, q in G.support_enumeration(c1, c2):
            table = G.product_table(p, q)
            assert np.max(G.obedience_slack(table, c1, c2)) <= 1e-8
            best_ne = min(best_ne, float(np.sum(table * obj)))
        res = G.solve_private(c1, c2, obj)
        assert res["x"] is not None
        assert np.max(G.obedience_slack(res["x"], c1, c2)) <= 1e-7
        assert res["objective"] <= best_ne + 1e-7
        hull = G.solve_public_hull(c1, c2, obj)
        assert hull["objective"] == pytest.approx(best_ne)


def test_margin_is_respected_and_costs_the_coordinator():
    obj = -(U1 + U2)
    previous = -np.inf
    for mu in (0.0, 0.2, 0.5, 0.9):
        res = G.solve_private(C1, C2, obj, margin=mu)
        assert res["x"] is not None
        for item in G.conditional_advantages(res["x"], C1, C2):
            assert item["advantage"] >= mu - 1e-7
        assert res["objective"] >= previous - 1e-9      # a larger margin can only make things worse
        previous = res["objective"]


def test_margin_too_large_is_reported_as_infeasible():
    res = G.solve_private(C1, C2, -(U1 + U2), margin=10.0)
    assert res["x"] is None and np.isinf(res["objective"])


def test_public_recommendations_use_only_pure_nash_and_are_never_better_than_private():
    obj = -(U1 + U2)
    pub = G.solve_public(C1, C2, obj)
    assert pub["objective"] == pytest.approx(-9.0)       # (Chicken, Dare) or (Dare, Chicken): 2 + 7
    support = {tuple(ix) for ix in np.argwhere(pub["x"] > 0)}
    assert support <= set(G.pure_nash(C1, C2))
    assert G.solve_private(C1, C2, obj)["objective"] <= pub["objective"] + 1e-9
    rng = np.random.default_rng(5)
    for _ in range(25):
        c1, c2, o = rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
        pub = G.solve_public(c1, c2, o)
        prv = G.solve_private(c1, c2, o)
        if pub["x"] is not None:
            assert prv["objective"] <= pub["objective"] + 1e-7


def test_public_with_margin_reports_when_no_equilibrium_qualifies():
    assert G.solve_public(C1, C2, -(U1 + U2), margin=5.0)["x"] is None


def test_extra_linear_constraint_is_enforced():
    # cap the probability of (Chicken, Chicken) at 0.3
    cap = np.zeros((2, 2))
    cap[0, 0] = 1.0
    res = G.solve_private(C1, C2, -(U1 + U2), extra=[(cap, 0.3)])
    assert res["x"][0, 0] <= 0.3 + 1e-8
    assert np.max(G.obedience_slack(res["x"], C1, C2)) <= 1e-8


# --------------------------------------------------------------------------- added in step 1
@pytest.mark.parametrize("profile", [(0, 1), [0, 1], np.array([0, 1]), (2, 2), [1, 0]])
def test_point_table_puts_a_single_one_whatever_the_profile_type(profile):
    x = G.point_table(profile)
    assert x.shape == (3, 3)
    assert x.sum() == 1.0
    assert np.count_nonzero(x) == 1
    assert x[int(profile[0]), int(profile[1])] == 1.0


def test_reveals_other_on_hand_made_tables():
    assert G.reveals_other(G.point_table((1, 1)))
    anti = np.zeros((3, 3))
    anti[0, 1], anti[1, 0] = 0.4, 0.6          # each recommendation tells exactly what the other got
    assert G.reveals_other(anti)
    shared = np.zeros((3, 3))
    shared[1, 1], shared[1, 0] = 0.7, 0.3      # operator 1 told S: the other may have been told S or N
    assert not G.reveals_other(shared)


def _belief_dependent_by_definition(c1, c2):
    """Direct reading of the definition: constraint (player, r, d) depends on the belief when
    c_i(r, a) - c_i(d, a) does not have the same sign for every action a of the other player."""
    out = []
    for player, c in ((0, c1), (1, c2)):
        for r in range(3):
            for d in range(3):
                if d == r:
                    continue
                diff = c[r, :] - c[d, :] if player == 0 else c[:, r] - c[:, d]
                out.append((player, r, d, len(set(np.sign(diff))) > 1))
    return out


def test_belief_dependent_rows_match_the_definition_computed_from_the_costs():
    from conjce import model as M
    rng = np.random.default_rng(21)
    games = [M.cost_tables(M.BASELINE)] + [(rng.normal(size=(3, 3)), rng.normal(size=(3, 3))) for _ in range(40)]
    n_dependent = 0
    for c1, c2 in games:
        A, _, info = G.obedience_system(c1, c2)
        expected = _belief_dependent_by_definition(c1, c2)
        assert [e[:3] for e in expected] == [tuple(i) for i in info]      # same row order
        assert G.belief_dependent_rows(A).tolist() == [e[3] for e in expected]
        n_dependent += sum(e[3] for e in expected)
    assert 0 < n_dependent < 12 * len(games)       # both kinds of row occur in the test set


def test_solve_public_picks_the_cheaper_of_two_pure_equilibria():
    assert G.pure_nash(C1, C2) == [(0, 1), (1, 0)]       # chicken: two pure equilibria
    # (Chicken, Chicken) is cheapest overall but not an equilibrium, so it must not be chosen
    obj = np.array([[-10.0, 1.0], [2.0, -20.0]])
    res = G.solve_public(C1, C2, obj)
    assert res["x"] == pytest.approx(G.point_table((0, 1), (2, 2)))
    assert res["objective"] == pytest.approx(1.0)
    res = G.solve_public(C1, C2, -obj)
    assert res["x"] == pytest.approx(G.point_table((1, 0), (2, 2)))
    assert res["objective"] == pytest.approx(-2.0)
