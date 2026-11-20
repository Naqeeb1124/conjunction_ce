"""Tests for the scenario-approach sample sizes and the scenario program."""
import numpy as np
import pytest

from conjce import games as G
from conjce import model as M
from conjce import scenario as SC


@pytest.mark.parametrize("eps,beta,d", [(0.01, 1e-3, 8), (0.05, 1e-3, 9), (0.05, 1e-6, 9), (0.2, 1e-6, 9), (0.1, 1e-2, 3)])
def test_campi_garatti_sample_size_is_the_smallest_that_satisfies_the_bound(eps, beta, d):
    n = SC.n_campi_garatti(eps, beta, d)
    assert SC.violation_confidence(n, eps, d) <= beta
    assert SC.violation_confidence(n - 1, eps, d) > beta


def test_sample_sizes_match_an_independent_computation():
    # values computed independently with the scipy binomial CDF and bisection
    assert SC.n_campi_garatti(0.05, 1e-6, 9) == 608
    assert SC.n_campi_garatti(0.05, 1e-3, 9) == 417
    assert SC.n_campi_garatti(0.01, 1e-6, 9) == 3085
    assert SC.n_campi_garatti(0.05, 1e-6, 8) == 573
    assert SC.n_calafiore_campi(0.05, 1e-6, 9) == 1899
    assert SC.n_calafiore_campi(0.01, 1e-6, 9) == 12319
    assert SC.n_calafiore_campi(0.05, 1e-3, 9) == 1623


def test_explicit_bound_is_more_conservative_than_the_exact_one():
    for eps in (0.01, 0.05, 0.2):
        assert SC.n_calafiore_campi(eps, 1e-6, 9) > SC.n_campi_garatti(eps, 1e-6, 9)


def test_scenario_program_with_one_scenario_is_the_ordinary_program():
    s = M.BASELINE
    c1, c2 = M.cost_tables(s)
    obj = M.coordinator_cost_table(s)
    one = SC.solve_scenario_private(c1[None], c2[None], obj)
    ref = G.solve_private(c1, c2, obj)
    assert one["objective"] == pytest.approx(ref["objective"], abs=1e-9)


def test_scenario_solution_obeys_in_every_scenario_it_was_built_from():
    s = M.BASELINE
    obj = M.coordinator_cost_table(s)
    rng = np.random.default_rng(0)
    g = M.sample_true_games(s, 40, rng, sig_fuel=0.05, sig_dis=0.05, sig_val=0.05, sig_k=0.02)
    res = SC.solve_scenario_private(g["c1"], g["c2"], obj)
    assert res["x"] is not None
    for c1, c2 in zip(g["c1"], g["c2"]):
        A, _, _ = G.obedience_system(c1, c2)
        assert np.max(A @ res["x"].ravel()) <= 1e-9 * np.max(np.abs(A))
