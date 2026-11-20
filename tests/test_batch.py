"""Tests for the batch model (many conjunctions at once), conjce/batch.py.

The checks of propositions P1 to P5 of the scale-up run here at small sizes: the
full joint program up to K = 4, the exact-one joint program up to K = 6.  run_scaleup.py repeats
them at every K within the size cap.
"""
import dataclasses

import numpy as np
import pytest

from conjce import batch as B
from conjce import games as G
from conjce import model as M

TOL = B.AGREEMENT_TOL
FEAS = B.FEASIBILITY_TOL


@pytest.fixture(scope="module")
def master():
    return B.generate_batch(B.MASTER_SEED, B.MASTER_SIZE)


@pytest.fixture(scope="module")
def conjs(master):
    return master.conjunctions


# Generator
def test_fleets_and_pair_count():
    assert np.bincount(B.OPERATOR_OF).tolist() == [12, 8, 6, 4]
    assert [int(B.OPERATOR_OF[i]) for i in (0, 11, 12, 19, 20, 25, 26, 29)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(B.PAIRS) == 320
    assert len(set(B.PAIRS)) == 320
    assert all(B.OPERATOR_OF[i] != B.OPERATOR_OF[j] for i, j in B.PAIRS)


def test_pairs_distinct_between_operators_and_times_in_window(conjs):
    pairs = [frozenset(c.sats) for c in conjs]
    assert len(set(pairs)) == len(conjs) == B.MASTER_SIZE
    for c in conjs:
        assert c.op1 != c.op2
        assert c.op1 == B.OPERATOR_OF[c.sat1] and c.op2 == B.OPERATOR_OF[c.sat2]
        assert 0 <= c.sat1 < 30 and 0 <= c.sat2 < 30
        assert 0.0 <= c.time_h <= B.WINDOW_HOURS
    # the coin puts the lower-numbered satellite on either side
    assert any(c.sat1 < c.sat2 for c in conjs) and any(c.sat1 > c.sat2 for c in conjs)


def _key(c):
    return (c.sat1, c.sat2, c.time_h, c.scn, c.stable, c.n_draws)


def test_same_seed_gives_the_same_batch(master):
    again = B.generate_batch(B.MASTER_SEED, B.MASTER_SIZE)
    assert [_key(c) for c in again.conjunctions] == [_key(c) for c in master.conjunctions]
    for a, b in zip(again.conjunctions, master.conjunctions):
        assert np.array_equal(a.c1, b.c1) and np.array_equal(a.c2, b.c2) and np.array_equal(a.obj, b.obj)
    other = B.generate_batch(B.MASTER_SEED + 1, 5)
    assert [_key(c) for c in other.conjunctions] != [_key(c) for c in master.conjunctions[:5]]


def test_batches_are_nested(master):
    for K in (1, 7, 20):
        small = B.generate_batch(B.MASTER_SEED, K)
        assert [_key(c) for c in small.conjunctions] == [_key(c) for c in master.conjunctions[:K]]


def test_kept_draws_have_a_stable_designation_and_rejected_ones_do_not(master):
    assert all(c.stable for c in master.conjunctions)
    assert master.n_draws == B.MASTER_SIZE + len(master.rejected)
    for _, scn in master.rejected:
        c1, c2 = M.cost_tables(scn)
        assert B.stable_designations(c1, c2) == ()


def test_stable_designations_are_exact_one_pure_equilibria(conjs):
    for c in conjs:
        nash = G.pure_nash(c.c1, c.c2)
        assert set(c.stable) == {e for e in nash if e in B.EXACT_ONE}
        for e in c.stable:
            assert (e[0] == 0) != (e[1] == 0)          # exactly one side burns


# P1
@pytest.mark.parametrize("K", B.P1_KS)
def test_p1_single_slot_deviations_are_enough(conjs, K):
    r = B.check_p1(conjs[:K])
    assert r["max_slots_of_an_operator"] >= 2          # the check is not vacuous
    assert r["rows_all_deviations"] > r["rows_single"]
    assert r["rel_diff"] <= TOL
    assert r["violation_of_all_deviation_rows"] <= FEAS


def _same_operator_pair(conjs, n=2):
    """n master conjunctions between the same two operators (an operator with several slots
    against the same other operator)."""
    by_pair = {}
    for c in conjs:
        by_pair.setdefault(frozenset((c.op1, c.op2)), []).append(c)
    group = max(by_pair.values(), key=len)
    assert len(group) >= n
    return group[:n]


def test_p1_and_p2_with_several_slots_against_the_same_operator(conjs):
    sub = _same_operator_pair(conjs, 3)
    r1 = B.check_p1(sub)
    assert r1["rel_diff"] <= TOL and r1["violation_of_all_deviation_rows"] <= FEAS
    r2 = B.check_p2(sub)
    assert r2["rel_diff"] <= TOL
    assert r2["marginal_violation"] <= FEAS and r2["product_violation"] <= FEAS


# P2
@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_p2_decomposition(conjs, K):
    r = B.check_p2(conjs[:K])
    assert r["rel_diff"] <= TOL
    assert r["marginal_violation"] <= FEAS
    assert r["product_violation"] <= FEAS
    assert r["product_objective_rel_diff"] <= TOL


def test_per_conjunction_program_matches_solve_private_at_k50(conjs):
    per = B.per_conjunction_program(conjs)
    total = sum(G.solve_private(c.c1, c.c2, c.obj)["objective"] for c in conjs)
    assert B.rel_diff(per["optimum"], total) <= TOL
    assert per["variables"] == 9 * len(conjs)


def test_joint_program_sizes(conjs):
    a = B.joint_program(conjs[:3], "full")
    b = B.joint_program(conjs[:3], "exact_one")
    assert a["variables"] == 9 ** 3 and b["variables"] == 4 ** 3
    # four nonzeros per slot and profile (two deviations, two slots per conjunction) plus the simplex row
    assert a["nonzeros"] == 9 ** 3 * (4 * 3 + 1)
    assert b["nonzeros"] == 4 ** 3 * (4 * 3 + 1)


# P3
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6])
def test_p3_exact_one_optimum_is_the_closed_form(conjs, K):
    r = B.check_p3(conjs[:K])
    assert r["rel_diff"] <= TOL
    assert r["mass_outside_stable_products"] <= FEAS


def test_p3_every_mixture_of_stable_products_is_obedient(conjs):
    sub = conjs[:5]
    ex = B.joint_program(sub, "exact_one")
    mask = B.stable_mask(sub, ex["profiles"])
    rng = np.random.default_rng(0)
    x = np.where(mask, rng.random(mask.size), 0.0)
    x /= x.sum()
    assert B.max_violation(ex["A"], x) <= FEAS
    # and a profile outside the stable products is not obedient on its own
    bad = np.nonzero(~mask)[0][0]
    assert B.max_violation(ex["A"], np.eye(mask.size)[bad]) > 1e-6


def test_p3_exact_one_program_is_infeasible_with_an_empty_E_k(master):
    r = B.check_p3_infeasible(master)
    assert r["empty_E_k"]
    assert r["infeasible"]
    assert r["closed_form"] == np.inf


def test_hand_built_game_without_a_stable_designation():
    # the miss is far outside the ellipse, so no burn is worth it
    c1, c2 = M.cost_tables(M.Scenario(miss_xi=2000.0, miss_zeta=2000.0))
    assert B.stable_designations(c1, c2) == ()
    assert G.pure_nash(c1, c2) == [(0, 0)]


# P4 and burden
def test_free_conjunctions_have_one_designation_per_side(conjs):
    for c in conjs:
        info = B.designation_info(c)
        if info["free"]:
            assert info["both_sides"] and len(info["optimal"]) == 2
            assert {B.burner(e) for e in info["optimal"]} == {0, 1}
        else:
            assert len(info["optimal"]) == 1
        assert info["cost"] == pytest.approx(B.closed_form([c]), rel=1e-12)


def _mirror(scn):
    """The same encounter with the two operators' cost data swapped (the geometry is symmetric:
    the miss depends on the sum of the two burns)."""
    return dataclasses.replace(scn, value=scn.value[::-1], dv_budget=scn.dv_budget[::-1],
                               disruption=scn.disruption[::-1])


def test_fixed_rule_side_agrees_with_the_project_rule(conjs):
    # hand-made: side 1's burn costs about 1 kUSD in disruption, side 2's about 1 billion USD, with the
    # same propellant price, so side 1's smallest sufficient burn is clearly the cheaper one
    cheap_first = dataclasses.replace(M.BASELINE, value=(100e6, 100e6), dv_budget=(100.0, 100.0),
                                      disruption=(1e3, 1e9))
    assert B.fixed_rule_side(B.make_conjunction(0, 0, 12, 1.0, cheap_first)) == 0
    assert B.fixed_rule_side(B.make_conjunction(0, 0, 12, 1.0, _mirror(cheap_first))) == 1
    # an instance and its mirror image designate opposite sides
    for c in conjs[:10]:
        m = B.make_conjunction(c.index, c.sat1, c.sat2, c.time_h, _mirror(c.scn))
        assert B.fixed_rule_side(m) == 1 - B.fixed_rule_side(c)


@pytest.mark.parametrize("K", B.P5_KS)
def test_p5_burden_needs_only_marginals(conjs, K):
    a = B.balanced_randomised(conjs[:K])
    b = B.balanced_randomised_joint(conjs[:K])
    assert a["variables"] == a["n_free"] + 1
    assert B.rel_diff(a["max_expected_burns"], b["max_expected_burns"]) <= TOL


@pytest.mark.parametrize("K", [6, 8, 10])
def test_deterministic_balance_matches_brute_force(conjs, K):
    det = B.balanced_deterministic(conjs[:K])
    assert det["n_free"] <= B.BRUTE_FORCE_MAX_FREE
    assert det["max_burns"] == B.brute_force_max_burns(conjs[:K])


def test_deterministic_balance_is_the_randomised_value_rounded_up(conjs):
    batches = [conjs[:20], conjs] + [B.generate_batch(s, B.MASTER_SIZE).conjunctions for s in B.FURTHER_SEEDS]
    for i, batch in enumerate(batches):
        rnd = B.balanced_randomised(batch)["max_expected_burns"]
        det = B.balanced_deterministic(batch)["max_burns"]
        assert det == int(np.ceil(rnd - 1e-9)), (i, rnd, det)


def test_burden_variants_keep_one_burn_per_conjunction(conjs):
    for K in (10, 50):
        b = B.burden(conjs[:K])
        assert b["burns_fixed_tie_break"].sum() == K
        assert b["burns_balanced_deterministic"].sum() == K
        assert b["expected_burns_balanced_randomised"].sum() == pytest.approx(K)
        # the three variants are ordered: randomised <= deterministic <= fixed tie-break
        assert b["max_expected_burns_balanced_randomised"] <= b["max_burns_balanced_deterministic"] + 1e-9
        assert b["max_burns_balanced_deterministic"] <= b["max_burns_fixed_tie_break"]


def test_carry_over_on_a_hand_made_batch(conjs):
    # a conjunction with a single cost-optimal designation, copied onto three pairs
    c0 = next(c for c in conjs if not B.designation_info(c)["free"])
    side = B.burner(B.designation_info(c0)["optimal"][0])

    def place(index, burner_sat, other_sat, t):
        s1, s2 = (burner_sat, other_sat) if side == 0 else (other_sat, burner_sat)
        return B.make_conjunction(index, s1, s2, t, c0.scn)

    batch = [place(0, 0, 12, 1.0),        # satellite 0 burns first
             place(1, 0, 20, 2.0),        # satellite 0 again: after an earlier burn
             place(2, 13, 21, 3.0)]       # no satellite in common with the others
    r = B.carry_over(batch)
    assert r["after_earlier_burn_fixed_tie_break"] == 1
    assert r["after_earlier_burn_balanced_deterministic"] == 1
    assert r["sharing_a_satellite"] == 2


# The no-tie assumption of P3
# Example: side 1 has g(S, N) = g(L, N) = 0 <= g(N, N) = 1.
TIE_C1 = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
TIE_C2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0]])
TIE_J = np.zeros((3, 3))
TIE_J[1, 0], TIE_J[2, 0], TIE_J[0, 1], TIE_J[0, 2] = 2.0, 1.0, 3.0, 3.0      # SN, LN, NS, NL


def _hand_conjunction(c1, c2, obj):
    return B.Conjunction(index=0, sat1=0, sat2=12, time_h=0.0, scn=M.BASELINE, c1=c1, c2=c2, obj=obj,
                         stable=B.stable_designations(c1, c2), n_draws=1)


def test_tie_example_exact_one_program_below_closed_form():
    conj = _hand_conjunction(TIE_C1, TIE_C2, TIE_J)
    assert conj.stable == ((1, 0),)                      # E = {SN}
    # side 1: c1(S,N) = c1(L,N) = 0 <= c1(N,N) = 1; side 2 also ties: c2(N,S) = c2(N,L) = 0 <= c2(N,N) = 1
    assert B.burn_size_tie_slots(conj) == 2
    ex = B.joint_program([conj], "exact_one")
    assert ex["status"] == 0
    assert ex["optimum"] == pytest.approx(1.5, rel=1e-9)
    assert B.closed_form([conj]) == pytest.approx(2.0, rel=1e-12)


def test_tie_variant_with_empty_E_k_is_still_feasible():
    c2 = TIE_C2.copy()
    c2[1, :] = [1.0, 2.0, 0.0]                           # side 2 now prefers L against S
    conj = _hand_conjunction(TIE_C1, c2, TIE_J)
    assert conj.stable == ()
    assert B.closed_form([conj]) == np.inf
    ex = B.joint_program([conj], "exact_one")
    assert ex["status"] == 0                             # feasible despite the empty E_k


def test_master_batch_has_no_burn_size_tie(conjs):
    assert B.n_slots_with_burn_size_tie(conjs) == 0


# Spread over batches
def test_direct_best_public_equals_solve_public(conjs):
    for c in conjs:
        cell, value = B.best_public_direct(c)
        ref = G.solve_public(c.c1, c.c2, c.obj)
        assert B.rel_diff(value, ref["objective"]) <= 1e-9
        assert ref["x"][cell] == 1.0


@pytest.fixture(scope="module")
def batch_1000():
    return B.generate_batch(B.FURTHER_SEEDS[0], B.MASTER_SIZE).conjunctions


def test_fast_totals_equal_the_master_code_path(batch_1000):
    fast = B.totals_fast(batch_1000)
    private = [G.solve_private(c.c1, c.c2, c.obj)["objective"] for c in batch_1000]
    public = [G.solve_public(c.c1, c.c2, c.obj)["objective"] for c in batch_1000]
    slow = B._totals(batch_1000, private, public)
    for key in ("sum_private", "sum_public", "sum_cheapest_stable"):
        assert B.rel_diff(fast[key], slow[key]) <= TOL


def test_private_public_stable_are_ordered_per_conjunction(batch_1000):
    fast = B.totals_fast(batch_1000)
    for p, u, s in zip(fast["private"], fast["public"], fast["stable"]):
        assert p <= u * (1 + 1e-9)
        assert u <= s * (1 + 1e-9)


def test_histogram_counts_every_satellite():
    assert B.histogram(np.array([0, 2, 2, 1, 0, 0])) == [3, 1, 2]
