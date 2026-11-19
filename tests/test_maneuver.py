"""Tests for the burn-to-miss-distance mapping.

Three levels of check:
1. the closed-form Clohessy-Wiltshire solution against a numerical integration of the CW equations;
2. the "-3 dv dt" along-track drift against a two-body (Keplerian) propagation;
3. the full mapping into the encounter plane against two-body propagation of both satellites
   through a 3D crossing encounter.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from conjce import maneuver as mv

A = mv.R_EARTH + 550e3          # 550 km circular orbit
N = mv.mean_motion(A)
T = mv.orbital_period(A)
V = mv.circular_speed(A)


def test_orbit_numbers_are_sensible():
    assert T == pytest.approx(95.65 * 60, rel=2e-3)     # about 95.6 minutes at 550 km
    assert V == pytest.approx(7585.0, rel=2e-3)         # about 7.59 km/s


def test_cw_closed_form_matches_numerical_integration_of_cw_equations():
    dv = 0.03

    def rhs(_t, s):
        x, y, xd, yd = s
        return [xd, yd, 3 * N**2 * x + 2 * N * yd, -2 * N * xd]

    for t_end in (0.3 * T, 0.5 * T, 1.0 * T, 1.7 * T, 2.0 * T):
        sol = solve_ivp(rhs, (0.0, t_end), [0.0, 0.0, 0.0, dv], method="DOP853", rtol=1e-12, atol=1e-12)
        x, y = mv.cw_tangential_impulse(dv, t_end, N)
        assert sol.y[0, -1] == pytest.approx(float(x), abs=1e-6)
        assert sol.y[1, -1] == pytest.approx(float(y), abs=1e-6)


def test_cw_after_whole_orbits_is_pure_along_track_drift():
    dv = 0.01
    for k in (1, 2, 3):
        x, y = mv.cw_tangential_impulse(dv, k * T, N)
        assert float(x) == pytest.approx(0.0, abs=1e-6)
        assert float(y) == pytest.approx(float(mv.secular_along_track_shift(dv, k * T)), abs=1e-6)


def test_cw_half_orbit_gives_maximum_radial_offset():
    dv = 0.01
    x, _ = mv.cw_tangential_impulse(dv, 0.5 * T, N)
    assert float(x) == pytest.approx(4 * dv / N, rel=1e-12)


@pytest.mark.parametrize("dv", [0.01, 0.03, -0.03])
def test_along_track_drift_against_two_body_propagation(dv):
    """Burn on a circular orbit, coast two reference periods, compare with the unburned satellite."""
    r0 = np.array([A, 0.0, 0.0])
    v0 = np.array([0.0, V, 0.0])
    dt = 2.0 * T
    r_ref, v_ref = mv.two_body_propagate(r0, v0, dt)
    r_brn, _ = mv.two_body_propagate(r0, v0 + dv * v0 / V, dt)
    along_hat = v_ref / np.linalg.norm(v_ref)
    radial_hat = r_ref / np.linalg.norm(r_ref)
    d = r_brn - r_ref
    predicted = float(mv.secular_along_track_shift(dv, dt))
    assert np.dot(d, along_hat) == pytest.approx(predicted, rel=1e-4)
    assert abs(np.dot(d, radial_hat)) < 1e-3 * abs(predicted)


def _encounter(phi, xi0, zeta0):
    """States at the nominal time of closest approach for two circular-speed satellites.

    Satellite 1 sits at radius A on the x axis.  The two velocities are horizontal and make an
    angle phi.  Satellite 2 is offset by xi0 along the bisector of the velocity directions and by
    zeta0 radially, so the nominal miss vector in the encounter plane is (xi0, zeta0).
    """
    t1 = np.array([0.0, np.cos(phi / 2), np.sin(phi / 2)])
    t2 = np.array([0.0, np.cos(phi / 2), -np.sin(phi / 2)])
    r1 = np.array([A, 0.0, 0.0])
    r2 = r1 + np.array([zeta0, xi0, 0.0])
    return r1, V * t1, r2, V * t2


def _miss_after_burns(phi, xi0, zeta0, dv1, dv2, dt):
    r1, v1, r2, v2 = _encounter(phi, xi0, zeta0)
    xi_hat, zeta_hat, _ = mv.bplane_axes(v1, v2)
    # go back to the burn epoch, burn along each satellite's own velocity, come forward again
    r1b, v1b = mv.two_body_propagate(r1, v1, -dt)
    r2b, v2b = mv.two_body_propagate(r2, v2, -dt)
    v1b = v1b + dv1 * v1b / np.linalg.norm(v1b)
    v2b = v2b + dv2 * v2b / np.linalg.norm(v2b)
    r1f, v1f = mv.two_body_propagate(r1b, v1b, dt)
    r2f, v2f = mv.two_body_propagate(r2b, v2b, dt)
    _, miss = mv.closest_approach(r1f, v1f, r2f, v2f)
    return np.array([np.dot(miss, xi_hat), np.dot(miss, zeta_hat)])


def test_bplane_axes_for_a_crossing_encounter():
    phi = np.deg2rad(90.0)
    r1, v1, r2, v2 = _encounter(phi, 92.0, 30.0)
    xi_hat, zeta_hat, s_hat = mv.bplane_axes(v1, v2)
    assert xi_hat == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert zeta_hat == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)        # radial
    assert np.dot(v1 / V, xi_hat) == pytest.approx(np.cos(phi / 2))     # projection factor
    assert np.dot(v2 / V, xi_hat) == pytest.approx(np.cos(phi / 2))
    _, miss = mv.closest_approach(r1, v1, r2, v2)
    assert [np.dot(miss, xi_hat), np.dot(miss, zeta_hat)] == pytest.approx([92.0, 30.0], abs=1e-9)


@pytest.mark.parametrize("phi_deg", [60.0, 90.0, 120.0])
@pytest.mark.parametrize("dv1,dv2", [(0.01, 0.0), (0.03, 0.0), (0.0, -0.01), (0.0, -0.03), (0.01, -0.01), (0.03, -0.03)])
def test_encounter_plane_mapping_against_two_body_propagation(phi_deg, dv1, dv2):
    """Miss-vector change predicted by 3 dt cos(phi/2) (dv1 - dv2) versus propagated truth."""
    phi = np.deg2rad(phi_deg)
    dt = 2.0 * T
    xi0, zeta0 = 92.0, 30.0
    base = _miss_after_burns(phi, xi0, zeta0, 0.0, 0.0, dt)
    assert base == pytest.approx([xi0, zeta0], abs=1e-3)       # the round trip itself is clean
    moved = _miss_after_burns(phi, xi0, zeta0, dv1, dv2, dt)
    predicted_dxi = mv.bplane_gain(dt, phi) * (dv1 - dv2)
    # observed agreement is about 1e-5 (a centimetre in 730 m); the tolerance leaves a factor 10
    assert moved[0] - base[0] == pytest.approx(predicted_dxi, rel=1e-4)
    assert abs(moved[1] - base[1]) < 1e-4 * abs(predicted_dxi)


def test_equal_burns_in_the_same_sense_cancel_in_the_encounter_plane():
    """Both satellites burning posigrade by the same amount shifts only the time of closest approach."""
    phi = np.deg2rad(90.0)
    dt = 2.0 * T
    base = _miss_after_burns(phi, 92.0, 30.0, 0.0, 0.0, dt)
    both = _miss_after_burns(phi, 92.0, 30.0, 0.02, 0.02, dt)
    single = mv.bplane_gain(dt, phi) * 0.02
    assert abs(both[0] - base[0]) < 1e-4 * single
    assert abs(both[1] - base[1]) < 1e-4 * single
