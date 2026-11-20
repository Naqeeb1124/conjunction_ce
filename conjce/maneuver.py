"""Tangential-burn to miss-distance mapping for near-circular orbits.

Frame for the Clohessy-Wiltshire (Hill) equations: x radial (outward), y along-track (direction
of motion), z cross-track; n is the mean motion of the reference circular orbit.

An impulsive tangential velocity change dv at t = 0 gives

    x(t) = (2 dv / n) (1 - cos(n t))
    y(t) = (4 dv / n) sin(n t) - 3 dv t

so after a whole number of orbits the radial offset is zero and the satellite is displaced
along-track by -3 dv t: a posigrade burn raises the orbit, lengthens the period, and the
satellite falls behind.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH = 3.986004418e14   # m^3 / s^2
R_EARTH = 6378137.0         # m (equatorial radius)

__all__ = [
    "MU_EARTH", "R_EARTH", "mean_motion", "orbital_period", "circular_speed",
    "cw_tangential_impulse", "secular_along_track_shift", "bplane_gain", "bplane_axes",
    "two_body_propagate", "closest_approach",
]


def mean_motion(a: float) -> float:
    """Mean motion n = sqrt(mu / a^3) in rad/s for semi-major axis a in metres."""
    return float(np.sqrt(MU_EARTH / a**3))


def orbital_period(a: float) -> float:
    return 2.0 * np.pi / mean_motion(a)


def circular_speed(a: float) -> float:
    return float(np.sqrt(MU_EARTH / a))


def cw_tangential_impulse(dv, t, n):
    """Radial and along-track displacement at time t after a tangential impulse dv (CW solution)."""
    dv, t = np.asarray(dv, dtype=float), np.asarray(t, dtype=float)
    x = (2.0 * dv / n) * (1.0 - np.cos(n * t))
    y = (4.0 * dv / n) * np.sin(n * t) - 3.0 * dv * t
    return x, y


def secular_along_track_shift(dv, dt):
    """Along-track displacement -3 dv dt (exact in the CW model after whole orbits)."""
    return -3.0 * np.asarray(dv, dtype=float) * np.asarray(dt, dtype=float)


def bplane_gain(dt: float, crossing_angle: float) -> float:
    """Metres of encounter-plane miss change per m/s of tangential burn.

    ``crossing_angle`` is the angle between the two velocity vectors at closest approach.
    An along-track displacement d of either satellite projects onto the encounter plane as
    d cos(crossing_angle / 2), along the bisector of the two velocity directions.
    """
    return float(3.0 * dt * np.cos(0.5 * crossing_angle))


def bplane_axes(v1, v2):
    """Encounter-plane axes for two satellites with velocities v1 and v2.

    Returns unit vectors ``(xi_hat, zeta_hat, s_hat)``: xi along the bisector of the two velocity
    directions (the direction in which along-track displacements show up), s along the relative
    velocity v2 - v1 (normal to the plane), and zeta = s x xi (radial for horizontal velocities).
    """
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    t1, t2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    s_hat = (v2 - v1) / np.linalg.norm(v2 - v1)
    xi_hat = (t1 + t2) / np.linalg.norm(t1 + t2)
    xi_hat = xi_hat - np.dot(xi_hat, s_hat) * s_hat   # exact orthogonality when speeds differ slightly
    xi_hat /= np.linalg.norm(xi_hat)
    zeta_hat = np.cross(s_hat, xi_hat)
    return xi_hat, zeta_hat, s_hat


def two_body_propagate(r0, v0, t: float, rtol: float = 1e-12, atol: float = 1e-6):
    """Propagate a Keplerian (two-body) orbit for t seconds (t may be negative)."""
    y0 = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])
    if t == 0.0:
        return y0[:3].copy(), y0[3:].copy()

    def rhs(_t, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3])

    sol = solve_ivp(rhs, (0.0, t), y0, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:3, -1], sol.y[3:, -1]


def closest_approach(r1, v1, r2, v2):
    """Time offset and miss vector of closest approach for straight-line relative motion.

    With rho = r2 - r1 and w = v2 - v1 at the reference time, the distance |rho + w t| is
    smallest at t* = -(rho . w) / |w|^2, where the miss vector is rho + w t*.
    """
    rho = np.asarray(r2, dtype=float) - np.asarray(r1, dtype=float)
    w = np.asarray(v2, dtype=float) - np.asarray(v1, dtype=float)
    t_star = -float(np.dot(rho, w) / np.dot(w, w))
    return t_star, rho + w * t_star
