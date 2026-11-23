from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0

__all__ = [
    "MU_EARTH", "R_EARTH", "mean_motion", "orbital_period", "circular_speed",
    "cw_tangential_impulse", "secular_along_track_shift", "bplane_gain", "bplane_axes",
    "two_body_propagate", "closest_approach",
]


def mean_motion(a: float) -> float:
    return float(np.sqrt(MU_EARTH / a**3))


def orbital_period(a: float) -> float:
    return 2.0 * np.pi / mean_motion(a)


def circular_speed(a: float) -> float:
    return float(np.sqrt(MU_EARTH / a))


def cw_tangential_impulse(dv, t, n):
    dv, t = np.asarray(dv, dtype=float), np.asarray(t, dtype=float)
    x = (2.0 * dv / n) * (1.0 - np.cos(n * t))
    y = (4.0 * dv / n) * np.sin(n * t) - 3.0 * dv * t
    return x, y


def secular_along_track_shift(dv, dt):
    return -3.0 * np.asarray(dv, dtype=float) * np.asarray(dt, dtype=float)


def bplane_gain(dt: float, crossing_angle: float) -> float:
    return float(3.0 * dt * np.cos(0.5 * crossing_angle))


def bplane_axes(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    t1, t2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    s_hat = (v2 - v1) / np.linalg.norm(v2 - v1)
    xi_hat = (t1 + t2) / np.linalg.norm(t1 + t2)
    xi_hat = xi_hat - np.dot(xi_hat, s_hat) * s_hat
    xi_hat /= np.linalg.norm(xi_hat)
    zeta_hat = np.cross(s_hat, xi_hat)
    return xi_hat, zeta_hat, s_hat


def two_body_propagate(r0, v0, t: float, rtol: float = 1e-12, atol: float = 1e-6):
    y0 = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])
    if t == 0.0:
        return y0[:3].copy(), y0[3:].copy()

    def rhs(_t, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3])

    sol = solve_ivp(rhs, (0.0, t), y0, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:3, -1], sol.y[3:, -1]


def closest_approach(r1, v1, r2, v2):
    rho = np.asarray(r2, dtype=float) - np.asarray(r1, dtype=float)
    w = np.asarray(v2, dtype=float) - np.asarray(v1, dtype=float)
    t_star = -float(np.dot(rho, w) / np.dot(w, w))
    return t_star, rho + w * t_star
