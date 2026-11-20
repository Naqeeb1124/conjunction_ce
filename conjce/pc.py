"""Two-dimensional (short-encounter) collision probability in the encounter plane.

Model (Foster and Estes 1992; Akella and Alfriend 2000; Alfano 2005; Chan 2008):
the relative motion during the encounter is a straight line, the relative position
uncertainty is Gaussian and constant during the encounter, and a collision happens when
the relative position passes within the combined hard-body radius R.  Integrating out the
direction of relative velocity leaves a 2D integral in the encounter plane:

    Pc = 1 / (2 pi sx sy) * double integral over x^2 + y^2 <= R^2 of
             exp( -(x - xm)^2 / (2 sx^2) - (y - ym)^2 / (2 sy^2) ) dx dy

with the plane axes rotated onto the principal axes of the projected combined covariance,
(sx, sy) its standard deviations and (xm, ym) the miss vector.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, special

__all__ = [
    "pc_foster",
    "pc_foster_quad",
    "pc_chan",
    "pc_small_hbr",
    "project_to_encounter_plane",
    "pc_foster_cov",
]

_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def pc_foster(xm, ym, sx, sy, R, n: int = 128):
    """Foster's 2D collision probability, vectorised (inputs broadcast).

    The inner integral (across the chord of the disc) is done in closed form with the normal
    CDF.  The outer integral runs along the axis with the larger sigma, uses the substitution
    a = R sin(t), and is evaluated with n-point Gauss-Legendre quadrature in t.

    Intended for R not much larger than the larger sigma, which covers this project and the
    Chan / CSM / CDM reference cases.  ``pc_foster_quad`` is the adaptive reference.
    """
    xm, ym, sx, sy, R = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (xm, ym, sx, sy, R))
    )
    swap = sy > sx
    a_m = np.where(swap, ym, xm)          # mean along the numerically integrated axis
    b_m = np.abs(np.where(swap, xm, ym))  # mean across it (only |.| matters, by symmetry)
    a_s = np.where(swap, sy, sx)
    b_s = np.where(swap, sx, sy)

    u, w = _gauss_legendre(n)             # nodes and weights on [-1, 1]
    t = 0.5 * np.pi * u                   # t in [-pi/2, pi/2]
    a = R[..., None] * np.sin(t)          # position along the outer axis
    h = R[..., None] * np.cos(t)          # half length of the chord at that position

    dens = np.exp(-0.5 * ((a - a_m[..., None]) / a_s[..., None]) ** 2) / (
        np.sqrt(2.0 * np.pi) * a_s[..., None]
    )
    inner = special.ndtr((h - b_m[..., None]) / b_s[..., None]) - special.ndtr(
        (-h - b_m[..., None]) / b_s[..., None]
    )
    # da = R cos(t) dt = h dt, and dt = (pi/2) du
    out = 0.5 * np.pi * np.sum(w * dens * inner * h, axis=-1)
    return out if out.ndim else float(out)


def pc_foster_quad(xm: float, ym: float, sx: float, sy: float, R: float, epsrel: float = 1e-12) -> float:
    """Adaptive reference for Foster's integral (scalar inputs).

    Same reduction as ``pc_foster`` but with ``scipy.integrate.quad``, so it stays accurate when
    the hard-body radius is much larger than the smaller sigma (the Alfano stress cases).
    """
    if sy > sx:
        a_m, b_m, a_s, b_s = ym, abs(xm), sy, sx
    else:
        a_m, b_m, a_s, b_s = xm, abs(ym), sx, sy

    def integrand(t: float) -> float:
        a = R * np.sin(t)
        h = R * np.cos(t)
        dens = np.exp(-0.5 * ((a - a_m) / a_s) ** 2) / (np.sqrt(2.0 * np.pi) * a_s)
        inner = special.ndtr((h - b_m) / b_s) - special.ndtr((-h - b_m) / b_s)
        return dens * inner * h

    # break points: where the density peaks, and where the chord half-length equals |b_m|
    pts = []
    if abs(a_m) < R:
        pts.append(float(np.arcsin(a_m / R)))
    if b_m < R:
        tb = float(np.arccos(b_m / R))
        pts.extend([tb, -tb])
    val, _ = integrate.quad(
        integrand, -0.5 * np.pi, 0.5 * np.pi, epsabs=0.0, epsrel=epsrel, limit=2000,
        points=sorted(set(pts)) or None,
    )
    return float(val)


def pc_chan(xm, ym, sx, sy, R, max_terms: int = 20000):
    """Chan's series (equivalent-area approximation), vectorised.

        u = R^2 / (sx sy),   v = xm^2 / sx^2 + ym^2 / sy^2
        Pc = exp(-v/2) * sum_{m>=0} (v/2)^m / m! * [ 1 - exp(-u/2) * sum_{k=0}^{m} (u/2)^k / k! ]

    The square bracket is the regularised lower incomplete gamma function P(m+1, u/2); it is
    evaluated with ``scipy.special.gammainc`` because the literal difference loses all digits
    for small u.  The series equals the CDF of a noncentral chi-square variable with 2 degrees
    of freedom and noncentrality v, evaluated at u.  Exact when sx == sy.
    """
    xm, ym, sx, sy, R = np.broadcast_arrays(
        *(np.asarray(v_, dtype=float) for v_ in (xm, ym, sx, sy, R))
    )
    u = R**2 / (sx * sy)
    v = (xm / sx) ** 2 + (ym / sy) ** 2
    half_v = 0.5 * v
    vmax = float(np.max(half_v)) if half_v.size else 0.0
    n_terms = int(min(max_terms, np.ceil(vmax + 12.0 * np.sqrt(vmax) + 40.0)))
    total = np.zeros_like(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_half_v = np.where(half_v > 0, np.log(np.where(half_v > 0, half_v, 1.0)), -np.inf)
    for m in range(n_terms + 1):
        if m == 0:
            log_w = -half_v
        else:
            log_w = m * log_half_v - half_v - special.gammaln(m + 1.0)
        total = total + np.exp(log_w) * special.gammainc(m + 1.0, 0.5 * u)
    return total if total.ndim else float(total)


def pc_small_hbr(xm, ym, sx, sy, R):
    """Small hard-body approximation: density at the origin times the disc area.

        Pc ~ R^2 / (2 sx sy) * exp( -(xm^2 / sx^2 + ym^2 / sy^2) / 2 )

    Valid when R is much smaller than both sigmas.  Used for intuition and as a sanity check.
    """
    xm, ym, sx, sy, R = (np.asarray(v_, dtype=float) for v_ in (xm, ym, sx, sy, R))
    return R**2 / (2.0 * sx * sy) * np.exp(-0.5 * ((xm / sx) ** 2 + (ym / sy) ** 2))


def project_to_encounter_plane(r1, v1, cov1, r2, v2, cov2):
    """Project a 3D conjunction onto the encounter plane.

    Returns ``(miss, cov2d, basis)``: the 2-vector miss vector, the 2x2 projected combined
    position covariance, and the 2x3 matrix whose rows span the plane.  The plane is
    perpendicular to the relative velocity; its first axis is along the miss vector's
    component in the plane.
    """
    r = np.asarray(r2, dtype=float) - np.asarray(r1, dtype=float)
    v = np.asarray(v2, dtype=float) - np.asarray(v1, dtype=float)
    y_hat = v / np.linalg.norm(v)
    r_perp = r - np.dot(r, y_hat) * y_hat
    if np.linalg.norm(r_perp) > 0:
        x_hat = r_perp / np.linalg.norm(r_perp)
    else:  # zero miss: any direction in the plane will do
        trial = np.array([1.0, 0.0, 0.0]) if abs(y_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_hat = np.cross(y_hat, trial)
        x_hat /= np.linalg.norm(x_hat)
    z_hat = np.cross(x_hat, y_hat)
    basis = np.vstack([x_hat, z_hat])
    cov = np.asarray(cov1, dtype=float) + np.asarray(cov2, dtype=float)
    return basis @ r, basis @ cov @ basis.T, basis


def pc_foster_cov(miss, cov2d, R: float, n: int = 128) -> float:
    """Foster's Pc for a general (non-diagonal) 2x2 covariance in the encounter plane."""
    cov2d = np.asarray(cov2d, dtype=float)
    evals, evecs = np.linalg.eigh(0.5 * (cov2d + cov2d.T))
    m = evecs.T @ np.asarray(miss, dtype=float)
    return float(pc_foster(m[0], m[1], np.sqrt(evals[0]), np.sqrt(evals[1]), R, n=n))
