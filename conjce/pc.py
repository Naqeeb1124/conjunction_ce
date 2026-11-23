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
    xm, ym, sx, sy, R = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (xm, ym, sx, sy, R))
    )
    swap = sy > sx
    a_m = np.where(swap, ym, xm)
    b_m = np.abs(np.where(swap, xm, ym))
    a_s = np.where(swap, sy, sx)
    b_s = np.where(swap, sx, sy)

    u, w = _gauss_legendre(n)
    t = 0.5 * np.pi * u
    a = R[..., None] * np.sin(t)
    h = R[..., None] * np.cos(t)

    dens = np.exp(-0.5 * ((a - a_m[..., None]) / a_s[..., None]) ** 2) / (
        np.sqrt(2.0 * np.pi) * a_s[..., None]
    )
    inner = special.ndtr((h - b_m[..., None]) / b_s[..., None]) - special.ndtr(
        (-h - b_m[..., None]) / b_s[..., None]
    )
    out = 0.5 * np.pi * np.sum(w * dens * inner * h, axis=-1)
    return out if out.ndim else float(out)


def pc_foster_quad(xm: float, ym: float, sx: float, sy: float, R: float, epsrel: float = 1e-12) -> float:
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
    xm, ym, sx, sy, R = (np.asarray(v_, dtype=float) for v_ in (xm, ym, sx, sy, R))
    return R**2 / (2.0 * sx * sy) * np.exp(-0.5 * ((xm / sx) ** 2 + (ym / sy) ** 2))


def project_to_encounter_plane(r1, v1, cov1, r2, v2, cov2):
    r = np.asarray(r2, dtype=float) - np.asarray(r1, dtype=float)
    v = np.asarray(v2, dtype=float) - np.asarray(v1, dtype=float)
    y_hat = v / np.linalg.norm(v)
    r_perp = r - np.dot(r, y_hat) * y_hat
    if np.linalg.norm(r_perp) > 0:
        x_hat = r_perp / np.linalg.norm(r_perp)
    else:
        trial = np.array([1.0, 0.0, 0.0]) if abs(y_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_hat = np.cross(y_hat, trial)
        x_hat /= np.linalg.norm(x_hat)
    z_hat = np.cross(x_hat, y_hat)
    basis = np.vstack([x_hat, z_hat])
    cov = np.asarray(cov1, dtype=float) + np.asarray(cov2, dtype=float)
    return basis @ r, basis @ cov @ basis.T, basis


def pc_foster_cov(miss, cov2d, R: float, n: int = 128) -> float:
    cov2d = np.asarray(cov2d, dtype=float)
    evals, evecs = np.linalg.eigh(0.5 * (cov2d + cov2d.T))
    m = evecs.T @ np.asarray(miss, dtype=float)
    return float(pc_foster(m[0], m[1], np.sqrt(evals[0]), np.sqrt(evals[1]), R, n=n))
