"""Tests for the 2D collision probability.

Reference values
----------------
The encounter-plane test cases below (sigma_x, sigma_y, R, x_m, y_m in metres) are the benchmark
set of Serra, Arzelier, Joldes, Lasserre, Rondepierre, Salvy, "Fast and Accurate Computation of
Orbital Collision Probability for Short-Term Encounters", J. Guidance, Control, and Dynamics 39(5),
2016, pp. 1009-1021, doi:10.2514/1.G001353.  Cases "Chan 1-12" are attributed there to F. K. Chan,
"Spacecraft Collision Probability" (The Aerospace Press, 2008).

The numbers were taken from the Orekit regression test that encodes that benchmark
(Laas2015Test.java, CS GROUP, https://github.com/CS-SI/Orekit) and, for Chan 1-12, cross-checked
against the authors' JNCF 2014 slides and LLNL-TR-827434 (Bernstein et al., 2021), which give the
same four-digit values.  The JGCD paper itself could not be opened when this test was written, so
the table number inside the paper is not verified.

The 3D case comes from the NASA CARA Analysis Tools unit test for Pc2D_Foster
(https://github.com/nasa/CARA_Analysis_Tools), which states a relative test accuracy of 1e-3.
"""
import numpy as np
import pytest
from scipy import stats

from conjce import pc

# name, sigma_x, sigma_y, R, x_m, y_m, published Pc
PUBLISHED = [
    ("Chan 1", 25.0, 50.0, 5.0, 0.0, 10.0, 9.742e-3),
    ("Chan 2", 25.0, 50.0, 5.0, 10.0, 0.0, 9.181e-3),
    ("Chan 3", 25.0, 75.0, 5.0, 0.0, 10.0, 6.571e-3),
    ("Chan 4", 25.0, 75.0, 5.0, 10.0, 0.0, 6.125e-3),
    ("Chan 5", 1000.0, 3000.0, 10.0, 0.0, 1000.0, 1.577e-5),
    ("Chan 6", 1000.0, 3000.0, 10.0, 1000.0, 0.0, 1.011e-5),
    ("Chan 7", 1000.0, 3000.0, 10.0, 0.0, 10000.0, 6.443e-8),
    ("Chan 8", 1000.0, 3000.0, 10.0, 10000.0, 0.0, 3.219e-27),
    ("Chan 9", 1000.0, 10000.0, 10.0, 0.0, 10000.0, 3.033e-6),
    ("Chan 10", 1000.0, 10000.0, 10.0, 10000.0, 0.0, 9.656e-28),
    ("Chan 11", 1000.0, 3000.0, 50.0, 0.0, 5000.0, 1.039e-4),
    ("Chan 12", 1000.0, 3000.0, 50.0, 5000.0, 0.0, 1.564e-9),
    ("CSM 1", 57.918666, 152.8814468, 10.3, 84.875546, 60.583685, 1.9002e-3),
    ("CSM 2", 15.988242, 5756.840725, 1.3, -81.618369, 115.055899, 2.0553e-11),
    ("CSM 3", 94.230921, 643.409272, 5.3, 102.177247, 693.405893, 7.2003e-5),
    ("CDM 1", 445.859950, 6095.858688, 3.5, -752.672701, 644.939441, 5.3904e-7),
    ("CDM 2", 193.454603, 562.027293, 13.2, -692.362272, 4475.456261, 2.2796e-20),
]

# Stress cases with a hard-body radius far larger than the smaller sigma (same benchmark).
STRESS = [
    ("Alfano 3", 1.4101830, 114.2585190, 15.0, -3.8872073, 0.1591646, 1.0038e-1),
    ("Alfano 5", 0.0373279, 177.8109003, 10.0, -1.2217895, 2.1230067, 4.4509e-2),
]

# Chan's own series on the Chan cases, as tabulated in the authors' JNCF 2014 slides.
CHAN_SERIES_PUBLISHED = [9.754e-3, 9.189e-3, 6.586e-3, 6.135e-3, 1.577e-5, 1.011e-5,
                         6.443e-8, 3.216e-27, 3.033e-6, 9.645e-28, 1.039e-4, 1.556e-9]


@pytest.mark.parametrize("name,sx,sy,R,xm,ym,ref", PUBLISHED, ids=[c[0] for c in PUBLISHED])
def test_foster_matches_published_values(name, sx, sy, R, xm, ym, ref):
    # published values carry 4 or 5 significant digits, so half a unit in the last place is <= 5e-4
    assert pc.pc_foster(xm, ym, sx, sy, R) == pytest.approx(ref, rel=5e-4)


@pytest.mark.parametrize("name,sx,sy,R,xm,ym,ref", PUBLISHED, ids=[c[0] for c in PUBLISHED])
def test_gauss_legendre_agrees_with_adaptive_reference(name, sx, sy, R, xm, ym, ref):
    assert pc.pc_foster(xm, ym, sx, sy, R) == pytest.approx(pc.pc_foster_quad(xm, ym, sx, sy, R), rel=1e-9)


@pytest.mark.parametrize("name,sx,sy,R,xm,ym,ref", STRESS, ids=[c[0] for c in STRESS])
def test_adaptive_reference_on_stress_cases(name, sx, sy, R, xm, ym, ref):
    assert pc.pc_foster_quad(xm, ym, sx, sy, R) == pytest.approx(ref, rel=1e-4)


def test_vectorised_call_matches_scalar_calls():
    arr = np.array([c[1:6] for c in PUBLISHED])
    vec = pc.pc_foster(arr[:, 3], arr[:, 4], arr[:, 0], arr[:, 1], arr[:, 2])
    for value, c in zip(vec, PUBLISHED):
        assert value == pytest.approx(pc.pc_foster(c[4], c[5], c[1], c[2], c[3]), rel=1e-13)


def test_chan_series_matches_published_chan_column():
    for (name, sx, sy, R, xm, ym, _), ref in zip(PUBLISHED[:12], CHAN_SERIES_PUBLISHED):
        assert pc.pc_chan(xm, ym, sx, sy, R) == pytest.approx(ref, rel=5e-4), name


def test_chan_series_is_the_noncentral_chi_square_cdf():
    for name, sx, sy, R, xm, ym, _ in PUBLISHED[:7] + PUBLISHED[8:9] + PUBLISHED[10:15]:
        u = R**2 / (sx * sy)
        v = (xm / sx) ** 2 + (ym / sy) ** 2
        assert pc.pc_chan(xm, ym, sx, sy, R) == pytest.approx(stats.ncx2.cdf(u, 2, v), rel=1e-8), name


def test_chan_series_is_exact_for_equal_sigmas():
    for xm, ym, s, R in [(30.0, 40.0, 60.0, 8.0), (0.0, 0.0, 20.0, 5.0), (200.0, -50.0, 90.0, 12.0)]:
        assert pc.pc_chan(xm, ym, s, s, R) == pytest.approx(pc.pc_foster_quad(xm, ym, s, s, R), rel=1e-9)


def test_centred_isotropic_case_has_closed_form():
    # Rayleigh distribution: P(r <= R) = 1 - exp(-R^2 / (2 sigma^2))
    for s, R in [(10.0, 3.0), (50.0, 5.0), (7.0, 7.0)]:
        assert pc.pc_foster(0.0, 0.0, s, s, R) == pytest.approx(1.0 - np.exp(-R**2 / (2 * s**2)), rel=1e-12)


def test_symmetries_and_scale_invariance():
    base = pc.pc_foster(84.9, 60.6, 57.9, 152.9, 10.3)
    assert pc.pc_foster(-84.9, 60.6, 57.9, 152.9, 10.3) == pytest.approx(base, rel=1e-13)
    assert pc.pc_foster(84.9, -60.6, 57.9, 152.9, 10.3) == pytest.approx(base, rel=1e-13)
    assert pc.pc_foster(60.6, 84.9, 152.9, 57.9, 10.3) == pytest.approx(base, rel=1e-13)       # swap axes
    assert pc.pc_foster(0.0849, 0.0606, 0.0579, 0.1529, 0.0103) == pytest.approx(base, rel=1e-12)  # km


def test_small_hard_body_approximation_when_radius_is_small():
    exact = pc.pc_foster(120.0, 30.0, 150.0, 60.0, 1.0)
    assert pc.pc_small_hbr(120.0, 30.0, 150.0, 60.0, 1.0) == pytest.approx(exact, rel=1e-4)


def test_monte_carlo_agrees_on_a_moderate_probability_case():
    # Chan 1: sample the relative position and count hits inside the disc.
    _, sx, sy, R, xm, ym, _ = PUBLISHED[0]
    rng = np.random.default_rng(20261118)
    n = 4_000_000
    x = rng.normal(xm, sx, n)
    y = rng.normal(ym, sy, n)
    p_hat = np.mean(x * x + y * y <= R * R)
    p = pc.pc_foster(xm, ym, sx, sy, R)
    std_err = np.sqrt(p * (1 - p) / n)
    assert abs(p_hat - p) < 4 * std_err


def test_projection_from_3d_states_nasa_cara_case():
    # km and km/s, covariance in km^2, combined hard-body radius 0.020 km
    r1 = [378.39559, 4305.721887, 5752.767554]
    v1 = [2.360800244, 5.580331936, -4.322349039]
    c1 = [[44.5757544811362, 81.6751751052616, -67.8687662707124],
          [81.6751751052616, 158.453402956163, -128.616921644857],
          [-67.8687662707124, -128.616921644858, 105.490542562701]]
    r2 = [374.5180598, 4307.560983, 5751.130418]
    v2 = [-5.388125081, -3.946827739, 3.322820358]
    c2 = [[2.31067077720423, 1.69905293875632, -1.4170164577661],
          [1.69905293875632, 1.24957388457206, -1.04174164279599],
          [-1.4170164577661, -1.04174164279599, 0.869260558223714]]
    miss, cov2d, basis = pc.project_to_encounter_plane(r1, v1, c1, r2, v2, c2)
    assert np.linalg.norm(miss) <= np.linalg.norm(np.subtract(r2, r1))
    assert basis @ (np.subtract(v2, v1)) == pytest.approx([0.0, 0.0], abs=1e-12)   # plane is normal to v_rel
    value = pc.pc_foster_cov(miss, cov2d, 0.020)
    assert value == pytest.approx(2.70601573490125e-05, rel=1e-3)   # CARA expected value and tolerance
