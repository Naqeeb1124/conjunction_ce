# DERIVATIONS.md

Every formula the code uses, with the steps. Section numbers are referred to from the code's docstrings. Numbers quoted here are for the baseline conjunction in `conjce/model.py`. They come from `results/results.json` (written by `run_all.py`; the key is named where it helps) or from the package call named next to them; the numbers of section 7 come from `results/scaleup.json` (written by `run_scaleup.py`). In sections 1.8 and 2.7 the measured numbers come from `matlab/results/twobody_check.json` (written by `matlab/run_twobody_check.m`) and, for the one comparison with the Simulink model, from `matlab/results/simulink_check.json`; the worked numbers there are computed from the baseline parameters as shown.

Symbols used throughout:

| Symbol | Meaning |
|---|---|
| `R` | combined hard-body radius (sum of the two satellites' radii) |
| `(xm, ym)`, `(sx, sy)` | miss vector and 1-sigma uncertainties in the encounter plane, principal axes |
| `Pc` | collision probability |
| `n`, `T`, `v` | mean motion, period and speed of the circular reference orbit |
| `dv`, `dt` | size of a tangential burn, and how long before closest approach it is done |
| `phi` | angle between the two satellites' velocity vectors at closest approach |
| `a = (a1, a2)` | joint action; each `a_i` is N (no burn), S (small) or L (large) |
| `x(a)` | probability that the coordinator recommends joint action `a` |
| `c_i(a)` | operator i's cost of joint action `a`, as a fraction of its satellite's value |
| `kappa_i(a_i)` | operator i's own maneuver cost for action `a_i`, same unit |
| `mu` | margin (slack) on the incentive constraints, same unit |
| `k` | number of past events an operator has learned the rule from |

---

## 1. Collision probability in the encounter plane

### 1.1 The short-encounter model

Assumptions (Foster and Estes 1992; Akella and Alfriend 2000; Chan 2008):

1. During the encounter the relative motion is a straight line at constant relative velocity. For the baseline the relative speed is 10.7 km/s, so the satellites cross the uncertainty region in a fraction of a second.
2. The uncertainty of the relative position is Gaussian and does not change during the encounter. Its covariance is the sum of the two satellites' position covariances (independent errors).
3. A collision happens if the relative position passes within `R` of zero.

Because the motion is a straight line, only the component of the relative position perpendicular to the relative velocity matters. That plane is the encounter plane (also called the B-plane). The component along the relative velocity only decides *when* the closest approach happens.

### 1.2 The 2D integral (Foster)

In the encounter plane the relative position is a 2D Gaussian with mean `m` (the miss vector) and 2x2 covariance `C`. The collision probability is its probability mass inside the disc of radius `R` around the origin:

```
Pc = 1 / (2 pi sqrt(det C)) * double integral over |r| <= R of exp( -1/2 (r - m)^T C^-1 (r - m) ) dA
```

Rotate the axes onto the eigenvectors of `C`. Then `C = diag(sx^2, sy^2)`, `det C = sx^2 sy^2`, and

```
Pc = 1 / (2 pi sx sy) * double integral over x^2 + y^2 <= R^2 of
         exp( -(x - xm)^2 / (2 sx^2) - (y - ym)^2 / (2 sy^2) ) dx dy                    (1.1)
```

`pc_foster_cov` does this rotation with `numpy.linalg.eigh` and then calls `pc_foster`.

### 1.3 Reducing it to one dimension

Fix `x` with `|x| <= R`. Inside the disc, `y` runs from `-h` to `+h`, where `h = sqrt(R^2 - x^2)` is half the chord. The integrand of (1.1) is a product of a function of `x` and a function of `y`, so

```
Pc = integral from -R to R of  f(x) * G(x) dx

f(x) = exp( -(x - xm)^2 / (2 sx^2) ) / ( sqrt(2 pi) sx )                    (normal density in x)
G(x) = integral from -h to h of exp( -(y - ym)^2 / (2 sy^2) ) / ( sqrt(2 pi) sy ) dy
```

For `G`, substitute `z = (y - ym) / sy`, so `dy = sy dz` and the limits become `(-h - ym)/sy` and `(h - ym)/sy`:

```
G(x) = Phi( (h - ym) / sy ) - Phi( (-h - ym) / sy )                                     (1.2)
```

with `Phi` the standard normal distribution function. Using `Phi(z) = 1/2 [1 + erf(z / sqrt 2)]` this is the form with two error functions that Alfano (2005) integrates numerically.

`G` does not change if `ym` is replaced by `-ym`:

```
Phi((h + ym)/sy) - Phi((-h + ym)/sy) = [1 - Phi((-h - ym)/sy)] - [1 - Phi((h - ym)/sy)]
                                     = Phi((h - ym)/sy) - Phi((-h - ym)/sy)
```

so the code uses `|ym|`. With `|ym|` both arguments are small or negative when the miss is large, which keeps the difference accurate in the far tail (values down to 1e-27 are reproduced in the tests).

### 1.4 The substitution that makes the outer integral easy

`h = sqrt(R^2 - x^2)` has infinite slope at `x = +R` and `x = -R`, which slows down any quadrature rule. Substitute `x = R sin t` with `t` from `-pi/2` to `pi/2`:

```
dx = R cos t dt,        h = sqrt(R^2 - R^2 sin^2 t) = R cos t

Pc = integral from -pi/2 to pi/2 of  f(R sin t) * [ Phi((R cos t - |ym|)/sy) - Phi((-R cos t - |ym|)/sy) ] * R cos t dt      (1.3)
```

The integrand is now smooth. Gauss-Legendre quadrature with nodes `u_j` and weights `w_j` on `[-1, 1]` and `t = (pi/2) u` gives

```
Pc ~ (pi/2) * sum over j of  w_j * F( (pi/2) u_j )         with F the integrand of (1.3)
```

`pc_foster` uses 128 nodes and does the numerical integral along the axis with the *larger* sigma, because then the density `f` varies slowly across the disc. `pc_foster_quad` evaluates the same formula with an adaptive rule and is the reference when `R` is much larger than the smaller sigma.

### 1.5 Chan's series

Chan's method turns (1.1) into a one-line series by one approximation.

**Step 1, make the Gaussian round.** Substitute `x' = x * sy / sx`. Then `(x - xm)/sx = (x' - xm')/sy` with `xm' = xm * sy/sx`, and `dx = (sx/sy) dx'`:

```
Pc = 1 / (2 pi sy^2) * double integral over the ellipse of exp( -[(x' - xm')^2 + (y - ym)^2] / (2 sy^2) ) dx' dy
```

The Gaussian is now isotropic with standard deviation `sy`. The disc `x^2 + y^2 <= R^2` has become the ellipse `x'^2 (sx/sy)^2 + y^2 <= R^2`, with semi-axes `R sy/sx` and `R`, and area `pi R^2 sy/sx`.

**Step 2, the approximation.** Replace the ellipse by the disc with the same area and the same centre. Its radius `rho` satisfies `pi rho^2 = pi R^2 sy/sx`. This is exact when `sx = sy` and good when `R` is small compared with both sigmas.

**Step 3, non-dimensional numbers.** Measure lengths in units of `sy`:

```
u = rho^2 / sy^2 = R^2 / (sx sy)
v = (xm'^2 + ym^2) / sy^2 = xm^2 / sx^2 + ym^2 / sy^2
```

`v` is the squared miss distance in sigmas; `u` is the squared hard-body radius in sigmas.

**Step 4, integrate.** We need the probability that a unit-variance isotropic 2D Gaussian centred at distance `sqrt(v)` from the origin falls within radius `sqrt(u)` of the origin. In polar coordinates `(r, theta)` around the origin the density is `(1/2pi) exp( -(r^2 + v - 2 r sqrt(v) cos theta)/2 ) r`. Integrating over `theta` uses `integral from 0 to 2pi of exp(z cos theta) d theta = 2 pi I0(z)`:

```
Pc = exp(-v/2) * integral from 0 to sqrt(u) of  r exp(-r^2/2) I0(r sqrt(v)) dr          (the Rice distribution)
```

Insert the power series `I0(z) = sum over m >= 0 of (z^2/4)^m / (m!)^2`:

```
Pc = exp(-v/2) * sum over m of  v^m / (4^m (m!)^2) * integral from 0 to sqrt(u) of r^(2m+1) exp(-r^2/2) dr
```

In the remaining integral put `s = r^2/2`, so `r dr = ds` and `r^(2m) = (2s)^m`:

```
integral = 2^m * integral from 0 to u/2 of s^m exp(-s) ds
         = 2^m * m! * [ 1 - exp(-u/2) * sum from k=0 to m of (u/2)^k / k! ]
```

(the last step is the closed form of the incomplete gamma function for integer `m`). Collecting factors, `v^m/(4^m (m!)^2) * 2^m m! = v^m / (2^m m!)`:

```
Pc = exp(-v/2) * sum over m >= 0 of  v^m / (2^m m!) * [ 1 - exp(-u/2) * sum from k=0 to m of u^k / (2^k k!) ]        (1.4)
```

This is Chan's series (Chan 1997, 2008; the same expression is eq. (3) of Alfano's 2007 review). Two remarks used in the code:

- The square bracket is the regularised incomplete gamma function `P(m+1, u/2)`. `pc_chan` calls `scipy.special.gammainc` for it, because the literal difference `1 - exp(-u/2)(...)` loses all significant digits when `u` is small.
- The weights `exp(-v/2) (v/2)^m / m!` are Poisson probabilities, and (1.4) is exactly the distribution function of a noncentral chi-square variable with 2 degrees of freedom and noncentrality `v`, evaluated at `u`. A test checks `pc_chan` against `scipy.stats.ncx2.cdf`.

The project's numbers use the Foster integral (1.3). Chan's series is implemented and tested because it is the standard closed form and it is the easiest way to explain how `Pc` depends on the inputs.

### 1.6 Small hard body, and why the residual risk is so sensitive to the covariance

If `R` is much smaller than both sigmas the density is nearly constant over the disc, so `Pc` is the density at the origin times the area `pi R^2`:

```
Pc ~ R^2 / (2 sx sy) * exp(-v/2)                                                        (1.5)
```

(the `m = 0` term of (1.4) with `1 - exp(-u/2) ~ u/2`). For the baseline with nobody burning, (1.5) gives 1.489e-3 against 1.488e-3 from the full integral.

Now scale both sigmas by a common factor `kc` (the covariance by `kc^2`). In (1.5), `sx sy` becomes `kc^2 sx sy` and `v` becomes `v / kc^2`:

```
ln Pc(kc) = const - 2 ln kc - v / (2 kc^2)
d ln Pc / d ln kc = -2 + v / kc^2          which is  v - 2  at kc = 1                    (1.6)
```

So a larger covariance *lowers* `Pc` when `v < 2` (the probability is diluted) and *raises* it when `v > 2`. In the baseline:

| Joint action | `v` | slope `v - 2` | `Pc` at 0.9, 1.0, 1.1, 1.2 times the nominal sigmas |
|---|---|---|---|
| nobody burns | 1.20 | -0.80 | 1.60e-3, 1.49e-3, 1.37e-3, 1.24e-3 |
| one small burn | 9.07 | +7.07 | 1.24e-5, 2.91e-5, 5.29e-5, 8.08e-5 |

After a burn the pass point sits far out on the Gaussian tail, so a 10% error in the sigmas almost doubles the residual collision probability. Section 4 shows that the operators' incentives depend on exactly this residual number, which is why uncertainty about the covariance matters so much in RESULTS.md.

### 1.7 What is tested (`tests/test_pc.py`, 45 tests)

- 17 published encounter-plane cases (Chan 1-12, CSM 1-3, CDM 1-2) from the benchmark of Serra et al. (2016): `pc_foster` matches every published value within the rounding of the published digits (largest relative difference 2.8e-4).
- `pc_foster` against the adaptive reference on the same cases: agreement to 3e-14.
- Two stress cases with `R` far larger than the smaller sigma (Alfano 3 and 5): the adaptive reference matches the published values to 1e-4.
- Chan's series against the published "Chan" column for the 12 Chan cases, against the noncentral chi-square function, and against Foster's integral when `sx = sy`.
- The closed form `1 - exp(-R^2 / (2 sigma^2))` for a centred, isotropic case; symmetries; unit invariance.
- A Monte Carlo count (4 million samples) on Chan case 1.
- The NASA CARA 3D case: projection of states and covariances onto the encounter plane, then (1.3), against CARA's stored value at CARA's own tolerance (1e-3).

### 1.8 A sampling check with two-body motion, and why it needs importance sampling

`matlab/twobody/pc_montecarlo.m` checks the collision probability by sampling. The idea is to draw the relative position at the nominal closest approach from the model's Gaussian, move both satellites along their two-body orbits to their true closest approach, and count a hit when that distance is below `R`. This tests the numerical integral (1.3) and assumption 1 of section 1.1 (straight-line motion) together. It does not test assumption 2, because the Gaussian is the model's own. Counting hits among Gaussian samples (`pc_montecarlo_naive.m`) cannot work with a few thousand samples, as shown next, so `pc_montecarlo.m` draws the positions from another density and weights them with the Gaussian.

**Plain counting cannot resolve these probabilities with a few thousand samples.** With `n` independent samples the number of hits is binomial with success probability `Pc`, so the estimate `hits / n` has standard error

```
se = sqrt( Pc (1 - Pc) / n )
```

For the baseline with nobody burning, `Pc = 1.488e-3`. With `n = 2000` the standard error is `sqrt(1.488e-3 * 0.9985 / 2000) = 8.6e-4`, which is 58% of `Pc`, and the expected number of hits is `n Pc = 2.98`. After one small burn `Pc = 2.9e-5` and the expected number of hits is 0.06.

**Importance sampling.** Write `p(x)` for the Gaussian density of the relative position `x` in the encounter plane (the integrand of (1.1) with its prefactor) and `H` for the set of positions that lead to a hit, so `Pc = integral of 1_H(x) p(x) dA`. Let `q(x)` be any density that is positive on `H`. Multiply and divide by `q`:

```
Pc = integral of 1_H(x) (p(x) / q(x)) q(x) dA = E_q[ 1_H(x) p(x) / q(x) ]                (1.7)
```

So if `x_1, ..., x_n` are drawn from `q`, the average of the terms `1_H(x_k) w_k` with weights `w_k = p(x_k) / q(x_k)` is an unbiased estimate of `Pc`, and its standard error is the sample standard deviation of those terms divided by `sqrt(n)`.

The proposal used here is uniform on the disc of radius `r_q = 2R` around zero relative position, `q = 1 / (pi r_q^2)`. For straight-line motion `H` is the disc of radius `R`, so `r_q = 2R` covers it and leaves room for any effect of the curved motion. The weight is

```
w_k = pi r_q^2 p(x_k) = ( r_q^2 / (2 sx sy) ) exp( -(x_k - xm)^2 / (2 sx^2) - (y_k - ym)^2 / (2 sy^2) )
```

The position along the relative velocity does not enter (1.1). Each sample draws it from its own Gaussian, with standard deviation `sigma_s`, and it needs no weight. `sigma_s` is not a parameter of the game model. The check sets it to `sx tan(phi/2)`: if the uncertainty comes from independent along-track errors `e1`, `e2` of equal variance on the two satellites, the relative position error is `e2 t2 - e1 t1`, which by section 2.4 has the component `(e2 - e1) cos(phi/2)` along `xihat` and `(e2 + e1) sin(phi/2)` along `shat`; the two variances are in the ratio `tan^2(phi/2)`.

**Expected accuracy.** If `p` were constant over the proposal disc, every weight would equal one value `w0` and the estimate would be `w0` times the fraction of samples that hit. That fraction is binomial with success probability `h = pi R^2 / (pi r_q^2) = 1/4`. Then `Pc = w0 h` and

```
se = w0 sqrt( h (1 - h) / n ) = Pc sqrt( (1 - h) / (h n) ) = Pc sqrt( 3 / n )           (1.8)
```

which is 3.9% of `Pc` for `n = 2000`, whatever the size of `Pc`.

When `p` varies across the disc, (1.8) is a lower bound. Write `m1` and `m2` for the averages of `w` and of `w^2` over `H`. A sample hits with probability `h`, so the mean of one term is `h m1 = Pc` and its mean square is `h m2`. The variance of one term is `h m2 - Pc^2`, and

```
se^2 / Pc^2 = ( h m2 - h^2 m1^2 ) / ( n h^2 m1^2 ) = ( m2 / (h m1^2) - 1 ) / n  >=  ( 1/h - 1 ) / n = 3 / n
```

because `m2 >= m1^2`, with equality when all weights on `H` are equal. The standard error that the check reports is estimated from the 2,000 samples themselves, so it scatters around this: a case with 534 hits instead of the expected 500 reports about `sqrt( 0.733 / (0.267 * 2000) ) = 3.7%`.

**What is tested** (`matlab/tests/test_twobody.m`; numbers from `matlab/results/twobody_check.json`):

- In each of the nine joint actions of the baseline (2,000 samples each, hit decided by the two-body closest approach) the estimate is within 2.1 standard errors of (1.3). The standard errors are 3.7% to 4.2% of `Pc`, for probabilities from 1.5e-3 down to 5.4e-43. The test bound is 4 standard errors.
- The two-body hit test and the test `|x| < R` in the encounter plane give the same answer for every one of the 18,000 samples.
- Plain counting on the no-burn case, for comparison only: 4 hits in 2,000 samples, exact 95% interval 5.5e-4 to 5.1e-3. It contains 1.488e-3 and is too wide to check anything.

With 2,000 samples the check resolves differences in `Pc` of a few percent. It cannot see the difference between the linear and the propagated miss of section 2.7, which changes `Pc` by at most 0.2%.

---

## 2. From a burn to a change of the miss vector

### 2.1 The Clohessy-Wiltshire equations

For small motion relative to a circular reference orbit with mean motion `n`, in axes `x` radial (outward), `y` along-track (direction of motion), `z` cross-track (Clohessy and Wiltshire 1960):

```
x'' - 2 n y' - 3 n^2 x = 0
y'' + 2 n x'           = 0
z'' + n^2 z            = 0
```

### 2.2 Solution for one tangential impulse

A burn `dv` along-track at `t = 0` means `x(0) = y(0) = 0`, `x'(0) = 0`, `y'(0) = dv`.

**Step 1.** Integrate the second equation once: `y' + 2 n x` is constant, and at `t = 0` it equals `dv`. So `y' = dv - 2 n x`.

**Step 2.** Put this into the first equation:

```
x'' = 2 n y' + 3 n^2 x = 2 n (dv - 2 n x) + 3 n^2 x = 2 n dv - n^2 x
x'' + n^2 x = 2 n dv
```

**Step 3.** A particular solution is the constant `2 dv / n`. The general solution is `x = 2 dv/n + A cos nt + B sin nt`. From `x(0) = 0`, `A = -2 dv/n`; from `x'(0) = 0`, `B = 0`:

```
x(t) = (2 dv / n) (1 - cos nt)                                                          (2.1)
```

**Step 4.** Back in step 1: `y' = dv - 4 dv (1 - cos nt) = -3 dv + 4 dv cos nt`. Integrate with `y(0) = 0`:

```
y(t) = -3 dv t + (4 dv / n) sin nt                                                      (2.2)
```

These agree with the general solution in standard lecture notes (MIT OpenCourseWare 16.346, lecture 26) and with the relative-motion form in Schaub (2004).

### 2.3 Whole orbits: the "3 dv dt" rule

After a whole number of orbits `nt` is a multiple of `2 pi`, so `cos nt = 1` and `sin nt = 0`:

```
x = 0,        y = -3 dv t                                                               (2.3)
```

A posigrade burn (`dv > 0`) leaves the satellite *behind* where it would have been, by `3 dv t`.

The same number from first principles, as a check. For a circular orbit `v^2 = mu / a`. The vis-viva equation `v^2 = mu (2/r - 1/a)` at fixed `r` gives `2 v dv = mu da / a^2`, so `da = 2 a^2 v dv / mu = 2 a dv / v`. Kepler's third law `T ~ a^(3/2)` gives `dT / T = (3/2) da / a = 3 dv / v`. After time `t` the satellite has completed `t/T` orbits, each `dT` longer, so it is late by `(t/T) dT` seconds, which at speed `v` is a distance `v (t/T)(3 T dv / v) = 3 dv t`.

The baseline burns two orbits before closest approach: `T = 5739.0 s`, `dt = 11478 s`, so 1 cm/s gives `3 * 0.01 * 11478 = 344 m` along-track. Away from whole orbits the periodic terms matter; the largest is the radial offset `4 dv / n` at half an orbit (36.5 m for 1 cm/s). Aida (2016) uses exactly these two expressions, `3 dv dt` and `4 a dv / v`, for operational avoidance planning.

### 2.4 Projecting onto the encounter plane

At closest approach let `rhat` be the radial unit vector and `t1`, `t2` the unit vectors along the two velocities. Both are horizontal (near-circular orbits) and the angle between them is `phi`, so `t1 . t2 = cos phi`.

Relative velocity (satellite 2 minus satellite 1), with equal speeds `v`:

```
w = v (t2 - t1),       |t2 - t1|^2 = 2 - 2 cos phi = 4 sin^2(phi/2),       shat = (t2 - t1) / (2 sin(phi/2))
```

The bisector of the two velocity directions:

```
|t1 + t2|^2 = 2 + 2 cos phi = 4 cos^2(phi/2),       xihat = (t1 + t2) / (2 cos(phi/2))
```

`xihat` is perpendicular to `shat` because `(t1 + t2) . (t2 - t1) = |t2|^2 - |t1|^2 = 0`. `rhat` is perpendicular to both. So the encounter plane is spanned by `xihat` (horizontal) and `zetahat = rhat` (radial).

How much of an along-track displacement shows up in the plane:

```
t1 . xihat = (1 + cos phi) / (2 cos(phi/2)) = 2 cos^2(phi/2) / (2 cos(phi/2)) = cos(phi/2)
t1 . zetahat = 0
t1 . shat = (cos phi - 1) / (2 sin(phi/2)) = -sin(phi/2)
```

and the same for `t2` with `t2 . shat = +sin(phi/2)`.

Why only the in-plane part counts. With straight-line relative motion `rho(t) = rho0 + w t`, the distance is smallest when `d|rho|^2/dt = 0`, at `t* = -(rho0 . w)/|w|^2`. The miss vector is `rho0 + w t* = rho0 - (rho0 . shat) shat`, the projection of `rho0` onto the encounter plane. A displacement of either satellite therefore changes the miss vector by its projection onto the plane; the part along `shat` only shifts the time of closest approach.

The general linear map from an impulse at any point of the orbit to the displacement in the encounter plane is given in Bombardelli and Hernando-Ayuso (2015); only the tangential, whole-orbit case is derived and used here.

### 2.5 The gain, the sign convention, and cancellation

Let `rho = r2 - r1`. A burn `dv1` by satellite 1 displaces it by `d1 t1` with `d1 = -3 dv1 dt` (2.3), which changes `rho` by `-d1 t1`. A burn `dv2` by satellite 2 changes `rho` by `+d2 t2` with `d2 = -3 dv2 dt`. Project with section 2.4:

```
change of xi   = -d1 cos(phi/2) + d2 cos(phi/2) = 3 dt cos(phi/2) (dv1 - dv2)           (2.4)
change of zeta = 0
```

Convention in the model: each operator burns in the sense that increases the nominal miss. For a nominal miss with `xi0 > 0` that is posigrade for satellite 1 (`dv1 = +u1`) and retrograde for satellite 2 (`dv2 = -u2`), with burn sizes `u1, u2 >= 0`:

```
xi(a) = xi0 + g (u1 + u2),        g = 3 dt cos(phi/2)                                   (2.5)
```

Baseline: `phi = 90 degrees`, `g = 3 * 11478 * cos(45 degrees) = 24348 m per m/s`. A 1 cm/s burn moves the pass point 243.5 m, a 3 cm/s burn 730.5 m.

Equation (2.4) also says that equal burns in the *same* sense cancel: both satellites fall behind by the same amount, the relative displacement is along `shat`, and only the time of closest approach changes. That is why the direction of the burn has to be agreed even when the question of who burns is settled. The model fixes the directions and leaves this out.

What (2.5) neglects: the periodic terms of (2.1)-(2.2) (zero at whole orbits), the change of the relative velocity direction caused by the burn (of order `dv / |w|`, about 1e-6), orbit perturbations over the two orbits, and the eccentricity the burn introduces.

### 2.6 What is tested (`tests/test_maneuver.py`, 27 tests)

- (2.1)-(2.2) against a numerical integration of the Clohessy-Wiltshire equations.
- (2.3) against a two-body (Keplerian) propagation of a burned and an unburned satellite: 344.341 m against the predicted 344.340 m.
- (2.4) against two-body propagation of *both* satellites through a 3D crossing encounter, for crossing angles of 60, 90 and 120 degrees and six burn combinations. The predicted shift agrees with the propagated one to about 1e-5 (one centimetre in 730 m); the radial component does not move. The test tolerance is 1e-4.
- The cancellation of equal same-sense burns, by the same propagation.

Not tested: anything beyond two-body motion.

### 2.7 Second-order terms of (2.5)

Sections 2.2 to 2.5 are linear in the burn. This section derives the terms of second order; they account for what a two-body propagation finds beyond (2.5). Here `a` is the radius of the reference orbit and `mu` the gravitational parameter, as in section 2.3; `N = dt / T` is the whole number of reference periods between the burn and the nominal closest approach (2 in the baseline); `s_i = +1` for a posigrade and `-1` for a retrograde burn of size `u_i >= 0`; and `e_i = u_i / v` (1.3e-6 for 1 cm/s).

**Step 1. The period after a tangential burn.** Before the burn let the satellite have semi-major axis `a_b` and period `T_b`, and at the burn point radius `r_b` and speed `v_b`. The burn changes the speed to `v_b (1 + s e)` at the same radius. The vis-viva equation before and after:

```
v_b^2             = mu (2 / r_b - 1 / a_b)
v_b^2 (1 + s e)^2 = mu (2 / r_b - 1 / a')
```

Subtract the first from the second. Since `s^2 = 1`, `(1 + s e)^2 - 1 = 2 s e + e^2`:

```
v_b^2 (2 s e + e^2) = mu (1 / a_b - 1 / a')
1 / a' = (1 / a_b) [ 1 - chi (2 s e + e^2) ],        chi = a_b v_b^2 / mu
```

(`chi = 1` for a circular orbit). Kepler's third law, `T ~ a^(3/2)`:

```
T' = T_b (a' / a_b)^(3/2) = T_b [ 1 - chi (2 s e + e^2) ]^(-3/2)
```

Use the binomial series `(1 - y)^(-3/2) = 1 + (3/2) y + (15/8) y^2 + ...` with `y = chi (2 s e + e^2)`. Then `(3/2) y = 3 chi s e + (3/2) chi e^2` and `(15/8) y^2 = (15/8) chi^2 * 4 e^2 + O(e^3) = (15/2) chi^2 e^2 + O(e^3)`:

```
T' = T_b [ 1 + 3 chi s e + ( (3/2) chi + (15/2) chi^2 ) e^2 ] + O(e^3)                  (2.6)
```

For a circular orbit `T' = T_b (1 + 3 s e + 9 e^2)`; the first-order term is the `dT / T = 3 dv / v` of section 2.3.

**Step 2. Where the satellite is at the nominal time of closest approach.** Call the burn point `B`, and `that_B`, `rhat_B` the along-track and radial unit vectors there. With or without the burn the satellite passes through `B` again after each of its own periods, because a closed orbit repeats: without the burn with velocity `v_b that_B`, with the burn with velocity `v_b (1 + s e) that_B` (a tangential burn changes the speed, not the direction). The burn is at time `-dt = -N T`. The `N`-th return to `B` is at time `-N T + N T_b` without the burn and at `-N T + N T'` with it. At time 0, the nominal closest approach, the time since that return is therefore

```
tau_b = N (T - T_b)     without the burn,        tau' = N (T - T')     with the burn
```

(a negative value means the satellite has not reached `B` yet). Both are below 0.15 s in the baseline, so expand the position around `B`: `r(tau) = B + V tau + (1/2) A tau^2 + O(tau^3)`, with `V` the velocity at `B` and `A = -mu B / |B|^3` the gravitational acceleration at `B`. `A` is the same in both cases because it depends only on the position; for a near-circular orbit `A = -(v_b^2 / r_b) rhat_B`. Write `p_b = v_b tau_b` and `p' = v_b (1 + s e) tau'` for the distances travelled past `B`:

```
without the burn:   r = B + p_b that_B - ( p_b^2 / (2 r_b) ) rhat_B
with the burn:      r = B + p'  that_B - ( p'^2  / (2 r_b) ) rhat_B
```

(in the quadratic term `v_b tau'` has been replaced by `p'`, an error of third order). The along-track displacement caused by the burn is `d = p' - p_b`:

```
d = v_b (1 + s e) N (T - T') - v_b N (T - T_b)
  = v_b N [ (T - T') + s e (T - T') - (T - T_b) ]
  = - v_b N [ (T' - T_b) + s e (T' - T) ]                                               (2.7)
```

The radial part is best read in the local axes at the position the satellite has without the burn. That position is the angle `theta_b = p_b / r_b` further along the orbit than `B`, so its along-track and radial unit vectors are those at `B` turned by `theta_b`: to first order `t = that_B - theta_b rhat_B` and `rhat = rhat_B + theta_b that_B`. The displacement vector is `D = (p' - p_b) that_B - ((p'^2 - p_b^2) / (2 r_b)) rhat_B`. Its radial component:

```
D . rhat = (p' - p_b) theta_b - (p'^2 - p_b^2) / (2 r_b)
         = (p' - p_b) [ 2 p_b - (p' + p_b) ] / (2 r_b)
         = - (p' - p_b)^2 / (2 r_b) = - d^2 / (2 r_b)
```

and its along-track component is `d` up to terms of third order. In words: the satellite is on the same curved track, `d` further along, and a circle of radius `r_b` drops `d^2 / (2 r_b)` below its tangent over an arc `d`.

The velocity at time 0 with the burn is `V + A tau' = v_b (1 + s e) that_B - v_b (p' / r_b) rhat_B` to first order. In the local axes (`that_B = t + theta_b rhat`, `rhat_B = rhat - theta_b t`), again to first order:

```
velocity = v_b (1 + s e) t + v_b (p_b / r_b - p' / r_b) rhat = v_b (1 + s e) t - v_b (d / r_b) rhat
```

Collecting, for satellite `i` with nominal position `r_i` and velocity direction `t_i` at time 0 (in these terms `r_b` and `v_b` can be replaced by `a` and `v`; the differences, parts in 1e5 or less, matter only in (2.7) and are kept there):

```
position:   r_i + d_i t_i - ( d_i^2 / (2a) ) rhat
velocity:   v (1 + s_i e_i) t_i - v (d_i / a) rhat                                      (2.8)
```

**Step 3. Satellite 1.** Its orbit without the burn is the reference orbit: `T_b = T`, `chi = 1`, `v_b = v`, and `s = +1`. From (2.6), `T' - T_b = T' - T = T (3 e1 + 9 e1^2)`. In (2.7), with `v e1 = u1` and `N T = dt`:

```
d1 = - v N [ T (3 e1 + 9 e1^2) + e1 T (3 e1) ] = - v N T (3 e1 + 12 e1^2) = - 3 u1 dt (1 + 4 u1 / v)        (2.9a)
```

**Step 4. Satellite 2.** In the reference encounter (the construction of `tests/test_maneuver.py`, also used by the MATLAB layer) satellite 2 is at `r1 + zeta0 rhat + xi0 xihat` at the nominal closest approach and has the same speed `v` as satellite 1. Its radius there is `a + delta` with `delta = sqrt((a + zeta0)^2 + xi0^2) - a` (30.0006 m in the baseline), so its orbit is not quite the reference orbit. Vis-viva at that point, with `v^2 = mu / a`:

```
1 / a_2 = 2 / (a + delta) - v^2 / mu = 2 / (a + delta) - 1 / a = (a - delta) / ( a (a + delta) )
a_2 = a (a + delta) / (a - delta) = a (1 + 2 delta / a) + O(delta^2 / a)
T_2 = T (a_2 / a)^(3/2) = T (1 + 3 delta / a)
chi = a_2 v^2 / mu = a_2 / a = 1 + 2 delta / a
```

to first order in `delta / a` (4.3e-6). The burn point is `N (T_2 - T) = 0.15 s` of flight away from this point, which changes the radius and the speed by parts in 1e9, so `r_b = a + delta` and `v_b = v` are used. With `T_b = T_2`, `s = -1` and `e = e2`, (2.6) gives, dropping terms of order `e^2 delta / a` and `e delta^2 / a^2`:

```
T' - T_2 = T (1 + 3 delta/a) [ -3 (1 + 2 delta/a) e2 + 9 e2^2 ] = T [ -3 e2 (1 + 5 delta/a) + 9 e2^2 ]
T' - T   = (T' - T_2) + (T_2 - T) = T [ -3 e2 (1 + 5 delta/a) + 9 e2^2 + 3 delta/a ]
s e (T' - T) = - e2 T [ -3 e2 + 3 delta/a ] = T [ 3 e2^2 - 3 e2 delta/a ]
(T' - T_2) + s e (T' - T) = T [ -3 e2 - 15 e2 delta/a + 9 e2^2 + 3 e2^2 - 3 e2 delta/a ]
                          = -3 e2 T [ 1 + 6 delta/a - 4 e2 ]
```

and (2.7) becomes

```
d2 = 3 v e2 N T [ 1 + 6 delta/a - 4 e2 ] = 3 u2 dt (1 + 6 delta/a - 4 u2 / v)           (2.9b)
```

With the brackets set to 1, (2.9a) and (2.9b) are the `d1 = -3 u1 dt` and `d2 = +3 u2 dt` of section 2.5.

**Step 5. Relative position and velocity at the nominal time.** Let `rho = r2 - r1` and `rho0 = xi0 xihat + zeta0 zetahat` the nominal miss. The radial unit vectors of the two satellites differ by the angle `xi0 / a = 1.3e-5`, which only multiplies terms that are already small, so `rhat = zetahat` for both. From (2.8):

```
rho = rho0 + d2 t2 - d1 t1 - ( (d2^2 - d1^2) / (2a) ) zetahat
```

Section 2.4 gives `t1 = cos(phi/2) xihat - sin(phi/2) shat` and `t2 = cos(phi/2) xihat + sin(phi/2) shat`, so `d2 t2 - d1 t1 = (d2 - d1) cos(phi/2) xihat + (d1 + d2) sin(phi/2) shat` and

```
rho . xihat   = xi0 + (d2 - d1) cos(phi/2)
rho . zetahat = zeta0 - (d2^2 - d1^2) / (2a)
rho . shat    = (d1 + d2) sin(phi/2)
```

The relative velocity, from (2.8) with `s1 = +1` and `s2 = -1`:

```
w = v (1 - e2) t2 - v (1 + e1) t1 - (v / a)(d2 - d1) zetahat
  = v (t2 - t1) - v (e2 t2 + e1 t1) - (v / a)(d2 - d1) zetahat
  = v [ 2 - (e2 - e1) ] sin(phi/2) shat - v (e1 + e2) cos(phi/2) xihat - (v / a)(d2 - d1) zetahat
```

using `t2 - t1 = 2 sin(phi/2) shat` and `e2 t2 + e1 t1 = (e1 + e2) cos(phi/2) xihat + (e2 - e1) sin(phi/2) shat`. Divide by the `shat` component, which is `2 v sin(phi/2)` to leading order. To first order the direction of the relative velocity is

```
shat' = shat - alpha xihat - beta zetahat
alpha = (e1 + e2) cos(phi/2) / (2 sin(phi/2)),        beta = (d2 - d1) / (2 a sin(phi/2))
```

The burns turn the relative velocity by `alpha` in the horizontal plane (the two speeds have changed) and by `beta` out of it (the two satellites are at different points of their curved tracks). In the baseline `alpha` is at most 4e-6 and `beta` at most 2.1e-4.

**Step 6. The miss vector.** As in section 2.4 the miss vector is the part of `rho` perpendicular to the relative velocity, which is now along `shat'`: `m = rho - (rho . shat') shat'`. Its components along the nominal axes, using `shat' . xihat = -alpha` and `shat' . zetahat = -beta`:

```
m . xihat   = rho . xihat   + alpha (rho . shat')
m . zetahat = rho . zetahat + beta  (rho . shat')
```

Here `rho . shat' = rho . shat - alpha (rho . xihat) - beta (rho . zetahat)`. Inserted above, the last two terms give

```
in xi:    - alpha^2 (rho . xihat) - alpha beta (rho . zetahat)
in zeta:  - alpha beta (rho . xihat) - beta^2 (rho . zetahat)
```

Each is a product of two tilts and a component of the miss itself (`rho . xihat` is at most 1553 m, `rho . zetahat` about 30 m), not of the much longer drift `d`. For two large burns, where they are largest, `alpha = 3.96e-6` and `beta = 2.11e-4`: the two terms in `xi` add up to 4.9e-8 m and the two in `zeta` to `1.30e-6 + 1.33e-6 = 2.6e-6 m`. They are dropped. With `rho . shat' = rho . shat`:

```
xi   = xi0 + (d2 - d1) cos(phi/2) + alpha (d1 + d2) sin(phi/2)
     = xi0 + cos(phi/2) [ (d2 - d1) + (1/2)(e1 + e2)(d1 + d2) ]
zeta = zeta0 - (d2^2 - d1^2) / (2a) + beta (d1 + d2) sin(phi/2)
     = zeta0 - (d2^2 - d1^2) / (2a) + (d2 - d1)(d1 + d2) / (2a)
     = zeta0
```

(`alpha sin(phi/2) = (1/2)(e1 + e2) cos(phi/2)` and `beta sin(phi/2) = (d2 - d1) / (2a)`). The radial component stays at `zeta0`: each satellite's drop below its tangent, `d^2 / (2a)`, which is 7.7 cm for a large burn, is cancelled exactly by the tilt `beta` of the relative velocity. What is left in `zeta` are the dropped products above, a few micrometres.

**Step 7. Insert (2.9).** Keeping terms up to second order,

```
d2 - d1 = 3 dt [ u2 (1 + 6 delta/a - 4 u2/v) + u1 (1 + 4 u1/v) ]
        = 3 dt [ (u1 + u2) + (4/v)(u1^2 - u2^2) + (6 delta/a) u2 ]
(1/2)(e1 + e2)(d1 + d2) = (1/2) ((u1 + u2) / v) * 3 dt (u2 - u1) = - (3 dt / (2v)) (u1^2 - u2^2)
```

Their sum is `3 dt [ (u1 + u2) + (4 - 1/2)(u1^2 - u2^2) / v + (6 delta/a) u2 ]`. With `g = 3 dt cos(phi/2)`:

```
xi(a)   = xi0 + g (u1 + u2) + ( 7 g / (2 v) ) (u1^2 - u2^2) + (6 delta / a) g u2        (2.10)
zeta(a) = zeta0
```

The first two terms are (2.5). Where the others come from:

- `4 u^2 / v` in (2.9): 3 of the 4 are the second-order term of the period in (2.6); the last 1 is the higher (or lower) speed with which the satellite covers the stretch to `B`.
- `-1/2`: the burns change the two speeds, which turns the relative velocity by `alpha`; the separation `(d1 + d2) sin(phi/2)` along `shat` then has a component in the new encounter plane.
- `6 delta / a`: satellite 2's own period is longer than `T` by `3 delta T / a`, so `N` reference periods are not `N` of its own. This term depends on how the reference encounter is set up (both satellites are given the speed `v`).

Baseline: `7 g / (2 v) = 7 * 24348.48 / (2 * 7585.09) = 11.235 m per (m/s)^2`, which is 1.12 mm for a 1 cm/s burn and 10.11 mm for 3 cm/s; `6 delta / a = 2.598e-5`, which is 6.33 mm on a 1 cm/s burn of satellite 2 and 18.98 mm on 3 cm/s. The same projection gives the time of closest approach, `t* = -(rho . w) / |w|^2 = -(d1 + d2) / (2v)` to first order: 0.0227 s for one small burn by satellite 1.

What (2.10) neglects: the terms of third order in the burns, of which the largest is the `O(tau^3)` term of step 2 (about 5e-6 m), and the products of two tilts with the miss that were dropped in step 6 (below 1e-7 m in `xi`, up to 2.6e-6 m in `zeta`). The second group is of second order in the burns, but it multiplies the miss, tens to hundreds of metres, where the terms kept in (2.10) multiply the distance flown during the lead time.

**Step 8. Effect on the collision probability.** From (1.5), `ln Pc = const - xm^2 / (2 sx^2) - ym^2 / (2 sy^2)`, so a small change `dxi` of the miss along `xihat` changes `Pc` by

```
dPc / Pc = - xi dxi / sx^2                                                              (2.11)
```

For two large burns, `xi = 1552.9 m` and `dxi = 18.98 mm` give `-1552.9 * 0.01898 / 115^2 = -2.2e-3`.

**What is tested** (`matlab/tests/test_twobody.m`; numbers from `matlab/results/twobody_check.json`, written by `matlab/run_twobody_check.m`). Both satellites are propagated on their two-body orbits for the nine joint actions of the baseline, with `ode113` (relative tolerance 1e-13, absolute tolerance 1e-9) and, independently, with the universal-variable solution of Kepler's problem.

| Joint action | `xi` from (2.5) (m) | propagated minus (2.5) (mm) | (2.10) minus (2.5) (mm) | relative change of `Pc` |
|---|---|---|---|---|
| N N | 92.000 | 0.006 | 0.000 | -2.8e-10 |
| N S | 335.485 | 5.202 | 5.203 | -1.3e-4 |
| N L | 822.455 | 8.865 | 8.867 | -5.5e-4 |
| S N | 335.485 | 1.128 | 1.124 | -2.9e-5 |
| S S | 578.970 | 6.325 | 6.326 | -2.8e-4 |
| S L | 1065.939 | 9.987 | 9.990 | -8.0e-4 |
| L N | 822.455 | 10.124 | 10.112 | -6.3e-4 |
| L S | 1065.939 | 15.319 | 15.314 | -1.2e-3 |
| L L | 1552.909 | 18.979 | 18.978 | -2.2e-3 |

- (2.5) against the propagated `xi`: off by at most 18.98 mm (both large burns). As a fraction of the shift the error is at most 2.1e-5 (one small burn by satellite 2).
- (2.10) against the propagated `xi`: largest difference 1.2e-5 m with `ode113` and 7.6e-6 m with the Kepler solution. The test bound is 2e-4 m.
- `zeta` changes by at most 2.5e-6 m.
- The two propagators agree to 1.4e-5 m in position after the lead time and to 1.3e-5 m in the miss vector. The Python construction of the same encounter (`export_matlab.py`, with the propagator of `conjce.maneuver`) agrees with the MATLAB one to 1.3e-5 m in the miss vector and 2.7e-9 s in the time of closest approach.
- The table and the bullets above use the closest approach found by iteration on the two-body orbits, with the axes built from the velocities at that instant. The derivation uses a straight-line closest approach from the nominal time and the nominal axes. The test compares the two constructions: they differ by at most 4.5e-6 m. If the axes are rebuilt from velocities, the velocities have to be taken at the closest approach itself. From (2.8) the velocity directions at the nominal time are `t_i - (d_i / a) zetahat`, so their bisector is tilted out of the horizontal by `eta = (d1 + d2) / (2 a cos(phi/2))`, up to 1.1e-4 rad, and axes built from it are turned about `shat` by that angle; in them the radial component reads `zeta + eta xi`, which is 8.7 cm off for one large burn. By the closest approach, a time `t* = -(d1 + d2) / (2v)` later, gravity has turned both velocities by `v t* / a` and the bisector is horizontal again. `matlab/tests/test_simulink.m` checks this rotation on the output of the Simulink model for the nine joint actions: predicted and observed differences agree within 8.2e-7 m.

Not tested: anything beyond two-body motion.

---

## 3. Operators' costs

### 3.1 Definition

Operator i's cost of the joint action `a = (a1, a2)`:

```
C_i(a) = w_fuel_i * u_i + w_dis_i * 1[u_i > 0] + V_i * Pc(a)        (in USD)
```

`u_i` is its own burn size, `w_fuel_i` the value it puts on 1 m/s of propellant, `w_dis_i` a fixed cost of doing any maneuver (planning, lost service), `V_i` what it loses if its satellite is destroyed.

### 3.2 Costs as fractions of the satellite's value

Divide by `V_i > 0`:

```
c_i(a) = kappa_i(a_i) + Pc(a),        kappa_i(a_i) = ( w_fuel_i u_i + w_dis_i 1[u_i > 0] ) / V_i          (3.1)
```

Dividing a player's whole cost table by a positive number changes none of that player's comparisons, so every equilibrium statement is unchanged. The benefit is that `kappa_i` and `Pc` are now in the same unit: operator i prefers the burn exactly when it removes more collision probability than `kappa_i`. In that sense `kappa_i` is the operator's private maneuver threshold.

### 3.3 Baseline numbers

Propellant is priced as `w_fuel_i = V_i / 100 m/s` (a 100 m/s lifetime budget is worth the satellite), so a 1 cm/s burn costs `1e-4` of the satellite's value in propellant for either operator.

| | Operator 1 | Operator 2 |
|---|---|---|
| Satellite value `V` | $200M | $80M |
| Propellant price `w_fuel` | $2.0M per m/s | $0.8M per m/s |
| Disruption cost `w_dis` | $60k | $16k |
| `kappa` of a small burn (1 cm/s) | 1e-4 + 3e-4 = **4.0e-4** | 1e-4 + 2e-4 = **3.0e-4** |
| `kappa` of a large burn (3 cm/s) | 3e-4 + 3e-4 = 6.0e-4 | 3e-4 + 2e-4 = 5.0e-4 |

Collision probabilities from (1.3) and (2.5), with the short names used below:

| Joint action | `xi` (m) | `Pc` | Name |
|---|---|---|---|
| nobody burns | 92 | 1.488e-3 | `P0` |
| one small burn | 335 | 2.913e-5 | `P1` |
| two small burns | 579 | 6.46e-9 | `P2` |
| one large burn | 822 | 1.6e-14 | |

### 3.4 Why this is a game of chicken

Three comparisons, each "collision probability removed" against "cost of the burn":

```
alone, a small burn pays:                 P0 - P1 = 1.459e-3   >  kappa_i(S) = 4.0e-4 or 3.0e-4
after the other's small burn, mine does not:  P1 - P2 = 2.913e-5   <  kappa_i(S)
a large burn instead of a small one never pays:  extra cost 2.0e-4  >  P1 - 1.6e-14 = 2.913e-5
```

So each operator will burn if the other does not, and will not if the other does. The large burn is strictly dominated by the small one for both operators in the nominal game. It stays in the action set, as specified, and matters only when the true covariance is larger than nominal.

---

## 4. Equilibria and the coordinator's programs

### 4.1 Nash equilibrium (cost form)

A pair of actions is a Nash equilibrium if neither operator can lower its own cost by changing only its own action:

```
c_1(a1, a2) <= c_1(d, a2) for every d,        c_2(a1, a2) <= c_2(a1, d) for every d
```

From section 3.4 the pure equilibria of the baseline are (S, N) and (N, S): exactly one operator does a small burn. `games.pure_nash` finds them by enumeration.

For mixed equilibria `games.support_enumeration` tries every pair of equal-size supports. For supports `I` (operator 1) and `J` (operator 2) it solves two small linear systems: operator 2's mixture `q` on `J` must make operator 1's expected cost the same for every action in `I`, and operator 1's mixture `p` on `I` must do the same for operator 2 on `J`. It keeps the pair if `p` and `q` are probabilities and no action outside the supports is cheaper.

### 4.2 The mixed equilibrium of the baseline

Only N and S are involved. Let operator 2 wait (play N) with probability `q`. Operator 1's expected cost is

```
if it waits:          q P0 + (1 - q) P1
if it burns small:    kappa_1 + q P1 + (1 - q) P2
```

Operator 1 is willing to mix only if the two are equal:

```
q (P0 - P1) + (1 - q)(P1 - P2) = kappa_1
q = ( kappa_1 - (P1 - P2) ) / ( (P0 - P1) - (P1 - P2) )                                 (4.1)
  = (4.0e-4 - 2.913e-5) / (1.4589e-3 - 2.913e-5) = 0.2594
```

By the same argument operator 1 waits with probability `p = (kappa_2 - (P1 - P2)) / ((P0 - P1) - (P1 - P2)) = 0.1894`. Each operator's waiting probability is set by the *other* operator's cost. The probability that neither burns is `p q = 0.0491`.

### 4.3 Correlated equilibrium: the obedience constraints

The coordinator draws a joint action `a` from a table `x` and tells each operator only its own part (Aumann 1974, 1987). Suppose operator 1 is told `r`. By Bayes' rule, the other operator was told `a2` with probability `x(r, a2) / pi_1(r)`, where `pi_1(r) = sum over a2 of x(r, a2)` is the probability of being told `r`.

Operator 1 follows the recommendation if no other action `d` has a lower expected cost:

```
sum over a2 of [x(r, a2) / pi_1(r)] c_1(r, a2)  <=  sum over a2 of [x(r, a2) / pi_1(r)] c_1(d, a2)
```

Multiply by `pi_1(r)` and collect:

```
sum over a2 of  x(r, a2) [ c_1(r, a2) - c_1(d, a2) ]  <=  0        for every r and every d != r      (4.2)
```

This is linear in `x`, and it is automatically true when `pi_1(r) = 0`. With 3 actions there are 6 such constraints per operator, 12 in all. `games.obedience_system` builds them as a 12x9 matrix `A` with `A x <= 0`, together with a matrix `P` such that `P x` gives `pi_i(r)` for each row.

### 4.4 The coordinator's linear program, and its solution for the baseline

```
minimise    sum over a of  x(a) J(a),        J(a) = u1 + u2 + lam * Pc(a)
subject to  A x <= 0,   x >= 0,   sum of x = 1
```

`J` is the coordinator's cost in m/s: total delta-v plus `lam` times collision probability. With `lam = 1000 m/s` the coordinator would spend 1 cm/s to remove 1e-5 of collision probability, where the operators need 3e-4 to 4e-4. Baseline values: `J(S,S) = 0.0200`, `J(L,N) = 0.0300`, `J(S,N) = J(N,S) = 0.0391`, `J(N,N) = 1.488`.

The program can be solved by hand for the baseline, which is how the solver's answer was checked.

**Step 1.** A large burn is never recommended. In the row "operator 1 told L, deviation S" every coefficient `c_1(L, a2) - c_1(S, a2) = 2.0e-4 - (Pc(S, a2) - Pc(L, a2))` is positive, so (4.2) forces `x(L, .) = 0`. The same holds for operator 2.

**Step 2.** (N, N) costs the coordinator 1.488 and is not needed, so the table has three entries: `p = x(S,S)`, `q1 = x(S,N)`, `q2 = x(N,S)`.

**Step 3.** Operator 1 told S believes the other was told N with probability `alpha_1 = q1 / (p + q1)`. It prefers S to N if

```
alpha_1 (P0 - P1 - kappa_1) + (1 - alpha_1)(P1 - P2 - kappa_1) >= 0
alpha_1 >= ( kappa_1 - (P1 - P2) ) / ( (P0 - P1) - (P1 - P2) ) = 0.2594                 (4.3)
```

the same expression as (4.1). For operator 2, `alpha_2 = q2 / (p + q2) >= 0.1894`.

**Step 4.** An operator told N knows the other was told S (the only entry in that row or column) and prefers N because `kappa_i > P1 - P2` (section 3.4).

**Step 5.** `J(S,S) < J(S,N)`, so the coordinator wants `p` as large as the two inequalities of step 3 allow. Both hold with equality: `q1 = p alpha_1 / (1 - alpha_1)`, `q2 = p alpha_2 / (1 - alpha_2)`, and `p + q1 + q2 = 1`:

```
p = 1 / ( 1 + alpha_1/(1 - alpha_1) + alpha_2/(1 - alpha_2) ) = 1 / (1 + 0.3502 + 0.2337) = 0.6313
q1 = 0.2211,      q2 = 0.1476
cost = 0.6313 * 0.0200 + 0.3687 * 0.0391 = 0.02706
```

`games.solve_private` returns the same table to four decimals. In words: 63% of the time both are told to do a small burn; an operator told to burn cannot tell whether the other was told to burn too, and the chance that it was not (26% or 19%) is exactly enough to make burning worthwhile.

### 4.5 Public recommendations

If both operators see the whole recommended joint action `a`, operator i knows the other's action exactly, and it follows the recommendation only if `c_i(a) <= c_i(d, a_-i)` for every `d`. Requiring this of both operators is the definition of a Nash equilibrium (4.1). So a public rule can only randomise over pure Nash equilibria. In the language of (4.2), the constraint must hold for each joint action separately, not only on average over the other's actions.

`games.solve_public` implements this. For the baseline the best public rule is one of the two pure equilibria, cost 0.0391. (A more general public message could also select the mixed equilibrium; `solve_public_hull` allows that, and for the baseline the mixed equilibrium costs 0.0989, so it is never chosen.)

### 4.6 A margin on the obedience constraints, and its link to CC-CE

Ask for slack: conditional on being told `r`, obeying must be cheaper than any deviation by at least `mu`. Multiplying by `pi_i(r)` as before:

```
sum over a_-i of  x(r, a_-i) [ c_i(r, a_-i) - c_i(d, a_-i) ]  +  mu * pi_i(r)  <=  0                  (4.4)
```

or `(A + mu P) x <= 0`. This is still a linear program.

This is the same form as the chance-constrained correlated equilibrium of Im, Topcu and Fridovich-Keil (arXiv:2603.14141). Their model adds one Gaussian term `eta_i ~ N(0, sigma_i^2)` to each of agent i's deviation costs. In the notation here, the left side of (4.2) becomes `M + eta_i pi_i(r)` (the `eta_i` multiplies `sum over a_-i of x(r, a_-i) = pi_i(r)`). This is a Gaussian variable with mean `M` and standard deviation `sigma_i pi_i(r)`, so

```
P( M + eta_i pi_i(r) <= 0 ) = Phi( -M / (sigma_i pi_i(r)) ) >= conf
        <=>   -M / (sigma_i pi_i(r)) >= Phi^-1(conf)
        <=>   M + Phi^-1(conf) sigma_i pi_i(r) <= 0
```

which is (4.4) with `mu = Phi^-1(conf) sigma_i`. The paper states the result (its equation 8) as the standard Gaussian reformulation of a chance constraint and cites Charnes and Cooper (1963) for it; the three lines above are that standard step. The equation numbers are those of the arXiv version read for this project and should be checked against the version being quoted.

The code uses one scalar `mu` for both operators (`solve_private(..., margin=mu, margin_rows="all")`), which is the case `sigma_1 = sigma_2`: the same margin on every constraint. This is the `uniform` family in the results. The `targeted` family of section 4.7 puts `mu` on two of the 12 constraints only; it has no reading as a chance constraint of this kind.

Effect on the baseline: in (4.3), `kappa_i` is replaced by `kappa_i + mu`, so

```
alpha_i  >=  ( kappa_i + mu - (P1 - P2) ) / D  =  alpha_i* + mu / D,        D = (P0 - P1) - (P1 - P2) = 1.42981e-3      (4.5)
```

where `alpha_i*` is the zero-margin threshold of (4.3) (0.2594 and 0.1894). Each 1e-4 of margin raises both thresholds by `1e-4 / D = 0.0699`. As in step 5 of section 4.4 the coordinator makes both inequalities hold with equality, and the table follows from the same formula, `x(S,S) = 1 / (1 + alpha_1/(1 - alpha_1) + alpha_2/(1 - alpha_2))`. Every table of both families in `results.json` (`families`, 19 + 53 tables) has exactly the support (S,S), (S,N), (N,S) and satisfies (4.5) and this formula; the largest difference is 2.2e-16.

| `mu` | family | `alpha_1` | `alpha_2` | `x(S,S)` | `x(S,N)` | `x(N,S)` | coordinator cost |
|---|---|---|---|---|---|---|---|
| 0 | both | 0.2594 | 0.1894 | 0.6313 | 0.2211 | 0.1476 | 0.02706 |
| 1.0e-4 | both | 0.3293 | 0.2594 | 0.5431 | 0.2667 | 0.1902 | 0.02875 |
| 1.8e-4 | both; last member of `uniform` | 0.3853 | 0.3153 | 0.4791 | 0.3003 | 0.2207 | 0.02997 |
| 3.0e-4 | `targeted` | 0.4692 | 0.3993 | 0.3924 | 0.3468 | 0.2608 | 0.03163 |
| 6.0e-4 | `targeted` | 0.6790 | 0.6091 | 0.2140 | 0.4526 | 0.3334 | 0.03504 |
| 1.0e-3 | `targeted` | 0.9588 | 0.8888 | 0.0310 | 0.7211 | 0.2479 | 0.03854 |
| 1.04e-3 | `targeted`; last member | 0.9868 | 0.9168 | 0.0116 | 0.8610 | 0.1274 | 0.03891 |

`alpha_1 = x(S,N) / (x(S,N) + x(S,S))` and `alpha_2 = x(N,S) / (x(N,S) + x(S,S))`. The `uniform` family is on a grid of step 1e-5 (`experiments.STEP_UNIFORM`), the `targeted` family on a grid of step 2e-5 (`experiments.STEP_TARGETED`). At every margin the two families share (0 to 1.8e-4 in steps of 2e-5) the two tables are identical: the extra constraints of the `uniform` family have slack there (section 4.7).

### 4.7 Which constraints need the margin

**The 12 constraints of the baseline.** Each row of (4.2) has one coefficient `c_i(r, a) - c_i(d, a)` per action `a` of the other operator: the cost of obeying minus the cost of deviating if the other plays `a`. From the cost tables of section 3.3 (`games.obedience_system`; units of 1e-4; for operator 2 the "other" is operator 1):

| constraint | other N | other S | other L | signs differ? | other's action uncertain given `r`? | targeted |
|---|---|---|---|---|---|---|
| op 1 told N, dev S | +10.589 | -3.709 | -4.000 | yes | no (always S) | no |
| op 1 told N, dev L | +8.881 | -5.709 | -6.000 | yes | no | no |
| op 1 told S, dev N | -10.589 | +3.709 | +4.000 | yes | yes (N or S) | **yes** |
| op 1 told S, dev L | -1.709 | -2.000 | -2.000 | no | yes | no |
| op 1 told L, dev N | -8.881 | +5.709 | +6.000 | yes | never told L | no |
| op 1 told L, dev S | +1.709 | +2.000 | +2.000 | no | never told L | no |
| op 2 told N, dev S | +11.589 | -2.709 | -3.000 | yes | no (always S) | no |
| op 2 told N, dev L | +9.881 | -4.709 | -5.000 | yes | no | no |
| op 2 told S, dev N | -11.589 | +2.709 | +3.000 | yes | yes (N or S) | **yes** |
| op 2 told S, dev L | -1.709 | -2.000 | -2.000 | no | yes | no |
| op 2 told L, dev N | -9.881 | +4.709 | +5.000 | yes | never told L | no |
| op 2 told L, dev S | +1.709 | +2.000 | +2.000 | no | never told L | no |

For example, operator 1 told S, deviation L, other N: `c_1(S,N) - c_1(L,N) = (kappa_1(S) - kappa_1(L)) + (P1 - Pc(L,N)) = -2.0e-4 + 2.913e-5 = -1.709e-4`. The column "uncertain" refers to the support (S,S), (S,N), (N,S) shared by every rule of both families.

**(a) Why the same margin on every constraint stops at 1.8e-4.** Take operator i told S, with `alpha_i` the probability that the other was told N. Its conditional advantages (obeying minus deviating, both divided by `pi_i(S)`) are

```
deviation N:  alpha_i (P0 - P1 - kappa_i) + (1 - alpha_i)(P1 - P2 - kappa_i)  =  alpha_i D - (kappa_i - (P1 - P2))          (4.6)
deviation L:  alpha_i [2e-4 - (P1 - Pc(L,N))] + (1 - alpha_i) [2e-4 - (P2 - Pc(L,S))]  =  F - alpha_i E                     (4.7)

E = (P1 - Pc(L,N)) - (P2 - Pc(L,S)) = 2.91272e-5,        F = 2e-4 - (P2 - Pc(L,S)) = 1.99994e-4
```

where `2e-4 = kappa_i(L) - kappa_i(S)` for both operators (section 3.3). An operator told N knows the other was told S, so its advantages are constants: `kappa_i - (P1 - P2)` against S (3.709e-4 and 2.709e-4) and `kappa_i(L) - (P1 - Pc(L,S))` against L (5.709e-4 and 4.709e-4).

With the margin on every constraint, (4.6) `>= mu` gives (4.5), `alpha_i >= alpha_i* + mu/D`, and (4.7) `>= mu` gives `alpha_i <= (F - mu)/E`. The coordinator wants `alpha_i` as small as possible, so (4.6) holds with equality; then (4.7) requires

```
F - E (alpha_i* + mu/D)  >=  mu
mu (1 + E/D)  <=  F - E alpha_i*
mu  <=  (F - E alpha_i*) / (1 + E/D)
```

With `E/D = 0.020371`: operator 1, `(1.99994e-4 - 2.91272e-5 x 0.259386) / 1.020371 = 1.92438e-4 / 1.020371 = 1.8860e-4`; operator 2, `(1.99994e-4 - 5.5180e-6) / 1.020371 = 1.9059e-4`. Both are below the limits of the told-N constraints (2.709e-4 and above), so the cap is 1.886e-4, set by operator 1's constraint "told S, deviation L". Above it no `alpha_1` satisfies both (4.6) and (4.7) with margin `mu`, so operator 1 cannot be told S, and neither operator can be told L (section 4.4, step 1); then operator 2, if told S, knows operator 1 waits, its advantage against L is `F - E = 1.709e-4 < mu`, so it cannot be told S either, and (N, N) fails operator 2's "told N, deviation S" constraint (coefficient +11.589e-4). Nothing is left. `games.solve_private(c1, c2, obj, margin=1.88e-4, margin_rows="all")` returns a table and `margin=1.886e-4` returns none (both run on the baseline tables). On the 1e-5 grid the last member is 1.8e-4 (`scheme_info.mu_uniform_max`).

The constraint that causes the cap has coefficients -1.709e-4, -2.000e-4, -2.000e-4: all negative. Whatever an operator believes about the other's action, its estimated advantage of S over L is a weighted average of these three numbers with non-negative weights summing to one, so it is at least 1.709e-4. No estimate of the rule, however wrong, can make an operator told S switch to L. `games.belief_dependent_rows` (entries of both signs) returns False for this constraint. A margin on it buys no protection against estimation error; it only caps the family. (In the chance-constrained reading of section 4.6 it protects against errors in the costs of this constraint, which is a different job.)

**(b) The targeted constraints, as coded.** `games.uncertain_belief_rows(x0, c1, c2)` selects constraint `(player, r, d)` when both hold:

1. its coefficients have both signs (`belief_dependent_rows`), so its sign depends on what the operator believes, and
2. under the table `x0`, more than one action of the other operator has probability above `tol = 1e-9` given `r`.

`experiments.baseline_schemes` calls it with `x0` = the zero-margin rule. Condition 1 keeps 8 of the 12 constraints (table above). Condition 2 removes "told N" (the other was told S, always) and "told L" (never recommended). What is left is "told S, deviation N" for each operator, which is `scheme_info.targeted_rows = [[0, 1, 0], [1, 1, 0]]` in the results. The family is then `experiments.margin_family(c1, c2, obj, mask, step=2e-5)`.

**(c) The obedience condition of the targeted constraint, and where the family stops.** By (4.6), "told S, deviation N" with margin `mu` holds when `alpha_i D - (kappa_i - (P1 - P2)) >= mu`, that is, when the conditional probability that the other was told to wait is at least the threshold `alpha_i* + mu/D` of (4.5). The margin moves this threshold up by `mu/D`. A probability cannot exceed 1, so the largest margin operator i's constraint can carry is reached at `alpha_i = 1`:

```
mu_max,i = D (1 - alpha_i*) = D - (kappa_i - (P1 - P2))
         = (P0 - P1) - (P1 - P2) - kappa_i + (P1 - P2)
         = P0 - P1 - kappa_i
```

that is, what a small burn saves when the operator is sure the other waits. Operator 1: `1.488073e-3 - 2.913e-5 - 4.0e-4 = 1.0589e-3`; operator 2: `1.1589e-3`. The margin is common to both operators, so operator 1 sets the limit. As `mu` approaches 1.0589e-3, `alpha_1` approaches 1 and `x(S,S) = 1 / (1 + alpha_1/(1 - alpha_1) + alpha_2/(1 - alpha_2))` approaches 0 (0.0116 at the last member, 1.04e-3). At `alpha_1 = 1` the table has only (S,N) and (N,S): each recommendation then tells its receiver exactly what the other was told, which is what `games.reveals_other` tests. Such a rule is a mixture of the two pure Nash equilibria, a public rule in effect, and `margin_family` stops there. On the 2e-5 grid: `solve_private(c1, c2, obj, margin=1.06e-3, margin_rows=mask)` returns the point table on (N, S), for which `reveals_other` is True; `margin=1.2e-3` (above operator 2's limit as well) returns none (both run). The last private member is 1.04e-3 (`scheme_info.mu_targeted_max`).

The constraint "told S, deviation L" carries no margin in this family and never binds: by (4.7) its advantage is at least `F - E = 1.709e-4 > 0` for any `alpha_i <= 1`. The told-N constraints have advantages of at least 2.709e-4 and carry no margin. So for `mu <= 1.8e-4` the `uniform` family, whose extra margins are not yet binding, gives the same tables as the `targeted` one (section 4.6).

Numerical check of (c) against `results.json` (`families.targeted`): at `mu = 6e-4`, `alpha_1 = 0.259386 + 6e-4 / 1.429813e-3 = 0.679021` (table: 0.6790); at `mu = 1.04e-3`, `alpha_1 = 0.259386 + 1.04e-3 / 1.429813e-3 = 0.259386 + 0.727368 = 0.986754` (table: 0.9868).

**(d) Why the other belief-dependent constraints need no slack against estimation error here.** Six belief-dependent constraints carry no targeted margin: "told N, deviation S", "told N, deviation L" and "told L, deviation N" for each operator.

- "Told N": in every rule of both families, an operator told N was told so only together with "other told S" (row N of `x` has only (N,S), column N only (S,N)). Under the convention used in every experiment (section 5.1, `start="observed"`), each operator's counts for "told N" consist of one initial observation drawn from this conditional, which is S with probability 1, plus shared past events in which the other was also told S. Its belief is therefore exactly (0, 1, 0) for every `k`, and its estimated advantages equal the true ones, 3.709e-4 and 5.709e-4 for operator 1, 2.709e-4 and 4.709e-4 for operator 2. There is no estimation error to protect against.
- "Told L": the large burn is never recommended, so these constraints are never faced.

In the Hoeffding bound of section 5.3 the told-N constraints accordingly contribute nothing (obeying saves at least as much as deviating for every action of the other that can occur). Under the other convention (`start="uniform"`) an operator that has never been told N has a uniform belief and burns when told N (section 5.4). That is a missing observation, not an estimation error, and no margin changes it, because the response to a uniform belief does not depend on `x`.

The `uniform` margin is the one that corresponds to CC-CE and protects against cost errors on every constraint. The `targeted` margin is the one aimed at learning. RESULTS.md reports both.

---

## 5. Operators who learn the rule from past events

### 5.1 The model

This is the two-receiver version of the inference setting of Karabag et al. (arXiv:2310.00468) and Probine, Karabag and Topcu (arXiv:2510.01434).

- The coordinator uses one fixed table `x` for every event and does not announce it.
- After each past event the joint recommendation was disclosed to both operators, so the past events are shared.
- For each recommendation `r` it can receive, operator i counts how often the other operator was told each action in the events in which i was told `r`. Its belief given `r` is the empirical frequency of these counts.
- How the counts start is a modelling convention. `inference.realised_joint` has two, chosen by its argument `start`:
  - **A, "one observation each"** (`start="observed"`, the default, used in every experiment and figure). The counts are `k - 1` shared past events, drawn independently from `x` (a multinomial over the joint recommendations), plus, for each operator separately and for each recommendation `r` with `pi_i(r) > 0`, one observation of the other's recommendation drawn from the true conditional `x(r, .) / pi_i(r)`. The two operators' extra observations are drawn independently of each other and of the shared events. This needs `k >= 1`. Every recommendation the operator can receive has at least one count, so every belief is an observed frequency. This is the convention of the two papers.
  - **B, "uniform when unseen"** (`start="uniform"`, used only in the exact check of section 5.4). The counts are exactly `k` shared past events drawn from `x` and nothing else (`k >= 0`); both operators have the same counts. A recommendation that did not occur in those `k` events leaves a uniform belief: 1/3 on each of the other's three actions, the never-recommended large burn included.
- In the new event operator i is told `r`, computes its expected cost of each action under its belief with its own true costs, and follows `r` if the expected cost of `r` is within a tie tolerance of the cheapest; otherwise it plays the cheapest action. The tolerance is 1e-9 times the largest entry of the two cost tables, 1.5e-12 in the baseline. `k = None` stands for operators who know `x`: their belief is the true conditional.

`inference.realised_joint` simulates this. For each simulated history it computes both operators' responses to every recommendation and then takes the expectation over the new event's recommendation exactly, so the only Monte Carlo noise is over histories. `inference.exact_realised_joint` computes the same expectations without sampling (section 5.4).

### 5.2 Why the rule of section 4.4 cannot be learned

Take operator 1 told S under the table of section 4.4. What it saves by obeying, compared with waiting, depends on what the other was told:

```
other told N:   g_N = P0 - P1 - kappa_1 = +1.059e-3
other told S:   g_S = P1 - P2 - kappa_1 = -3.709e-4
```

With belief `alphahat` that the other was told N, the estimated saving is `alphahat g_N + (1 - alphahat) g_S`, which is positive exactly when `alphahat > 0.2594`, the threshold of (4.3). But the coordinator chose the table so that the true `alpha_1` *equals* 0.2594. The estimate `alphahat` is a sample frequency centred on the true value, so it falls below the threshold about half the time, for any number of observations (central limit theorem: the estimate is symmetric around the truth to first order). More data shrinks the error but never moves the truth off the threshold.

With the rule known the operator is exactly indifferent and follows the recommendation only because ties go to obeying (tolerance 1.5e-12, section 5.1).

The numbers (costs known, convention A) show it. `inference.exact_realised_joint(x0, c1, c2, k, start="observed")`, with `x0` the zero-margin table and the baseline cost tables, gives the probability that an operator told S follows the recommendation as 0.475 (operator 1) and 0.489 (operator 2) at `k = 30`, and 0.497 and 0.492 at `k = 300`. In the stored learning curves (`learning_curves.private_m0`, 40,000 simulated histories per `k`) both operators follow in 34% to 36% of events at every `k` from 30 to 10,000, and the realised coordinator cost is between 0.52 and 0.61 at every `k` from 3 to 10,000 (1.00 at `k = 1`, 0.71 at `k = 2`), against the 0.0271 the table promises and 0.0391 for the public rule. When both are told to burn and both drop out, nobody burns.

The one-receiver version of this effect is known: Probine et al. note that schemes that are optimal under known commitment put the receiver on a decision boundary and do badly for every `k`, and Sood and Best (arXiv:2510.25066) show the probability of a credible recommendation tends to one half at the classical optimum.

### 5.3 With a margin: a Hoeffding bound

Fix operator i, a recommendation `r` with probability `pi_r`, and a deviation `d`. For each action `a` of the other operator let `g(a) = c_i(d, a) - c_i(r, a)` be what obeying saves. Let `gamma` be its mean under the true conditional distribution (the conditional advantage; at least `mu` by (4.4) on a constraint that carries the margin) and `W = max g - min g` over the actions that can occur with `r`.

Suppose the operator has `n` observations for recommendation `r`. Its estimated saving is the average of `n` independent values of `g`, each within a range of width `W`, with mean `gamma`. Hoeffding's inequality says an average of `n` such values falls below its mean by `t` or more with probability at most `exp(-2 n t^2 / W^2)`. The operator deviates to `d` only if the average is below zero, that is, below its mean by `gamma`:

```
P( deviates to d | n observations )  <=  exp( -2 n gamma^2 / W^2 ) = exp(-s n),        s = 2 gamma^2 / W^2        (5.1)
```

`n` is random. Under convention A (section 5.1) it is the one starting observation plus the number of the `k - 1` shared past events in which this operator was told `r`, which is Binomial(`k - 1`, `pi_r`); given `n`, the `n` observed actions of the other are independent draws from the true conditional. (The bound is derived for convention A only.) Average (5.1) over `n = 1 + B`, using `E[z^B] = (1 - pi_r + pi_r z)^(k-1)` for a binomial `B` with `z = exp(-s)`:

```
P( deviates to d )  <=  exp(-s) * ( 1 - pi_r (1 - exp(-s)) )^(k - 1)                    (5.2)
```

Finally, the probability that at least one operator does not follow the recommendation it receives in the new event is at most the sum of (5.2) over deviations, weighted by the probability `pi_r` of receiving each recommendation, over both operators (union bound):

```
P( some recommendation not followed )  <=  sum over i, r, d of  pi_i(r) * exp(-s) (1 - pi_i(r)(1 - exp(-s)))^(k - 1)      (5.3)
```

Constraints where obeying saves at least as much as deviating for every action of the other that can occur with `r` (`g >= 0` on the support of the conditional) contribute nothing; a constraint whose true advantage `gamma` is not positive contributes `pi_i(r)` (no information). The sum is capped at 1. `inference.hoeffding_disobedience_bound` computes (5.3).

How to read it. For small `s`, (5.2) is about `exp(-2 k pi_r gamma^2 / W^2)`: the failure probability falls exponentially in (number of past events) x (probability of this recommendation) x (margin / payoff range) squared. Three things make a rule easy to learn: a large margin, few distinct recommendations so each is seen often, and many events. The same three ingredients appear in the one-receiver bounds of Probine et al. (number of signals, distance from the decision boundary) and Zhang and Li (arXiv:2608.20766); the Hoeffding-plus-union-bound argument for samples from a correlated equilibrium is in Babichenko, Barman and Peretz (arXiv:1310.7654). (5.3) is that standard argument written per recommendation. It is not a new technique.

### 5.4 Exact enumeration and the starting convention

**How `inference.exact_realised_joint` enumerates histories.** Let `s` be the number of joint recommendations with `x(a) > 0` (3 for every rule of both families: (S,S), (S,N), (N,S)) and `n_ev` the number of shared past events (`k - 1` under A, `k` under B). The shared events matter only through their count vector `n = (n_1, ..., n_s)` with `n_1 + ... + n_s = n_ev`, and its probability is multinomial:

```
w(n) = n_ev! / (n_1! ... n_s!) * p_1^n_1 * ... * p_s^n_s,        p_j = x(a_j)
```

There are `C(n_ev + s - 1, s - 1)` such vectors (`inference.n_histories`): for `s = 3` and `k = 300`, 45,150 under A and 45,451 under B. The function refuses to enumerate more than 5,000,000.

Under B both operators' counts are the shared counts, so for each `n` the code computes each operator's response to each recommendation with the same function the simulation uses. Under A each operator also has one initial observation per recommendation `r`, drawn from the true conditional of `r`. That observation enters only the belief given `r`, and it is independent of the shared events and of the other operator's initial observations. So for each `n` the code computes `d_i(n, r, a)`, the probability that operator i plays `a` when told `r`, by averaging its response over the (at most three) possible initial observations, weighted by their conditional probabilities (under B, `d_i` is 0 or 1). Given `n`, the two operators' responses are independent, and the new event's recommendation `(r1, r2)` is drawn from `x` independently of the past. Hence

```
y(a1, a2)      = sum over n of  w(n) * sum over (r1, r2) of  x(r1, r2) d_1(n, r1, a1) d_2(n, r2, a2)
P(both follow) = sum over n of  w(n) * sum over (r1, r2) of  x(r1, r2) d_1(n, r1, r1) d_2(n, r2, r2)
```

These are the expectations that `realised_joint` estimates by sampling, without sampling error.

**Comparison with the simulation.** `exact_checks.simulation_agreement` compares the two in 8 cases: the rule with the same margin on every constraint at `mu = 1e-4` and the targeted rule at `mu = 6e-4`, each under conventions A and B, at `k = 5` and `k = 50`, with 200,000 simulated histories per case. In each case it compares 12 quantities: the probability that both operators follow, that each one follows, and the 9 cells of `y`; 96 quantities in all. Each difference is divided by the standard error of the simulated mean. When every simulated history gave the same value (sample standard deviation below 1e-9), the standard error is replaced by `sqrt((1 - P_both) / n)`, which bounds the true one: in each history every compared quantity differs from its full-obedience value by at most the history's probability `q` that some recommendation is not followed, so its variance is at most `E[q^2] <= E[q] = 1 - P_both`. Result (`exact_checks.simulation_agreement.max_z` in the results): the largest difference over the 96 quantities is 2.49 standard errors, for the same-margin rule under A at `k = 50`. `tests/test_exact.py` requires at most 4 standard errors in every case; it also compares simulation and enumeration on a random table that uses all nine cells (`k` = 1 and 3, both conventions) and checks a two-cell table under B at `k = 1` against a calculation by hand.

**Why an operator with a uniform belief does a small burn.** With belief 1/3 on each of the other's three actions, operator 1's expected costs are, from the tables of section 3.3,

```
N:  (P0 + P1 + Pc(N,L)) / 3                              = (1.488073e-3 + 2.9134e-5 + 1.6e-14) / 3      = 5.057e-4
S:  kappa_1(S) + (P1 + P2 + Pc(S,L)) / 3                 = 4.0e-4 + (2.9134e-5 + 6.5e-9 + 4.6e-22) / 3   = 4.097e-4
L:  kappa_1(L) + (Pc(L,N) + Pc(L,S) + Pc(L,L)) / 3       = 6.0e-4 + (1.6e-14 + 4.6e-22 + 5.4e-43) / 3    = 6.000e-4
```

Operator 2's are the same with `kappa_2`: 5.057e-4, 3.097e-4 and 5.000e-4. The small burn is the cheapest for both. A uniform belief is the same whatever the recommendation, so an operator holding one does a small burn whether it is told N or S.

What this does to the coordinator's cost. Under B, an unseen recommendation S is followed and an unseen recommendation N is turned into S. A recommended (S, N) or (N, S) then becomes (S, S), which costs the coordinator 0.0200 instead of 0.0391 (section 4.4), and an unseen recommendation never leads to (N, N). Under A every recommendation has been seen at least once, and a single observation can mislead: an operator told S whose only observation for S is "the other was told S" waits (`c_1(N,S) = 2.913e-5 < c_1(S,S) = 4.0e-4`), and if the other was in fact told N, nobody maneuvers.

Worked example: the targeted rule at `mu = 1.0e-3` (`x(S,S) = 0.0310`, `x(S,N) = 0.7211`, `x(N,S) = 0.2479`, section 4.6) at `k = 1`.

Under B the operators share one past event. The responses follow from section 3.4 (an operator sure the other waits burns; sure the other burns, it waits) and from the uniform-belief result above:

| past event (probability) | op 1 told S | op 1 told N | op 2 told S | op 2 told N | new (S,S) | new (S,N) | new (N,S) |
|---|---|---|---|---|---|---|---|
| (S,N) (0.7211) | S (other waits) | S (unseen) | S (unseen) | N (other burns) | (S,S) | (S,N) | (S,S) |
| (N,S) (0.2479) | S (unseen) | N (other burns) | S (other waits) | S (unseen) | (S,S) | (S,S) | (N,S) |
| (S,S) (0.0310) | N (other burns) | S (unseen) | N (other burns) | S (unseen) | (N,N) | (N,S) | (S,N) |

With `J(S,S) = 0.020006`, `J(S,N) = J(N,S) = 0.039134`, `J(N,N) = 1.488073`:

```
cost = x(S,N) [ (x(S,S) + x(N,S)) J(S,S) + x(S,N) J(S,N) ]
     + x(N,S) [ (x(S,S) + x(S,N)) J(S,S) + x(N,S) J(N,S) ]
     + x(S,S) [ x(S,S) J(N,N) + (x(S,N) + x(N,S)) J(S,N) ]
     = 0.7211 x 0.033799 + 0.2479 x 0.024748 + 0.0310 x 0.084051
     = 0.02437 + 0.00614 + 0.00261 = 0.0331
```

`exact_checks` stores 0.03311 for this case (`best_per_k.targeted.uniform`, `k = 1`). Only 61% of recommendations are followed (`obey` = 0.611 in the same entry), yet the cost is 15% below the public rule's 0.03913, because most of the recommendations not followed are an N turned into S.

Under A the same rule at `k = 1` has no shared event; each operator has one observation per recommendation. Operator i told N always observed "other told S" and obeys. Operator i told S observed "other told N" with probability `alpha_i` and burns, otherwise it waits. So nobody maneuvers with probability

```
x(S,N)(1 - alpha_1) + x(N,S)(1 - alpha_2) + x(S,S)(1 - alpha_1)(1 - alpha_2)
  = 0.7211 x 0.0412 + 0.2479 x 0.1112 + 0.0310 x 0.0412 x 0.1112  =  0.0574
```

and `exact_realised_joint(x, c1, c2, 1, start="observed")` gives (N,N) probability 0.0574 and cost 0.1218 for this table (run for this document).

**Best cost per `k` under both conventions.** `exact_checks.run` evaluates every member of both families exactly at every `k` from 1 to 40 and at 50, 70, 100, 150, 200 and 300 (the same-margin family also at every `k` from 41 to 130), and keeps the cheapest member (`best_per_k` in the results; costs known; margin of the cheapest member in brackets). The public rule costs 0.03913 at every `k` under both conventions (its learned and known-rule costs differ by 0.0, `public_learned_max_abs_deviation`).

| `k` | targeted, A | targeted, B | same margin, A | same margin, B |
|---|---|---|---|---|
| 1 | 0.07083 (1.04e-3) | 0.03311 (1.00e-3) | 0.81652 | 0.36439 |
| 2 | 0.05462 (1.04e-3) | 0.03486 (9.4e-4) | 0.53466 | 0.19531 |
| 3 | 0.05094 (1.04e-3) | 0.03610 (8.4e-4) | 0.36661 | 0.11225 |
| 5 | 0.04784 (1.04e-3) | 0.03768 (1.02e-3) | 0.35281 | 0.20543 |
| 9 | 0.04159 (9.2e-4) | 0.03819 (9.4e-4) | 0.31558 | 0.27355 |
| 10 | 0.04033 (9.0e-4) | 0.03810 (9.2e-4) | 0.27922 | 0.24886 |
| 12 | 0.03885 (8.8e-4) | 0.03789 (8.8e-4) | 0.23802 | 0.21849 |
| 13 | 0.03843 (8.6e-4) | 0.03781 (8.6e-4) | 0.24194 | 0.23273 |
| 15 | 0.03786 (8.2e-4) | 0.03765 (8.2e-4) | 0.21895 | 0.21472 |
| 20 | 0.03672 (7.0e-4) | 0.03687 (7.0e-4) | 0.17873 | 0.18070 |
| 30 | 0.03516 (5.4e-4) | 0.03526 (5.6e-4) | 0.11923 | 0.12011 |
| 95 | | | 0.03899 | 0.03911 |
| 96 | | | 0.03867 | 0.03878 |
| 97 | | | 0.03840 | 0.03849 |
| 100 | 0.03162 (2.8e-4) | 0.03163 (2.8e-4) | 0.03774 | 0.03781 |
| 300 | 0.02977 (1.6e-4) | 0.02977 (1.6e-4) | 0.02973 (1.5e-4) | 0.02973 (1.5e-4) |

The cheapest member of the same-margin family is its last one, 1.8e-4, at every `k` up to 200, under both conventions.

Crossover (`exact_checks.crossover`): the first `k` at which the cost is at most 0.99 times the public rule's, 0.038742.

| | convention A | convention B |
|---|---|---|
| targeted margin | 13 | 1 |
| same margin on every constraint | 96 | 97 |

In all four cases no later `k` of the grid fails the criterion (`later_k_failing` is empty). Under B the targeted rule is below the threshold at every `k`: its cost rises from 0.0331 at `k = 1` to its largest value, 0.0382 at `k = 9`, and falls after that. From `k = 20` on, the two conventions give costs within 0.41% of each other for the targeted family (largest difference at `k = 21`) and within 2.1% for the same-margin family. The simulation grid of the experiments (`crossover_k` in the results: 15 and 100, convention A; criterion: at least 1% and three standard errors below the public rule at that grid point and every later one, with the margin chosen on one set of histories and evaluated on another) has no point between 10 and 15 or between 70 and 100, so its crossovers are the first grid points at or after 13 and 96.

What this shows, in this model: where operators have seen many events, the starting convention does not matter. Where they have seen few, it decides the answer, because it fixes what an operator does with a recommendation it has barely or never seen. The number of past events after which the private rule wins therefore measures how far operators trust thin evidence; it is not a constant of the problem. The same-margin rule is barely affected (96 or 97), because at small `k` it is far above the public rule under either convention.

Numbers for the rule with the same margin on every constraint, `mu = 1.8e-4` (scheme `private_uniform`; operator 1 told S, deviation N: `gamma = 1.8e-4`, `W = 1.0589e-3 + 3.709e-4 = 1.4298e-3`, `s = 0.0317`, `pi_r = 0.779`). Probability that some recommendation is not followed, convention A:

| `k` | bound (5.3) | exact | simulated (40,000 histories) |
|---|---|---|---|
| 30 | 0.727 | 0.141 | 0.139 |
| 100 | 0.142 | 0.0144 | 0.0147 |
| 300 | 0.0014 | 3.5e-5 | 0 (no history with a deviation) |

The bound column is `learning_curves.private_uniform[...].hoeffding` in the results; "exact" is `1 - exact_realised_joint(x, c1, c2, k, start="observed")["obey"]` with `x` that rule's table (run for this table); "simulated" is `1 - obey` from `learning_curves.private_uniform`.

The bound is valid and loose, as Hoeffding bounds are; it gets the exponential rate right. What the tests check:

- `tests/test_inference.py`, against exact enumeration: for every table of both margin families (19 + 53 tables) and every `k` from 1 to 60, convention A, the bound is at least the exact probability that some recommendation is not followed; at least one of these comparisons is against a bound below 1.
- the same file, against simulation: for the rules with the same margin on every constraint at `mu` = 5e-5, 1e-4 and 1.8e-4 and `k` = 5, 50, 500 and 5000 (40,000 histories), the simulated probability does not exceed the bound by more than 4 `sqrt(p / 40000)`, an allowance for Monte Carlo noise.
- a hand-made 2x2 game where (5.3) has a closed form: the code matches it to a relative 1e-12 at `k` = 1, 2, 10, 100 and 1000, and the bound is above the exact probability for `k` = 1 to 60.
- the bound is 1 for the zero-margin rule at `k = 1000` and 0 for a pure Nash profile used as a public rule.

---

## 6. Guarantees from samples

### 6.1 The scenario approach, and why its plain form fails here

To make a constraint hold with probability at least `1 - eps` over uncertain data, the scenario approach samples `N` realisations and imposes the constraint for each. For a convex program with `d` decision variables that is feasible for every sample, the solution violates the constraint on a new realisation with probability more than `eps` only with probability at most

```
beta = sum from i = 0 to d - 1 of  C(N, i) eps^i (1 - eps)^(N - i)                      (6.1)
```

(Campi and Garatti 2008). The earlier explicit condition of Calafiore and Campi (2006) is `N >= (2/eps) ln(1/beta) + 2d + (2d/eps) ln(2/eps)`. Here `x` has 9 entries, so `d = 9`. (Eliminating one entry through `sum x = 1` would allow `d = 8`; `d = 9` is the conservative choice and is what the code uses.) For `eps = 0.05` and `beta = 1e-6`, (6.1) gives `N = 608` and the explicit formula gives `N = 1899`. Calafiore (2010) treats random convex programs that may be infeasible; it was read only in part and no number here relies on it. "Violation" means that at least one of the 12 obedience constraints fails in the new game, which is the event measured in RESULTS.md. The same sampled-constraint idea, with the same kind of guarantee, is used for chance-constrained games in Scenario-Game ADMM (Li et al., arXiv:2304.01945).

`scenario.solve_scenario_private` stacks the `12 N` rows and solves the program. With the baseline uncertainty (`experiments.scenario_program_check`: log-standard deviation 0.3 on each operator's propellant price, disruption cost and satellite value, 0.1 on the covariance scale, seed 5) it is **infeasible** for both sample sizes.

**Why: the conflict found by the code.** For an infeasible program, `experiments.infeasible_subset` runs a deletion filter (drop blocks of sampled games, then single constraints, as long as the rest stays infeasible) and returns a set of constraints that is infeasible on its own while every proper subset is feasible. `describe_conflict` stores, for each member, its coefficients and the sampled parameters of its game (`scenario_program.runs[i].conflict` in the results). In both runs the set has six constraints from three games, and the code's re-checks report it infeasible and irreducible (`verified_infeasible`, `verified_irreducible`: both True).

Each constraint reads `sum over the other's action a of coefficient(a) x(r, a) <= 0` (form (4.2)); the coefficients below are copied from the results (units of 1e-4, other's action N, S, L). The maneuver costs follow from the stored parameters with (3.1): `kappa_i(S) = (w_fuel_i x 0.01 m/s + w_dis_i) / V_i`, and the large burn costs `w_fuel_i x 0.02 m/s / V_i` more than the small one. "Covariance scale" is the sampled factor on both sigmas (`k` in the stored record; not a number of events).

`N = 608` (games 217, 251, 270):

| game | covariance scale | constraint | coefficients | what is unusual about the game |
|---|---|---|---|---|
| 217 | 1.051 | op 1 told S, dev N | -0.446, +13.000, +13.405 | operator 1's small burn costs `(2.192e6 x 0.01 + 1.424e5) / 1.226e8 = 1.3405e-3`, 3.35 times nominal |
| 251 | 1.315 | op 1 told L, dev S | +1.008, +2.140, +2.149 | |
| 251 | 1.315 | op 2 told S, dev L | +0.414, -0.718, -0.727 | large covariance; operator 2's large burn costs only `3.307e5 x 0.02 / 9.102e7 = 7.27e-5` more than its small one |
| 270 | 1.200 | op 1 told S, dev L | +0.046, -0.758, -0.761 | large covariance; operator 1's large burn costs only `1.105e6 x 0.02 / 2.904e8 = 7.61e-5` more |
| 270 | 1.200 | op 2 told N, dev L | +9.451, -2.165, -2.972 | |
| 270 | 1.200 | op 2 told L, dev S | +0.220, +1.025, +1.027 | |

The residual collision probability of a lone small burn in these games can be read off the stored coefficients. In game 251, "op 1 told L, dev S" against N is `c_1(L,N) - c_1(S,N) = [kappa_1(L) - kappa_1(S)] - [Pc(S,N) - Pc(L,N)] = 1.008e-4`, with `kappa_1(L) - kappa_1(S) = 2.181e6 x 0.02 / 2.029e8 = 2.149e-4`, so `Pc(S,N) - Pc(L,N) = 1.141e-4`. "Op 2 told S, dev L" against N gives the same number from operator 2's side: `4.14e-5 + 7.27e-5 = 1.141e-4`. In game 270, "op 1 told S, dev L" against N gives `Pc(S,N) - Pc(L,N) = 0.46e-5 + 7.61e-5 = 8.07e-5`. The nominal value is 2.913e-5 (section 3.3). Section 1.6 explains why: after a burn the pass point is on the Gaussian tail, where a larger covariance raises the collision probability steeply.

The argument, step by step (all entries of `x` are non-negative and sum to 1):

1. Game 251, op 1 told L, dev S: all three coefficients are positive, so `1.008 x(L,N) + 2.140 x(L,S) + 2.149 x(L,L) <= 0` forces `x(L,N) = x(L,S) = x(L,L) = 0`. Operator 1 is never told L.
2. Game 270, op 2 told L, dev S: likewise `x(N,L) = x(S,L) = x(L,L) = 0`. These two constraints have the form of section 4.4, step 1 (a large burn costs more than it removes whatever the other does). Only (N,N), (N,S), (S,N), (S,S) remain.
3. Game 217, op 1 told S, dev N: `-0.446 x(S,N) + 13.000 x(S,S) <= 0`, so `x(S,S) <= 0.0343 x(S,N)`. With a small burn costing 1.3405e-3, operator 1 burns only if it is almost sure the other waits.
4. Game 270, op 1 told S, dev L: `0.046 x(S,N) - 0.758 x(S,S) <= 0`, so `x(S,S) >= 0.0611 x(S,N)`. In this game a lone small burn leaves 8.07e-5, more than the 7.61e-5 the large burn costs extra, so operator 1, if alone, prefers the large burn; it accepts "small burn" only if the other burns often enough.
5. Steps 3 and 4 together: `0.0611 x(S,N) <= x(S,S) <= 0.0343 x(S,N)`, possible only with `x(S,N) = 0`, and then `x(S,S) = 0`. Operator 1 is never told S.
6. Game 251, op 2 told S, dev L: `0.414 x(N,S) - 0.718 x(S,S) <= 0` with `x(S,S) = 0` gives `x(N,S) = 0`. Operator 2 told to burn while operator 1 waits would, in this game, add the large burn (lone small burn leaves 1.141e-4, the large burn costs 7.27e-5 extra).
7. Game 270, op 2 told N, dev L: `9.451 x(N,N) - 2.165 x(S,N) <= 0` with `x(S,N) = 0` gives `x(N,N) = 0`. If nobody else burns, operator 2 prefers a large burn to none.

All nine entries are zero, which contradicts `sum x = 1`.

`N = 1899` (games 1671, 1828, 1835) has the same structure with the operators' roles partly swapped:

| game | covariance scale | constraint | coefficients | what is unusual about the game |
|---|---|---|---|---|
| 1671 | 1.375 | op 2 told S, dev L | +0.626, -0.664, -0.679 | large covariance; operator 2's large burn costs `3.139e5 x 0.02 / 9.243e7 = 6.79e-5` more |
| 1828 | 1.256 | op 1 told L, dev S | +0.636, +1.601, +1.606 | |
| 1828 | 1.256 | op 2 told S, dev N | -2.355, +7.469, +8.435 | operator 2's small burn costs `(8.964e5 x 0.01 + 2.482e4) / 4.005e7 = 8.43e-4`, 2.81 times nominal (its satellite value is half the nominal one) |
| 1835 | 1.456 | op 1 told S, dev L | +0.002, -1.476, -1.505 | large covariance; operator 1's large burn costs `2.047e6 x 0.02 / 2.721e8 = 1.505e-4` more, and the lone small burn leaves 2.2e-7 more than that |
| 1835 | 1.456 | op 2 told N, dev L | +1.320, -6.821, -8.328 | |
| 1835 | 1.456 | op 2 told L, dev S | +1.978, +3.457, +3.485 | |

1. Game 1828, op 1 told L, dev S, and game 1835, op 2 told L, dev S: all coefficients positive, so neither operator is told L.
2. Game 1828, op 2 told S, dev N (the other is operator 1): `-2.355 x(N,S) + 7.469 x(S,S) <= 0`, so `x(S,S) <= 0.3154 x(N,S)`.
3. Game 1671, op 2 told S, dev L: `0.626 x(N,S) - 0.664 x(S,S) <= 0`, so `x(S,S) >= 0.9421 x(N,S)`.
4. Steps 2 and 3 give `x(N,S) = x(S,S) = 0`: operator 2 is never told S.
5. Game 1835, op 1 told S, dev L: `0.002 x(S,N) - 1.476 x(S,S) <= 0` with `x(S,S) = 0` gives `x(S,N) = 0` (the first coefficient is 2.17e-7 in the stored record: small, but positive).
6. Game 1835, op 2 told N, dev L: `1.320 x(N,N) - 6.821 x(S,N) <= 0` with `x(S,N) = 0` gives `x(N,N) = 0`. Nothing is left.

In words: one operator's "small burn" recommendation is squeezed from two sides. A game in which its maneuver cost is about three times nominal (3.35 and 2.81) allows that recommendation only if the other almost surely waits; a game with a large covariance scale, in which a lone small burn leaves a collision probability near 1e-4 and the operator would rather do the large burn, allows it only if the other burns often enough. No table meets both, so that operator is never told to burn. The other operator, told to burn, is then always alone, and a large-covariance game makes it want the large burn; "nobody burns" fails because someone then prefers to burn; and no large burn can be recommended, because for the operator concerned it costs more than it removes whatever the other does.

What is proved and what is observed:

- Proved (the steps above, from the stored coefficients): these six constraints admit no probability table, so the program is infeasible for these two samples.
- Computed by the code, not checked by hand: that each six-constraint set is irreducible. The deletion filter returns one irreducible set; others may exist.
- From a re-run of the sampling (`model.sample_true_games(BASELINE, N, default_rng(5), 0.3, 0.3, 0.3, 0.1)`, the call `scenario_program_check` makes): the covariance scales of games 251, 270, 1671, 1828 and 1835 rank 3rd, 20th, 3rd, 26th and 1st from the top in their samples; game 217 has the largest relative maneuver cost of operator 1 among the 608 games, and game 1828 the 10th largest of operator 2 among the 1899. The two large-burn constraints of step 1 have all coefficients positive in 606 and 605 of the 608 games (1893 and 1893 of the 1899).
- Observed, not proved: that this mechanism is what makes the plain scenario program infeasible in general at this uncertainty. Only one sample was drawn for each `N`.

A scenario program requires obedience in *every* sampled game, so the rarest sampled games decide. The theorem says nothing when the program is infeasible. With small uncertainty (log-standard deviation 0.05 on costs, 0.02 on the covariance scale; `scenario_program_small_uncertainty` in the results) the program is feasible and its violation rate on 10,000 fresh games is 0.84% for `N = 608` and 0.04% for `N = 1899`, both below `eps = 5%`, as the theory promises.

### 6.2 A finite list of candidate rules with exact binomial bounds

What the project uses instead (`experiments.calibrate_margin`). Take a finite list of `M` candidate rules: every rule of the two margin families of section 4.6, with the zero-margin rule, which both families contain, counted once (19 + 53 - 1 = 71), plus each pure Nash equilibrium of the nominal game as a public rule (2), so `M = 73` (`calibration.n_candidates` in the results). Sample `n` true games. For candidate `c` let `X_c` be the number of sampled games in which some recommendation would not be followed. If the true violation probability is `v_c`, then `X_c` is Binomial(`n`, `v_c`).

The Clopper-Pearson upper bound at confidence `1 - beta/M` is the largest `v` for which seeing `X_c` or fewer violations still has probability at least `beta/M`:

```
U_c = BetaInverse( 1 - beta/M ;  X_c + 1,  n - X_c )                                    (6.2)
```

By construction `P(v_c > U_c) <= beta/M` for each candidate. By the union bound, with probability at least `1 - beta` *all* `M` bounds hold at once. Therefore any candidate chosen with `U_c <= eps`, however it is chosen, has true violation probability at most `eps` with confidence `1 - beta`. The project uses `n = 10,000` and `beta = 1e-3`.

Compared with the scenario approach this gives up optimising over all tables `x` and only searches the list. In exchange it tolerates the rare sampled games that make the scenario program infeasible, and the argument is two lines.

### 6.3 A safety chance constraint is linear

Let the covariance scale `kc` be uncertain. For joint action `a` let `unsafe(a) = P( Pc(a; kc) > 1e-4 )`, estimated from 200,000 draws of `kc`. The probability, over both the coordinator's draw and the covariance scale, that the recommended joint action leaves the collision probability above the threshold is

```
sum over a of  x(a) * unsafe(a)   <=  delta                                             (6.3)
```

One linear constraint. `experiments.safety_constraint_check` adds it to both programs.

---

## 7. Many conjunctions

This section covers the batch model of `conjce/batch.py`. Many conjunctions among 30 satellites of 4 operators are handled at once. Every proposition is proved step by step and followed by the tests that check it. The numbers that come from a run are taken from `results/scaleup.json` (written by `run_scaleup.py`), and the key is named. No timing is quoted: the timing table in that file holds wall-clock seconds on one laptop.

### 7.1 The batch game

**Sets and notation.**

| Symbol | Meaning |
|---|---|
| `i` | satellite, 0 to 29; `o(i)` is the operator that owns it (`batch.OPERATOR_OF`) |
| `p`, `q` | operators, 0 to 3 |
| `k = 1, ..., K` | conjunctions; conjunction `k` is a pair of satellites of two *different* operators, one on side 1 and one on side 2 |
| `s = (k, j)` | slot: side `j` (1 or 2) of conjunction `k`; it belongs to the operator of that satellite |
| `S_p`, `n_p` | the slots of operator `p` and their number; `n_0 + n_1 + n_2 + n_3 = 2K` |
| `e_k = (e_k1, e_k2)` | cell: the pair of actions in conjunction `k`, side 1 first, each N, S or L |
| `a = (e_1, ..., e_K)` | joint profile, one cell per conjunction |
| `a_s`, `a_-s` | for slot `s = (k, j)`: the action of side `j` in `e_k`, and the action of the other side of `k` |
| `a_p` | the tuple `(a_s)` for `s` in `S_p`: what operator `p` is told |

Because the two satellites of a conjunction belong to different operators, **an operator has at most one slot in each conjunction**, and the action on the other side of its slot is always chosen by another operator. Every proof below uses this fact. An operator can still have several slots against the same other operator, in different conjunctions.

**Costs.** Conjunction `k` has the two cost tables of section 3.2, built from its own scenario by `model.cost_tables`, and the coordinator table `J_k = model.coordinator_cost_table` of section 4.4. Side 1 plays the part of operator 1 of that game and side 2 the part of operator 2. Write each side's table with its own action first:

```
g_s(own, other) = c_k1(own, other)   if s = (k, 1)
g_s(own, other) = c_k2(other, own)   if s = (k, 2)                    (batch._side_cost)
```

The cost of operator `p` and the coordinator's cost of a profile are

```
C_p(a) = sum over s in S_p of  g_s(a_s, a_-s)                                           (7.1)
J(a)   = sum over k of  J_k(e_k)                                                       (7.2)
```

Each `g_s` is a fraction of the value of that slot's satellite (section 3.2). Section 7.2 shows that the obedience conditions do not change if each slot's term is multiplied by a different positive number, for example that satellite's value in USD.

**Recommendation rule and obedience.** The coordinator draws a profile `a` from a distribution `x` over all profiles. It tells operator `p` the tuple `a_p`, that is, the recommendations for all of its slots at once, and nothing else. Suppose `p` is told the tuple `r`, which has probability `pi_p(r) = sum over a with a_p = r of x(a)`. By Bayes' rule, `p` believes that the profile is `a` with probability `x(a) / pi_p(r)`. Suppose `p` changes the action of one slot `s` from `r_s` to `d`. Every other term of (7.1) stays the same: those terms involve `p`'s own actions in its other slots, which are unchanged, and actions of other operators. So these terms cancel in the comparison, and

```
expected cost of obeying - expected cost of the deviation
   = sum over a with a_p = r of  [x(a) / pi_p(r)] [ g_s(r_s, a_-s) - g_s(d, a_-s) ]
```

The operator obeys if this is at most zero. Multiply by `pi_p(r)`, as in (4.2):

```
sum over a with a_p = r of  x(a) [ g_s(r_s, a_-s) - g_s(d, a_-s) ]  <=  0
        for every operator p, every tuple r, every slot s in S_p and every d != r_s       (7.3)
```

The condition is on the whole tuple `r`, not only on the recommendation in slot `s`. As in (4.2), it holds trivially when `pi_p(r) = 0`, because every term is zero. `batch.joint_rows` builds these rows. `batch.joint_program` divides every row by its largest absolute coefficient, as `games.solve_private` does; dividing a row by a positive number does not change the set of `x` it allows.

**The programs.**

- (a) Full joint program (`joint_program(conjs, "full")`): minimise `sum over a of x(a) J(a)` subject to (7.3), `x >= 0` and `sum of x = 1`, over all `9^K` profiles.
- (b) Exact-one joint program (`joint_program(conjs, "exact_one")`): the same, with `x` restricted to the `4^K` profiles in which every cell is one of SN, LN, NS, NL. The rows are the same and the deviation `d` ranges over all three actions. The restriction limits what may be recommended, not what an operator may do.
- (c) Per-conjunction program (`per_conjunction_program`): one 3x3 table per conjunction with its own 12 rows (4.2). This gives `9K` variables.

**Sizes.** Variables: `9^K` for (a) and `4^K` for (b), so `9^5 = 59,049` and `4^8 = 65,536` are the largest within the size cap of 100,000. Rows of (7.3): operator `p` has `3^(n_p)` tuples, `n_p` slots and 2 deviations per slot, so

```
number of rows of (7.3) = sum over p of  2 n_p 3^(n_p)                                   (7.4)
```

plus one row for `sum of x = 1`. In the full program every tuple occurs in some profile, so all of these rows are built. Example: in the first two conjunctions of the master batch, one operator has two slots and two operators have one each. That gives `2 x 2 x 9 + 2 x 1 x 3 + 2 x 1 x 3 = 48` rows, plus one, which is the 49 stored for `K = 2` under `timing` and `propositions.P1.checks`. Each profile appears in 2 rows per slot and has `2K` slots, so a column of (a) or (b) has `4K` nonzeros plus 1 for the row `sum of x = 1`. `tests/test_batch.py::test_joint_program_sizes` checks the variable counts `9^3` and `4^3` and the nonzero counts `9^3 (4 x 3 + 1)` and `4^3 (4 x 3 + 1)`. The number of rows grows like `3^(n_p)`, so it is exponential in the number of slots of the busiest operator.

### 7.2 Single-slot deviations are enough (P1)

**Proposition 1.** Suppose `x` satisfies every row of (7.3). Then no operator, told any tuple `r`, can lower its expected cost by changing the actions of several slots at once. So adding the rows for all multi-slot deviations changes neither the feasible set nor the optimum.

*Proof.* Let operator `p`, told `r`, switch to another tuple `d != r`. Let `D` be the set of slots where `d_s != r_s`. By (7.1), the change of its cost in profile `a` is a sum over its slots. Slots outside `D` contribute zero, and the term of slot `s` depends only on `p`'s action there and on `a_-s`, which belongs to another operator and so does not change when `p` deviates:

```
sum over a with a_p = r of  x(a) [ C_p(a) - C_p(a with a_p replaced by d) ]
  = sum over a with a_p = r of  x(a) * sum over s in D of [ g_s(r_s, a_-s) - g_s(d_s, a_-s) ]
  = sum over s in D of  ( sum over a with a_p = r of  x(a) [ g_s(r_s, a_-s) - g_s(d_s, a_-s) ] )
```

Each bracket in the last line is the left side of a row of (7.3) for the same `p` and the same tuple `r`, with deviation `d_s` in slot `s`. Each is `<= 0`, so their sum is `<= 0`. Conversely, every single-slot row is a multi-slot row with `|D| = 1`. So the two sets of rows allow the same `x`. A deviation that depends on what the operator is told (a map from tuples to tuples) gives one such condition for each tuple, so it is covered as well. ∎

What the proof uses: (1) the cost is a sum over slots; (2) each term depends only on the operator's own action in that slot and on the other side's action; (3) the other side belongs to another operator; (4) all rows for the same `r` condition on the same event. If each term is multiplied by a positive weight `w_s`, every row is multiplied by `w_s` and the conclusion is unchanged. So the obedient rules are the same whether an operator adds up fractions of each satellite's value, as here, or USD.

The number of multi-slot rows is `sum over p of 3^(n_p) (3^(n_p) - 1)`. For the two-conjunction example above that is `9 x 8 + 3 x 2 + 3 x 2 = 84`, plus one, which is the 85 stored as `rows_all_deviations`.

Tests: `tests/test_batch.py::test_p1_single_slot_deviations_are_enough` (K = 2 and 3; the test first checks that some operator has at least two slots, so the check is not vacuous). It requires that the optimum with all multi-slot rows equals the single-slot optimum within 1e-6 relative, and that the single-slot optimum violates no multi-slot row by more than 1e-9. `test_p1_and_p2_with_several_slots_against_the_same_operator` runs the same check on three master conjunctions between the same two operators. `run_scaleup.py` repeats it at every K of the grid within the size cap (K = 1 to 5; `propositions.P1`).

### 7.3 Decomposition (P2)

Let `t_k(e) = sum over a with e_k = e of x(a)` be the marginal of conjunction `k`: a 3x3 table.

**Proposition 2.** (i) If `x` satisfies (7.3), each marginal `t_k` satisfies the 12 rows (4.2) of conjunction `k`'s own game. (ii) If each `t_k` satisfies its 12 rows, the product `x(a) = t_1(e_1) t_2(e_2) ... t_K(e_K)` satisfies (7.3).

*Proof of (i).* Fix an operator `p`, its slot `s = (k, j)`, a recommendation `b` and a deviation `d != b`. Add the rows (7.3) of all tuples `r` with `r_s = b`. Tuples with `pi_p(r) = 0` add zero, so including them is harmless. Every profile with `a_s = b` has exactly one such tuple, namely its own `a_p`, so the double sum becomes one sum:

```
sum over r with r_s = b of  sum over a with a_p = r of  x(a) [ g_s(b, a_-s) - g_s(d, a_-s) ]
  = sum over a with a_s = b of  x(a) [ g_s(b, a_-s) - g_s(d, a_-s) ]      <=  0
```

The bracket depends on `a` only through the cell `e_k`. Grouping the profiles by `e_k` replaces `x` by the marginal:

```
sum over cells e of conjunction k with e_j = b of  t_k(e) [ g_s(b, e_other) - g_s(d, e_other) ]   <=  0
```

where `e_other` is the other side's action in `e`. This is row (4.2) of conjunction `k` for side `j`, told `b`, deviation `d`. ∎

*Proof of (ii).* Fix `p`, a tuple `r`, a slot `s = (k, j)` of `p` and `d != r_s`. Split the conjunctions into three groups: `k` itself; the other conjunctions in which `p` has a slot (one slot `s'` each, because `p` has at most one slot per conjunction); and the conjunctions without `p`. The event `a_p = r` fixes one side of each cell in the first two groups and nothing in the third. The bracket of (7.3) depends only on `e_k`. Since `x` is a product, the sum factorises:

```
sum over a with a_p = r of  x(a) [ g_s(r_s, a_-s) - g_s(d, a_-s) ]
  =   [ sum over e with e_j = r_s of  t_k(e) ( g_s(r_s, e_other) - g_s(d, e_other) ) ]       (row (4.2) of k: <= 0)
    x  product over the other conjunctions k' of p of  [ sum over e whose side of p equals r_s' of  t_k'(e) ]   (>= 0)
    x  product over the conjunctions without p of  [ sum over e of t_k''(e) ]                (= 1)
```

The first factor is `<= 0` and the others are `>= 0`, so the product is `<= 0`. ∎

In words: under a product rule, what an operator's whole tuple tells it about the other side of conjunction `k` is what its recommendation in `k` alone tells it, because the other conjunctions are drawn independently. This includes conjunctions against the same other operator.

**The optimum splits.** Let `T_k` be the set of 3x3 probability tables that satisfy conjunction `k`'s 12 rows. By (i), the marginals of every jointly obedient `x` lie in `T_1 x ... x T_K`. By (ii), every element `(t_1, ..., t_K)` of that set is the list of marginals of a jointly obedient rule, namely the product, whose marginals are the `t_k`. So for any objective that depends on `x` only through its marginals, the joint program has the same optimum as the program over `(t_1, ..., t_K)` in `T_1 x ... x T_K`. The coordinator's expected cost is such an objective, and a sum over conjunctions:

```
sum over a of x(a) J(a) = sum over k of  sum over e of t_k(e) J_k(e)                        (7.5)
```

Each term involves one `t_k` only, so the minimum over `T_1 x ... x T_K` is the sum of the minima over each `T_k`:

```
optimum of (a)  =  sum over k of V_k,      V_k = min over t in T_k of  sum over e of t(e) J_k(e)        (7.6)
```

`V_k` is the cost `games.solve_private(c_k1, c_k2, J_k)` returns (zero margin). Program (c) is the `K` programs side by side, so its optimum is the same sum. Each minimum exists. `T_k` is closed and bounded. It is not empty: the game has a mixed Nash equilibrium `(p*, q*)` (Nash 1951), and its product table `p*(a1) q*(a2)` satisfies (4.2). For side 1 told `r`, the row is `p*(r) [ sum over a2 of q*(a2) c_k1(r, a2) - sum over a2 of q*(a2) c_k1(d, a2) ]`; this is zero when `p*(r) = 0` and `<= 0` otherwise, because an action played with positive probability is a best reply to `q*`. In particular, program (a) is always feasible.

Tests: `tests/test_batch.py::test_p2_decomposition` (K = 1 to 4) requires four things: optimum of (a) = optimum of (c) = sum of `games.solve_private` within 1e-6 relative; the marginals of the optimum of (a) violate no row of their conjunction by more than 1e-9 (scaled rows); the product of the (c) tables violates no row of (a) by more than 1e-9; and the product has the (c) objective. `test_per_conjunction_program_matches_solve_private_at_k50` checks (c) against the sum of `games.solve_private` at K = 50. `test_p1_and_p2_with_several_slots_against_the_same_operator` covers the case of several slots against the same operator. `run_scaleup.py` checks (a) against (c) up to K = 5 and (c) against the sum of `solve_private` at every K up to 50 (`propositions.P2`).

**Scope.** The argument holds for any objective that depends only on the per-conjunction marginals. Example: the largest expected number of burns of a satellite in section 7.5. That objective is not a sum over conjunctions like (7.5), so it does not split into `K` separate programs; it only needs the marginals. It does not cover an objective that depends on joint events, such as the probability that the same satellite burns twice. It also stops covering the model once a cost couples conjunctions. Example: a burn by satellite `i` in conjunction `k` changes the miss vector of `i`'s later conjunction `k'`, so the cost of `k'` depends on the cell of `k` (section 7.6). Then (7.1) is no longer a sum of terms that each depend on one cell, and the steps "the bracket depends on `a` only through `e_k`" in (i) and "the sum factorises" in (ii) fail. Nothing in this section then says that the joint optimum is a sum of per-conjunction optima.

This split, not the exact-one restriction, is what makes K = 50 fast in this model. Program (c) has `9K` variables. The restriction on its own still leaves `4^K` joint profiles. Within the restriction, the split together with section 7.4 reduces the program to a list of stable designations per conjunction.

### 7.4 The exact-one restriction (P3)

**Words.** Exact-one cells: SN, LN, NS, NL (exactly one satellite burns). A *stable designation* of conjunction `k` is an exact-one cell that is a pure Nash equilibrium of `k`'s game (section 4.1). `E_k` is the set of them (`batch.stable_designations`, which uses `games.pure_nash` with its tolerance of 1e-12).

**No-tie assumption.** For every slot `s`, the small and the large burn are not both best replies to N on the other side. That is, it is not the case that `g_s(S, N) = g_s(L, N) <= g_s(N, N)`.

**Proposition 3.** (a) Every distribution `x` over `E_1 x ... x E_K` satisfies (7.3). It still does if the whole profile is announced to every operator. This part needs no assumption. (b) Under the no-tie assumption, if `x` puts mass only on exact-one profiles and satisfies (7.3), then every profile with `x(a) > 0` lies in `E_1 x ... x E_K`.

*Proof of (a).* Take any row of (7.3) and any term with `x(a) > 0`. The cell `e_k` of the slot's conjunction is a pure equilibrium of `k`'s game, so the action `r_s = a_s` is a best reply to `a_-s`, and `g_s(r_s, a_-s) - g_s(d, a_-s) <= 0`. Each term is `<= 0`, so their sum is too. The argument works term by term, so it holds for any conditioning, including an announcement of the whole profile. Every profile in `E_1 x ... x E_K` is therefore a pure equilibrium of the batch game; by section 7.2, multi-slot deviations add nothing. ∎

*Proof of (b).* Take a profile `a` with `x(a) > 0` and a conjunction `k`. By symmetry of the two sides, let `e_k = (b, N)` with `b` = S or L: side 1 burns. Let `s1 = (k, 1)` belong to operator `p` and `s2 = (k, 2)` to operator `q`, with `q != p`.

1. *The side told to burn.* Let `r = a_p`; then `pi_p(r) >= x(a) > 0`. Every profile `a'` with `x(a') > 0` and `a'_p = r` has side 1 of `k` told `b`, which is not N. Since `a'` is exact-one, side 2 of `k` is told N in `a'`. So in each row `(p, r, s1, d)` every term with positive weight has `a_-s1 = N`, and the row reads `pi_p(r) [ g_s1(b, N) - g_s1(d, N) ] <= 0`. Hence `g_s1(b, N) <= g_s1(d, N)` for every `d`: `b` is a best reply of side 1 to N.
2. *One burn size.* Step 1 holds in every profile of the support in which side 1 of `k` burns. If side 1 were told S in one such profile and L in another, both S and L would be best replies to N, which the no-tie assumption excludes. So there is one size `b*` (here `b* = b`) such that in every profile of the support where side 2 of `k` is told N, side 1 is told `b*`. Side 1 does burn in all of these profiles: "both N" is not an exact-one cell.
3. *The side told to wait.* Let `r' = a_q`; then `pi_q(r') >= x(a) > 0`. Every profile of the support with `a'_q = r'` has side 2 of `k` told N, so by step 2 side 1 is told `b`. The rows `(q, r', s2, d)` read `pi_q(r') [ g_s2(N, b) - g_s2(d, b) ] <= 0`, so N is a best reply of side 2 to `b`.

By steps 1 and 3, `(b, N)` is a pure equilibrium of conjunction `k`, that is, `e_k` is in `E_k`. Since `a` and `k` were arbitrary, the claim follows. ∎

Points checked while writing the proof:
- Conditioning on the whole tuple does no harm. Steps 1 and 3 hold for whatever the operator is told in its other slots, including slots against the same other operator, because the other side's action in `k` is already fixed by the exact-one structure and by step 2.
- Recommendations of probability zero impose nothing. Their rows are `0 <= 0`, and the proof uses only tuples of profiles in the support.
- A conjunction with only one stable designation is covered by the same steps.
- **The no-tie assumption is used in step 2 only.** With a tie, side 2 told N may face a mixture of S and L on side 1. N then needs to be a best reply only on average, and the support can contain a cell that is not an equilibrium.

Example with a tie, for one conjunction (rows: side 1's action N, S, L; columns: side 2's action N, S, L):

```
side 1:  c_1 = [[1, 1, 1],         side 2:  c_2 = [[1, 0, 0],        coordinator: J(S,N) = 2, J(L,N) = 1,
                [0, 0, 0],                         [0, 1, 1],                     J(N,S) = J(N,L) = 3
                [0, 0, 0]]                         [1, 0, 2]]
```

Side 1 has the tie `c_1(S,N) = c_1(L,N) = 0 <= c_1(N,N) = 1`. Side 1 prefers to burn whatever side 2 does, so NS and NL are not equilibria. SN is an equilibrium: side 2 against S pays 0 for N and 1 for S or L. LN is not one: side 2 against L pays 1 for N and 0 for S. So `E = {SN}`, and the closed form (7.7) below gives 2. Now take `x(SN) = x(LN) = 1/2`. Side 1, told S or L, knows side 2 waits and loses nothing by obeying. Side 2, told N, faces S or L with probability 1/2 each. Against deviation S its advantage of obeying is `-[1/2 (0 - 1) + 1/2 (1 - 0)] = 0`; against L it is `-[1/2 (0 - 1) + 1/2 (1 - 2)] = 1`. Both are `>= 0`, so `x` is obedient. It costs `1/2 x 2 + 1/2 x 1 = 1.5 < 2`, and half of its mass is on LN, which is not a stable designation. Changing side 2's row "side 1 plays S" to `[1, 2, 0]` (side 2 now prefers L against S) makes SN unstable as well, so `E` is empty. The same `x` is still obedient: both advantages of side 2 become 0. Side 2 has a tie too: against N its costs are `c_2(N, .) = [1, 0, 0]`, so S and L are both best replies to N for side 2. The example does not need that tie, only side 1's. Side 2 is never told to burn in `x`, and NS and NL fail because of side 1, so the row `c_2(N, .)` enters neither the obedience rows of `x` nor `E`. Removing side 2's tie, for example by setting `c_2(N, L) = 0.5`, changes neither the computation above nor its result. Tests: `tests/test_batch.py::test_tie_example_exact_one_program_below_closed_form` checks, on these tables, that `E = {SN}`, that the exact-one program (`batch.joint_program([conj], "exact_one")`) has optimum 1.5 and that `batch.closed_form` gives 2. `test_tie_variant_with_empty_E_k_is_still_feasible` checks, on the variant, that `E` is empty, that the closed form is infinite and that the exact-one program is feasible.

In the random family a tie means equality of two continuous functions of the drawn parameters. The run contains none. `batch.n_slots_with_burn_size_tie` counts the slots with a tie, as defined in the docstring of `batch.burn_size_tie_slots`: S and L are both best replies to N, within the absolute tolerance 1e-12 of `games.pure_nash`. The count is 0 over the 20,100 slots of the 10,050 kept conjunctions of the 201 batches (`family.n_slots_with_burn_size_tie`; `family.n_kept` = 10,050) and 0 over the 100 slots of the master batch (`family_master_only.n_slots_with_burn_size_tie`). `test_master_batch_has_no_burn_size_tie` checks the master batch.

**Consequences** (all under the no-tie assumption, except where stated).

1. *Value of the restricted program.* By (a) and (b), the feasible set of (b) is exactly the set of distributions over `E_1 x ... x E_K`. On it, by (7.2), the expected cost is `sum over k of E[J_k(e_k)]`, and each term is at least the cheapest stable designation. A point mass on the product of the cheapest designations reaches this bound, so

   ```
   optimum of (b)  =  sum over k of  min over e in E_k of J_k(e)                          (7.7)
   ```

   This is `batch.closed_form` (formulation (c')): a list per conjunction, with no program. Without the assumption, (a) still makes this point mass feasible, so the optimum of (b) is at most (7.7). The tie example above shows it can be lower.
2. *No obedient rule when some `E_k` is empty.* `E_1 x ... x E_K` is then empty, so by (b) program (b) is infeasible. The tie example shows this needs the assumption. Program (a) stays feasible (section 7.3). This is why `batch.generate_batch` redraws a conjunction's game until `E_k` is not empty; the share of draws kept is `family.share_kept`.
3. *A private recommendation carries nothing that a public announcement would not.* By steps 1 and 2, in an obedient exact-one rule an operator's recommendation in slot `s` fixes the whole cell of `k`. "Burn `b`" means the other side waits; "N" means the other side burns `b*`. So privately each operator learns exactly the designation of each of its conjunctions, and its costs depend on nothing else. More precisely, the set of exact-one rules that are obedient when the whole profile is announced equals the set that is obedient under private recommendations. In one direction, a rule obedient under announcement satisfies the row inequality for every single profile, and summing over the profiles with `a_p = r` gives (7.3). In the other direction, (b) and then (a) show that a privately obedient rule is a distribution over `E_1 x ... x E_K`, which is obedient under announcement. So with the restriction, the coordinator gains nothing by keeping recommendations private. The difference between (7.6) and (7.7) comes entirely from rules outside the restriction, such as (S,S) in section 4.4. That difference is computed with the rule known and zero margin; RESULTS.md 4.2 shows how much of such a gain survives when operators must learn the rule.
4. *An announced stable designation needs no learning.* This is an argument, not a batch experiment: nothing about learning was simulated on the batch. When the designation is announced, each operator knows the other side's action in each of its conjunctions. Obeying is a best reply, because the cell is a pure equilibrium of that conjunction's game (part (a)), whatever the operator believes about how the coordinator chooses. It is the argument of section 4.5 for public rules. In the single-conjunction learning model it shows up as identical learned and known-rule costs for the public rule (`public_learned_max_abs_deviation`, section 5.4).

Tests: `tests/test_batch.py::test_p3_exact_one_optimum_is_the_closed_form` (K = 1 to 6) requires that the optimum of (b) equals (7.7) within 1e-6 relative and that the optimum of (b) has at most 1e-9 mass outside `E_1 x ... x E_K`. `test_p3_every_mixture_of_stable_products_is_obedient` checks part (a) on a random distribution over the stable products at K = 5, and checks that a point mass on a profile outside them violates a row. `test_p3_exact_one_program_is_infeasible_with_an_empty_E_k` checks consequence 2 on three conjunctions, one replaced by the first rejected draw (`propositions.P3.infeasible_batch`). `test_kept_draws_have_a_stable_designation_and_rejected_ones_do_not` and `test_stable_designations_are_exact_one_pure_equilibria` check the generator and the definition. `test_hand_built_game_without_a_stable_designation` checks the fallback game of the brief. `run_scaleup.py` checks (b) against (7.7) at every K up to 8 (`propositions.P3`). The two tie examples and the absence of ties in the run are tested as described above (`test_tie_example_exact_one_program_below_closed_form`, `test_tie_variant_with_empty_E_k_is_still_feasible`, `test_master_batch_has_no_burn_size_tie`; key `n_slots_with_burn_size_tie`).

**Relation to arXiv:2609.13565 (second-hand).** The following is read through notes made for this project from the arXiv v1 text; the paper was not opened for this document, and the theorem and equation numbers are those the notes give, to be checked against the version being quoted. Im, Topcu and Fridovich-Keil consider agents that each occupy or yield shared resources, with costs of a fixed form (their eq. 14). They restrict the coordinator to profiles in which every resource has exactly one occupant. Their Theorem 1 (eqs. 22 and 23) states that, on this set, the chance-constrained correlated equilibria are exactly the mixtures of chance-constrained pure Nash equilibria in the set, and that the set of such rules is non-empty if and only if such an equilibrium exists. Their proof uses Lemma 1: on the restricted set, what an agent gains by a deviation does not depend on the other agents' actions, so each obedience row is a probability times a constant (eqs. 24 to 26). Proposition 3 is the counterpart for the model here, with zero margin. Here what a deviation gains does depend on the other side's action, through `Pc`. Steps 1 and 2 play the part of their Lemma 1: under exact-one and no ties, an operator's own recommendation fixes the other side's action, so each row is again `pi` times a constant. The no-tie assumption has no counterpart in the notes, because there an agent's choice on a resource has two values, while here a burn has two sizes. The notes also say that their restricted set has `n^r` profiles, which is exponential in the number of resources `r`; the restricted set here has `4^K` profiles, which is exponential in `K`.

### 7.5 Burden-sharing

**Stage 1.** Every conjunction gets a cheapest stable designation, `e_k` in `argmin over e in E_k of J_k(e)`. The total cost is (7.7). A conjunction is *free* if `E_k` holds exactly one designation for each side and the two have the same coordinator cost (relative difference at most 1e-9, `batch.FREE_TOL`; `batch.designation_info`). A conjunction that is not free has a single cheapest designation in the run: no kept conjunction has two stable designations on the same side or more than two (`family.n_two_on_the_same_side` and `family.n_more_than_two_stable_designations` are 0). `test_free_conjunctions_have_one_designation_per_side` checks this for the master batch. Stage 1 therefore fixes the burning satellite of every conjunction that is not free. Let `n_i^0` be the number of such conjunctions in which satellite `i` burns.

**Stage 2** decides the free conjunctions `f` in `F` only. Each free conjunction's two designations cost the coordinator the same, so the coordinator's cost is the same in all variants, up to the 1e-9 tolerance per free conjunction. By Proposition 3(a), every variant is obedient, the randomised one included, because it is a distribution over products of stable designations. Write `z_f = 1` if side 1 of `f` burns and `z_f = 0` if side 2 burns. The number of burns of satellite `i` is

```
n_i(z) = n_i^0 + sum over f in F with i on side 1 of  z_f  +  sum over f in F with i on side 2 of  (1 - z_f)        (7.8)
```

which the code writes as `base + s2 + B z` (`batch._incidence`). Every conjunction has exactly one burning satellite, so `sum over i of n_i(z) = K` for every `z`.

- Fixed tie-break: `z_f = 1` exactly when `model.fixed_rule_profile` designates side 1 for `f`'s scenario. `T_fix = max over i of n_i(z_fix)`.
- Balanced, randomised (`batch.balanced_randomised`):

  ```
  T_rand = minimise t   over q in [0, 1]^F and t
           subject to  n_i^0 + sum over f with i on side 1 of q_f + sum over f with i on side 2 of (1 - q_f)  <=  t   for every satellite i        (7.9)
  ```

  `q_f` is the probability that side 1 of `f` burns. The program has `|F| + 1` variables.
- Balanced, deterministic (`batch.balanced_deterministic`, `scipy.optimize.milp`): the same program with `q` restricted to `{0, 1}^F`. Its value `T_det` is the smallest possible largest number of burns. The value is unique; the assignment that reaches it need not be, and the code returns one of them.

**Proposition 5.** Minimise the largest expected number of burns of a satellite over all distributions `y` on the `2^|F|` joint assignments `z` of the free conjunctions, with the other conjunctions fixed by stage 1. The minimum equals `T_rand` of (7.9).

*Proof.* Let `q_f = P_y(z_f = 1)`. (7.8) is affine in `z` and expectation is linear, so

```
E_y[ n_i(z) ] = n_i^0 + sum over f with i on side 1 of  q_f  +  sum over f with i on side 2 of  (1 - q_f)
```

which depends on `y` only through `q`. So every `y` gives a point of (7.9) with the same value of `max over i of E_y[n_i]`. Conversely, every `q` in `[0, 1]^F` comes from some `y`: decide each free conjunction independently, with side 1 burning with probability `q_f`. The two minima are therefore equal. ∎

The quantity minimised is the largest *expected* number of burns, not the expected largest number. One draw from a rule that achieves `T_rand` can give some satellite more than `T_rand` burns.

Tests: `tests/test_batch.py::test_p5_burden_needs_only_marginals` (K = 6, 8, 10 of the master batch) requires that (7.9), with `|F| + 1` variables, and the program over all `2^|F|` assignments (`batch.balanced_randomised_joint`) agree within 1e-6 relative (`propositions.P5`).

**Why `T_rand <= T_det <= T_fix`.** `z_fix` is one point of `{0, 1}^F`, so the 0/1 minimum is at most its value: `T_det <= T_fix`. Every point of `{0, 1}^F` lies in `[0, 1]^F`, and for a 0/1 vector the expected and the realised numbers of burns coincide. So (7.9) minimises over a larger set, and `T_rand <= T_det`. Two further bounds follow. First, `T_det` is a largest number of burns, hence an integer, so `T_det >= ceil(T_rand)`. Second, the burns of the 30 satellites add up to `K` for every assignment, and in expectation for every distribution, so all three values are at least `K / 30`.

Tests: `test_burden_variants_keep_one_burn_per_conjunction` (K = 10 and 50) checks the ordering and that each variant gives `K` burns in all. `test_deterministic_balance_matches_brute_force` (K = 6, 8, 10) checks `T_det` against a search over all `2^|F|` assignments. `run_scaleup.py` makes the same brute-force comparison wherever at most 12 conjunctions are free (`burden.brute_force_agreement`).

**Proposition 6.** `T_det = ceil(T_rand)`.

*Proof.* By the bound above, `T_det >= ceil(T_rand)`. It remains to find a 0/1 assignment with largest number of burns at most `t = ceil(T_rand)`. `t` is an integer. If `T_rand` is already an integer, then `t = T_rand`, and nothing below changes.

1. *A system with two variables per free conjunction.* For each free conjunction `f` let `y_f1` be the weight on "side 1 burns" and `y_f2` the weight on "side 2 burns". Consider the polytope `P(t)` of all `y` with

   ```
   y_f1 + y_f2 = 1                                                   for every free conjunction f
   sum over (f, j) with satellite i on side j of f of  y_fj  <=  t - n_i^0      for every satellite i      (7.10)
   0 <= y_fj <= 1                                                    for every f and j
   ```

   A 0/1 point of `P(t)` is an assignment: `z_f = y_f1`, and then `y_f2 = 1 - z_f`. For it, the left side of the satellite row plus `n_i^0` is `n_i(z)` of (7.8), so the row says `n_i(z) <= t`.
2. *`P(t)` is not empty.* Let `q` be an optimal point of (7.9) and put `y_f1 = q_f`, `y_f2 = 1 - q_f`. The equalities and bounds hold because `0 <= q_f <= 1`. Each satellite row is the constraint of (7.9) with `t` in place of `T_rand`. That constraint holds with `T_rand`, and `T_rand <= t`, so it holds with `t` as well.
3. *The matrix is totally unimodular.* In the matrix of the equality rows and satellite rows of (7.10), the column of `y_fj` has exactly two entries, both 1: one in the row of conjunction `f` and one in the row of the satellite on side `j` of `f`. So it is the incidence matrix of a bipartite graph. One part has the free conjunctions and the other the satellites, and each `y_fj` is an edge from `f` to its side-`j` satellite. The two satellites of a conjunction are different, so no edge is a loop. Satellites in no free conjunction give zero rows. The incidence matrix of a bipartite graph is totally unimodular (Schrijver 1986). Writing each equality as two inequalities (the rows `A` and `-A`) and adding the bounds (rows of `I` and `-I`) keeps the matrix totally unimodular, because multiplying rows by -1 and appending unit rows both preserve the property (Schrijver 1986).
4. *The right-hand sides are integers.* They are 1 and -1 for the equalities, 1 and 0 for the bounds, and `t - n_i^0` for the satellite rows. `t` is an integer and `n_i^0` counts conjunctions. A satellite whose fixed burns alone exceed `t` would have a negative right-hand side. That cannot happen here: in step 2 every term added to `n_i^0` is at least 0, so `n_i^0 <= T_rand <= t`. The argument would not need this anyway, since negative integers are still integers.
5. *An integral vertex.* `P(t)` is bounded (all variables lie between 0 and 1) and not empty (step 2), so it has a vertex. A polyhedron `{y : M y <= h}` with `M` totally unimodular and `h` integral has only integral vertices (Hoffman and Kruskal 1956; Schrijver 1986). So `P(t)` contains a point `y` whose entries are integers between 0 and 1, and `z_f = y_f1` is an assignment with `n_i(z) <= t` for every satellite.

Hence `T_det <= t = ceil(T_rand)`, and with the bound above, `T_det = ceil(T_rand)`. ∎

The proof needs no extra condition. It uses only three facts: stage 1 fixes the conjunctions that are not free with integer burn counts; each free conjunction has two choices, each charged to one satellite; and the two satellites of a conjunction are different.

Consequence: the value of the 0/1 program follows from the linear program (7.9) by rounding up. The 0/1 program (`scipy.optimize.milp`) is needed only to produce an assignment that reaches this value. Computed in floating point, a `T_rand` that is an integer may come out slightly above it, so the rounding needs a small tolerance. The test below rounds `T_rand - 1e-9` up.

Test: `tests/test_batch.py::test_deterministic_balance_is_the_randomised_value_rounded_up`. It covers 202 batches: the first 20 conjunctions of the master batch, the whole master batch (K = 50), and the 200 further batches of 50 conjunctions (seeds 1000 to 1199). For each it requires that `batch.balanced_deterministic(...)["max_burns"]` equals `ceil(T_rand - 1e-9)`, with `T_rand = batch.balanced_randomised(...)["max_expected_burns"]`.

### 7.6 What is not modelled

- **Carry-over of a burn.** By section 2.3, a burn moves the satellite along its orbit by `3 dv t`, a shift that grows with time. So a burn in one conjunction changes the miss vector, and hence `Pc` and both cost tables, of the same satellite's later conjunctions; it can also remove or create conjunctions. The batch draws each conjunction's game independently and treats the batch as one simultaneous game; the times of closest approach are used only to order conjunctions for the counts below. With carry-over, (7.1) and (7.2) are no longer sums of terms that each depend on one cell. The proofs of sections 7.2 to 7.5 all use that structure, so none of them covers it. `batch.carry_over` only counts how often it would matter, for the master batch at K = 50 in time order: the conjunctions that involve a satellite already designated to burn earlier, under the fixed tie-break and under the balanced deterministic assignment (`carry_over.after_earlier_burn_fixed_tie_break`, `carry_over.after_earlier_burn_balanced_deterministic`), and the conjunctions that share a satellite with another one (`carry_over.sharing_a_satellite`). `tests/test_batch.py::test_carry_over_on_a_hand_made_batch` checks the counting on three hand-placed conjunctions. Nothing corrects for it.
- **Margins.** The batch uses zero margin everywhere; margins belong to the learning question of sections 4.6 and 5. The following is argued here and not tested, since `conjce/batch.py` has no margin. With the margin `mu >= 0` of (4.4) added to every row of (7.3), the proofs carry over. In section 7.2, the multi-slot sum is at most `-|D| mu pi_p(r) <= -mu pi_p(r)`. In section 7.3, the margins `mu pi_p(r)` of the tuples with `r_s = b` add up to `mu` times the probability of `b` in `k`, and under a product rule `pi_p(r)` is the product of the factors in (ii). In section 7.4 each cell of the support becomes a pure equilibrium with slack `mu` (`games.pure_nash(..., margin=mu)`). For `mu > 0`, step 2 needs no assumption: told S, side 1 needs `g(S,N) + mu <= g(L,N)`, and told L it needs `g(L,N) + mu <= g(S,N)`; together these give `2 mu <= 0`, which is impossible.
- **One satellite, one set of parameters.** Each conjunction's game is an independent draw of the instance family, so the same satellite has a different value, propellant budget and disruption cost in each of its conjunctions. The proofs above use no link between them: they hold for any cost table per slot. The numbers of a run do depend on it.
- **Learning** is not modelled on the batch (section 7.4, consequence 4).

---

## 8. References

Collision probability
- Foster, J. L., and Estes, H. S., "A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles", NASA/JSC-25898, 1992. (Cited through later papers; the report itself was not opened.)
- Akella, M. R., and Alfriend, K. T., "Probability of Collision Between Space Objects", J. Guidance, Control, and Dynamics 23(5), 2000, pp. 769-772.
- Alfano, S., "A Numerical Implementation of Spherical Object Collision Probability", J. Astronautical Sciences 53(1), 2005, pp. 103-109.
- Alfano, S., "Review of Conjunction Probability Methods for Short-term Encounters", AAS 07-148, 2007.
- Chan, F. K., "Spacecraft Collision Probability", The Aerospace Press / AIAA, 2008, doi:10.2514/4.989186. Chan, K., "Collision Probability Analyses for Earth-Orbiting Satellites", 1997.
- Serra, R., Arzelier, D., Joldes, M., Lasserre, J.-B., Rondepierre, A., Salvy, B., "Fast and Accurate Computation of Orbital Collision Probability for Short-Term Encounters", J. Guidance, Control, and Dynamics 39(5), 2016, pp. 1009-1021, doi:10.2514/1.G001353. Test values taken from the Orekit regression test `Laas2015Test.java` (https://github.com/CS-SI/Orekit) and cross-checked with the authors' JNCF 2014 slides and LLNL-TR-827434.
- NASA CARA Analysis Tools, `Pc2D_Foster.m` and its unit test, https://github.com/nasa/CARA_Analysis_Tools.

Relative motion and maneuvers
- Clohessy, W. H., and Wiltshire, R. S., "Terminal Guidance System for Satellite Rendezvous", J. Aerospace Sciences 27(9), 1960, pp. 653-658, doi:10.2514/8.8704.
- MIT OpenCourseWare 16.346 Astrodynamics (2008), lecture 26, "The Clohessy-Wiltshire Equations of Relative Motion".
- Schaub, H., "Relative Orbit Geometry Through Classical Orbit Element Differences", J. Guidance, Control, and Dynamics 27(5), 2004, pp. 839-848.
- Aida, S., "Conjunction Risk Assessment and Avoidance Maneuver Planning Tools", ICATT 2016, https://elib.dlr.de/103586/1/ICATT2016.pdf.
- Bombardelli, C., and Hernando-Ayuso, J., "Optimal Impulsive Collision Avoidance in Low Earth Orbit", J. Guidance, Control, and Dynamics 38(2), 2015, pp. 217-225 (the general map from an impulse to the encounter plane).

Games and information
- Aumann, R. J., "Subjectivity and Correlation in Randomized Strategies", J. Mathematical Economics 1(1), 1974, pp. 67-96; "Correlated Equilibrium as an Expression of Bayesian Rationality", Econometrica 55(1), 1987, pp. 1-18.
- Im, J., Topcu, U., Fridovich-Keil, D., "Chance-Constrained Correlated Equilibria for Robust Noncooperative Coordination", arXiv:2603.14141; IEEE Control Systems Letters 10, 2026, pp. 1273-1278.
- Im, J., Topcu, U., Fridovich-Keil, D., "Scaling Chance-Constrained Correlated Equilibrium Computation for Exclusive Resource-Assignment Games", arXiv:2609.13565 (v1, September 2026). (Read through notes made for this project from the arXiv v1 text; the paper itself was not opened for this document. Used in section 7.4.)
- Nash, J., "Non-Cooperative Games", Annals of Mathematics 54(2), 1951, pp. 286-295. (Cited for the standard existence theorem of a mixed equilibrium, used in section 7.3; not opened.)
- Charnes, A., and Cooper, W. W., "Deterministic Equivalents for Optimizing and Satisficing under Chance Constraints", Operations Research 11(1), 1963 (cited through the CC-CE paper; not opened).
- Karabag, M. O., Smith, S., Mehr, N., Fridovich-Keil, D., Topcu, U., "When Should a Leader Act Suboptimally? The Role of Inferability in Repeated Stackelberg Games", arXiv:2310.00468.
- Probine, C., Karabag, M. O., Topcu, U., "Designing Inferable Signaling Schemes for Bayesian Persuasion", arXiv:2510.01434.
- Babichenko, Y., Barman, S., Peretz, R., "Empirical Distribution of Equilibrium Play and Its Testing Application", arXiv:1310.7654.
- Sood, A., and Best, J., "Frequentist Persuasion", arXiv:2510.25066. Zhang and Li, arXiv:2608.20766. (Both found in the prior-art check; not opened for this document.)

Sampling guarantees
- Calafiore, G. C., and Campi, M. C., "The Scenario Approach to Robust Control Design", IEEE Trans. Automatic Control 51(5), 2006, pp. 742-753.
- Campi, M. C., and Garatti, S., "The Exact Feasibility of Randomized Solutions of Uncertain Convex Programs", SIAM J. Optimization 19(3), 2008, pp. 1211-1230.
- Calafiore, G. C., "Random Convex Programs", SIAM J. Optimization 20(6), 2010, pp. 3427-3464 (programs that may be infeasible).
- Li, J., Chiu, C.-Y., Peters, L., Palafox, F., Karabag, M., Alonso-Mora, J., Sojoudi, S., Tomlin, C., Fridovich-Keil, D., "Scenario-Game ADMM: A Parallelized Scenario-Based Solver for Stochastic Noncooperative Games", arXiv:2304.01945.
- Hoffman, A. J., and Kruskal, J. B., "Integral Boundary Points of Convex Polyhedra", in H. W. Kuhn and A. W. Tucker (eds.), Linear Inequalities and Related Systems, Annals of Mathematics Studies 38, Princeton University Press, 1956. (Not opened; cited for the standard result that a totally unimodular system with integer right-hand side has integral vertices, section 7.5.)
- Schrijver, A., Theory of Linear and Integer Programming, Wiley, 1986. (Not opened; cited for the standard results that the incidence matrix of a bipartite graph is totally unimodular, that total unimodularity survives sign changes of rows and appended unit rows, and the integrality result above, section 7.5.)
- Hoeffding, W., "Probability Inequalities for Sums of Bounded Random Variables", J. American Statistical Association 58(301), 1963, pp. 13-30.
- Clopper, C. J., and Pearson, E. S., "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial", Biometrika 26(4), 1934, pp. 404-413.
