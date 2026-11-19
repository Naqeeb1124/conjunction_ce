"""The baseline two-operator conjunction and its cost model.

Everything here is synthetic.  The numbers were fixed from the design principles listed in
RESULTS.md (section "How the baseline was chosen") before any comparison between coordination
schemes was run.

Indexing convention used across the package: joint actions are (a1, a2) with a_i in {0, 1, 2}
meaning no burn, small burn, large burn, and every 3x3 table is indexed [a1, a2].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import maneuver as mv
from .pc import pc_foster

ACTION_LABELS = ("N", "S", "L")          # none, small, large
ACTION_NAMES = ("no burn", "small burn", "large burn")


@dataclass(frozen=True)
class Scenario:
    # --- orbit and encounter geometry -------------------------------------------------------
    altitude: float = 550e3              # m, both satellites near-circular at this altitude
    crossing_angle_deg: float = 90.0     # angle between the two velocity vectors at closest approach
    lead_orbits: float = 2.0             # burns are executed this many orbits before closest approach
    burns: tuple = (0.0, 0.01, 0.03)     # m/s: none, small, large (tangential, miss-increasing sense)
    # --- encounter plane (xi = bisector of the velocity directions, zeta = radial) ----------
    sigma_xi: float = 115.0              # m, combined 1-sigma along xi
    sigma_zeta: float = 40.0             # m, combined 1-sigma along zeta
    hbr: float = 5.0                     # m, combined hard-body radius
    miss_xi: float = 92.0                # m, nominal miss vector, xi component
    miss_zeta: float = 30.0              # m, nominal miss vector, zeta component
    # --- operators (index 0 = operator 1, index 1 = operator 2) -----------------------------
    value: tuple = (200e6, 80e6)         # USD lost by the operator if its satellite is destroyed
    dv_budget: tuple = (100.0, 100.0)    # m/s lifetime budget; propellant is priced at value / budget
    disruption: tuple = (60e3, 16e3)     # USD per maneuver, independent of its size
    # --- coordinator -------------------------------------------------------------------------
    lam: float = 1000.0                  # m/s of total delta-v the coordinator would spend per unit of Pc
    pc_threshold: float = 1e-4           # the usual maneuver threshold, used for reporting

    def __post_init__(self):
        # The encounter-plane gain 3 * dt * cos(phi / 2) uses the secular term of the
        # Clohessy-Wiltshire solution only; the periodic term vanishes after whole orbits.
        lo = float(self.lead_orbits)
        if not (np.isfinite(lo) and lo > 0 and lo.is_integer()):
            raise ValueError(f"lead_orbits must be a positive whole number of orbits, got {self.lead_orbits!r}")

    # derived quantities --------------------------------------------------------------------
    @property
    def sma(self) -> float:
        return mv.R_EARTH + self.altitude

    @property
    def period(self) -> float:
        return mv.orbital_period(self.sma)

    @property
    def lead_time(self) -> float:
        return self.lead_orbits * self.period

    @property
    def gain(self) -> float:
        """Metres of miss change along xi per m/s of burn (same for both operators here)."""
        return mv.bplane_gain(self.lead_time, np.deg2rad(self.crossing_angle_deg))

    @property
    def fuel_price(self) -> tuple:
        """USD per m/s for each operator."""
        return tuple(v / b for v, b in zip(self.value, self.dv_budget))

    @property
    def n_actions(self) -> int:
        return len(self.burns)


BASELINE = Scenario()


def burn_tables(scn: Scenario):
    """Delta-v of operator 1, of operator 2, and their sum, as 3x3 tables indexed [a1, a2]."""
    u = np.asarray(scn.burns, dtype=float)
    dv1 = np.repeat(u[:, None], len(u), axis=1)
    dv2 = np.repeat(u[None, :], len(u), axis=0)
    return dv1, dv2, dv1 + dv2


def miss_xi_table(scn: Scenario) -> np.ndarray:
    """Miss-vector xi component after each joint action.

    Each operator burns in the sense that increases the nominal miss, so the two shifts add:
    xi(a) = xi0 + gain * (u1 + u2).
    """
    return scn.miss_xi + scn.gain * burn_tables(scn)[2]


def pc_table(scn: Scenario, k=1.0) -> np.ndarray:
    """Collision probability for each joint action.

    ``k`` scales both sigmas (the covariance is multiplied by k^2).  It may be an array, in which
    case the result has shape k.shape + (3, 3).
    """
    k = np.asarray(k, dtype=float)
    xi = miss_xi_table(scn)
    return pc_foster(xi, scn.miss_zeta, k[..., None, None] * scn.sigma_xi,
                     k[..., None, None] * scn.sigma_zeta, scn.hbr)


def maneuver_cost_fraction(scn: Scenario, fuel_price=None, disruption=None, value=None):
    """Each operator's own maneuver cost per action, as a fraction of its satellite's value.

    Returns an array of shape (..., 2, 3): [operator, action].  Inputs may carry a leading sample
    dimension, shape (n, 2).
    """
    fuel_price = np.asarray(scn.fuel_price if fuel_price is None else fuel_price, dtype=float)
    disruption = np.asarray(scn.disruption if disruption is None else disruption, dtype=float)
    value = np.asarray(scn.value if value is None else value, dtype=float)
    u = np.asarray(scn.burns, dtype=float)
    usd = fuel_price[..., None] * u + disruption[..., None] * (u > 0)
    return usd / value[..., None]


def cost_tables(scn: Scenario, pc=None, fuel_price=None, disruption=None, value=None):
    """Operator costs as fractions of each operator's satellite value.

        c_i(a) / V_i = (w_fuel_i * dv_i + w_dis_i * 1[i burns]) / V_i + Pc(a)

    Dividing by V_i does not change any incentive comparison for operator i, and it puts both
    operators' costs in the same unit as a collision probability.  Returns (c1, c2), each of
    shape (..., 3, 3).
    """
    pc = pc_table(scn) if pc is None else np.asarray(pc, dtype=float)
    frac = maneuver_cost_fraction(scn, fuel_price, disruption, value)
    c1 = frac[..., 0, :, None] + pc
    c2 = frac[..., 1, None, :] + pc
    return c1, c2


def coordinator_cost_table(scn: Scenario, pc=None) -> np.ndarray:
    """The coordinator's cost of each joint action: total delta-v + lam * Pc (in m/s)."""
    pc = pc_table(scn) if pc is None else np.asarray(pc, dtype=float)
    return burn_tables(scn)[2] + scn.lam * pc


def fixed_rule_profile(scn: Scenario, cheaper: bool = True):
    """A fixed, cost-based responsibility rule.

    For each operator, find the smallest burn that alone brings the nominal Pc to or below the
    threshold and price it in USD.  With ``cheaper=True`` the operator with the lower price does
    that burn and the other does nothing; ``cheaper=False`` designates the other operator instead
    (a stand-in for a rule that ignores costs, such as a category or tie-break rule).
    Returns the joint action (a1, a2).
    """
    pc = pc_table(scn)
    u = np.asarray(scn.burns, dtype=float)
    prices, choices = [], []
    for i in range(2):
        solo = pc[:, 0] if i == 0 else pc[0, :]
        ok = [a for a in range(1, len(u)) if solo[a] <= scn.pc_threshold]
        a = ok[0] if ok else len(u) - 1
        choices.append(a)
        prices.append(scn.fuel_price[i] * u[a] + scn.disruption[i])
    mover = int(np.argmin(prices)) if cheaper else int(np.argmax(prices))
    return (choices[0], 0) if mover == 0 else (0, choices[1])


def sample_true_games(scn: Scenario, n: int, rng: np.random.Generator,
                      sig_fuel: float = 0.3, sig_dis: float = 0.3, sig_val: float = 0.3,
                      sig_k: float = 0.1) -> dict:
    """Draw n "true" games the coordinator does not observe.

    Each operator's propellant price, disruption cost and satellite value are lognormal with the
    nominal value as median; the covariance sigma-scale k is lognormal with median 1.  ``sig_*``
    are the standard deviations of the logarithms.
    """
    def logn(nominal, sig):
        nominal = np.asarray(nominal, dtype=float)
        return nominal * np.exp(sig * rng.standard_normal((n,) + nominal.shape))

    fuel = logn(scn.fuel_price, sig_fuel)
    dis = logn(scn.disruption, sig_dis)
    val = logn(scn.value, sig_val)
    k = np.exp(sig_k * rng.standard_normal(n))
    pc = pc_table(scn, k)
    c1, c2 = cost_tables(scn, pc=pc, fuel_price=fuel, disruption=dis, value=val)
    return {"c1": c1, "c2": c2, "pc": pc, "k": k, "fuel_price": fuel, "disruption": dis, "value": val}


def outcome_metrics(scn: Scenario, y, pc=None) -> dict:
    """Summary of a joint-action distribution y (3x3, or (n, 3, 3) paired with pc of shape (n, 3, 3)).

    Averages over the leading sample dimension when there is one.
    """
    y = np.asarray(y, dtype=float)
    pc = pc_table(scn) if pc is None else np.asarray(pc, dtype=float)
    dv = burn_tables(scn)[2]
    mean = (lambda t: float(np.mean(np.sum(t, axis=(-2, -1)))))
    out = {
        "delta_v": mean(y * dv),
        "mean_pc": mean(y * pc),
        "p_exceed": mean(y * (pc > scn.pc_threshold)),
        "p_dual_inaction": float(np.mean(y[..., 0, 0])),
    }
    out["coordinator_cost"] = out["delta_v"] + scn.lam * out["mean_pc"]
    return out
