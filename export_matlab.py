"""Export the model parameters and the reference numbers that the MATLAB layer reads.

    .venv\\Scripts\\python export_matlab.py

writes

    matlab/params.json     every model parameter, taken from conjce (nothing is typed again in MATLAB)
    matlab/expected.json   the numbers the MATLAB code has to reproduce

Deterministic quantities are recomputed here with the conjce functions and compared with
results/results.json before anything is written; the script stops if they differ.  Simulated
quantities (the learning curves) are copied from results/results.json.

Index convention in both files: actions are 0 = no burn, 1 = small burn, 2 = large burn, and every
3x3 table is indexed [a1][a2] (operator 1 = row).  Flattened tables are row-major (a1 * 3 + a2).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from conjce import experiments as E
from conjce import games as G
from conjce import maneuver as mv
from conjce import model as M

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "matlab"

# Sense of each operator's tangential burn along its own velocity: posigrade for satellite 1,
# retrograde for satellite 2, so that both increase the nominal miss.
BURN_SIGN = (1.0, -1.0)

# Settings of the two-body check on the MATLAB side (they are not part of the game model).
ODE_RELTOL = 1e-13
ODE_ABSTOL = 1e-9
TCA_TOL_S = 1e-10
MC_SAMPLES = 2000
MC_SEED = 1
MC_PROPOSAL_RADIUS_HBR = 2.0

# The gallery uses the first draws of the random instance family (same generator, same seed).
GALLERY_SEED = 4
GALLERY_SIZE = 12

LEARNING_RULES = ("public", "private_m0", "private_uniform", "private_targeted_3", "private_targeted_6")
LEARNING_FIELDS = ("coordinator_cost", "cost_se", "obey", "p_dual_inaction", "delta_v", "mean_pc")
SCHEMES = ("fixed_rule", "nash_mixed", "public", "private_m0", "private_uniform",
           "private_targeted_3", "private_targeted_6")
SCHEME_FIELDS = ("table", "delta_v", "mean_pc", "p_exceed", "p_dual_inaction", "coordinator_cost")


def clean(o):
    """Plain Python containers and numbers, so that json can write them."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    return o


def dumps(o, level: int = 0) -> str:
    """JSON with one key per line and each array on one line."""
    pad = "  " * level
    if isinstance(o, dict):
        items = [f"{pad}  {json.dumps(k)}: {dumps(v, level + 1)}" for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(o, list) and o and isinstance(o[0], dict):
        items = [f"{pad}  {dumps(v, level + 1)}" for v in o]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(o)


def need_close(name, got, want, rtol=1e-12, atol=1e-15):
    if not np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=rtol, atol=atol):
        raise SystemExit(f"export_matlab: {name} differs from results/results.json; rerun run_all.py first")


def derived(scn: M.Scenario) -> dict:
    return {
        "sma_m": scn.sma,
        "mean_motion_rad_s": mv.mean_motion(scn.sma),
        "period_s": scn.period,
        "lead_time_s": scn.lead_time,
        "circular_speed_mps": mv.circular_speed(scn.sma),
        "gain_m_per_mps": scn.gain,
        "fuel_price_usd_per_mps": list(scn.fuel_price),
    }


def sigma_s(scn: M.Scenario) -> float:
    """1-sigma of the relative position along the relative velocity.

    The game model does not need it (the collision probability depends only on the encounter
    plane).  It is used to draw Monte Carlo samples and the horizontal uncertainty footprint, and
    is set to sigma_xi * tan(phi/2): the value that results when the combined uncertainty comes
    from along-track errors of equal size on the two satellites.
    """
    return scn.sigma_xi * float(np.tan(0.5 * np.deg2rad(scn.crossing_angle_deg)))


def game(scn: M.Scenario) -> dict:
    """Tables of one conjunction and the coordinator's best private and public rules."""
    c1, c2, obj, pc = E.nominal(scn)
    prv = G.solve_private(c1, c2, obj)
    pub = G.solve_public(c1, c2, obj)
    x = prv["x"]
    modal = np.unravel_index(int(np.argmax(x)), x.shape)
    return {
        "derived": derived(scn),
        "miss_xi_table": M.miss_xi_table(scn),
        "pc_table": pc,
        "maneuver_cost_fraction": M.maneuver_cost_fraction(scn),
        "cost_op1": c1,
        "cost_op2": c2,
        "coordinator_cost_table": obj,
        "ce_x": x,
        "ce_objective": prv["objective"],
        "ce_modal_recommendation": [int(modal[0]), int(modal[1])],
        "ce_modal_probability": float(x[modal]),
        "pure_nash": [list(a) for a in G.pure_nash(c1, c2)],
        "public_x": pub["x"] if pub["x"] is not None else np.full(x.shape, -1.0),
        "public_objective": pub["objective"] if pub["x"] is not None else -1.0,
        "has_public_rule": pub["x"] is not None,
    }


def two_body_miss(scn: M.Scenario, u1: float, u2: float):
    """Encounter-plane miss vector after two-body propagation of both satellites.

    Same construction as tests/test_maneuver.py: states at the nominal time of closest approach,
    back to the burn epoch, burn along each satellite's own velocity, forward again, straight-line
    closest approach, projection on the nominal encounter-plane axes.
    """
    a, v, dt = scn.sma, mv.circular_speed(scn.sma), scn.lead_time
    half = 0.5 * np.deg2rad(scn.crossing_angle_deg)
    t1 = np.array([0.0, np.cos(half), np.sin(half)])
    t2 = np.array([0.0, np.cos(half), -np.sin(half)])
    r1 = np.array([a, 0.0, 0.0])
    r2 = r1 + np.array([scn.miss_zeta, scn.miss_xi, 0.0])
    v1, v2 = v * t1, v * t2
    xi_hat, zeta_hat, _ = mv.bplane_axes(v1, v2)
    tol = {"rtol": ODE_RELTOL, "atol": ODE_ABSTOL}
    r1b, v1b = mv.two_body_propagate(r1, v1, -dt, **tol)
    r2b, v2b = mv.two_body_propagate(r2, v2, -dt, **tol)
    v1b = v1b + BURN_SIGN[0] * u1 * v1b / np.linalg.norm(v1b)
    v2b = v2b + BURN_SIGN[1] * u2 * v2b / np.linalg.norm(v2b)
    r1f, v1f = mv.two_body_propagate(r1b, v1b, dt, **tol)
    r2f, v2f = mv.two_body_propagate(r2b, v2b, dt, **tol)
    t_star, miss = mv.closest_approach(r1f, v1f, r2f, v2f)
    return float(np.dot(miss, xi_hat)), float(np.dot(miss, zeta_hat)), float(t_star)


def two_body_tables(scn: M.Scenario) -> dict:
    n = scn.n_actions
    xi, zeta, t_star = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for a1, u1 in enumerate(scn.burns):
        for a2, u2 in enumerate(scn.burns):
            xi[a1, a2], zeta[a1, a2], t_star[a1, a2] = two_body_miss(scn, u1, u2)
    return {
        "method": "scipy DOP853 round trip, straight-line closest approach, nominal encounter-plane axes",
        "rtol": ODE_RELTOL, "atol": ODE_ABSTOL,
        "miss_xi": xi, "miss_zeta": zeta, "tca_shift_s": t_star,
    }


def main():
    res = json.loads((ROOT / "results" / "results.json").read_text())
    scn = M.BASELINE
    base = game(scn)

    # recomputed here versus stored by run_all.py
    stored = res["baseline"]
    for key in ("pc_table", "miss_xi_table", "cost_op1", "cost_op2", "coordinator_cost_table",
                "maneuver_cost_fraction"):
        need_close(key, base[key], stored[key])
    need_close("private_m0 table", base["ce_x"], res["nominal_comparison"]["private_m0"]["table"], atol=1e-10)
    need_close("public table", base["public_x"], res["nominal_comparison"]["public"]["table"], atol=1e-10)
    need_close("gain", scn.gain, res["scenario"]["gain_m_per_mps"])
    if sorted(map(tuple, base["pure_nash"])) != sorted(map(tuple, stored["pure_nash"])):
        raise SystemExit("export_matlab: pure Nash equilibria differ from results/results.json")

    c1, c2 = base["cost_op1"], base["cost_op2"]
    A, P, info = G.obedience_system(c1, c2)
    targeted = {tuple(r) for r in res["scheme_info"]["targeted_rows"]}
    p, q = (np.array(v) for v in res["scheme_info"]["mixed_nash"])
    p_here, q_here = E.select_mixed_nash(c1, c2)
    need_close("mixed Nash", np.concatenate([p_here, q_here]), np.concatenate([p, q]), atol=1e-10)

    margins = {
        "fixed_rule": 0.0, "nash_mixed": 0.0, "public": 0.0, "private_m0": 0.0,
        "private_uniform": res["scheme_info"]["mu_uniform_max"],
        "private_targeted_3": res["scheme_info"]["mu_targeted_3"],
        "private_targeted_6": res["scheme_info"]["mu_targeted_6"],
    }
    margin_rows = {"private_uniform": "all", "private_targeted_3": "targeted", "private_targeted_6": "targeted"}
    schemes = {}
    for name in SCHEMES:
        block = {f: res["nominal_comparison"][name][f] for f in SCHEME_FIELDS}
        block["margin"] = margins[name]
        block["margin_rows"] = margin_rows.get(name, "none")
        schemes[name] = block

    ks = [row["k"] for row in res["learning_curves"]["public"]]
    learning = {"k": ks, "rules": {}, "known_rule_cost": {}}
    for name in LEARNING_RULES:
        rows = res["learning_curves"][name]
        if [row["k"] for row in rows] != ks:
            raise SystemExit("export_matlab: learning curves do not share one k grid")
        learning["rules"][name] = {f: [row[f] for row in rows] for f in LEARNING_FIELDS}
        learning["known_rule_cost"][name] = res["nominal_comparison"][name]["coordinator_cost"]
    for family in ("targeted", "uniform"):
        rows = res["best_margin_per_k"][family]
        if [row["k"] for row in rows] != ks:
            raise SystemExit("export_matlab: best-margin curves do not share the k grid")
        learning[f"best_{family}"] = {f: [row[f] for row in rows]
                                      for f in ("coordinator_cost", "cost_se", "mu", "obey", "known_rule_cost")}
    learning["crossover_k"] = res["crossover_k"]

    expected = {
        "source": "conjce (Python); see export_matlab.py",
        "baseline": {
            **base,
            "obedience": {
                "A": A, "P": P,
                "info": [list(r) for r in info],            # (player, recommended, deviation), zero-based
                "targeted_mask": [tuple(r) in targeted for r in info],
            },
            "mixed_nash": {
                "p": p, "q": q,
                "table": res["nominal_comparison"]["nash_mixed"]["table"],
                "p_dual_inaction": res["nominal_comparison"]["nash_mixed"]["p_dual_inaction"],
                "coordinator_cost": res["nominal_comparison"]["nash_mixed"]["coordinator_cost"],
            },
            "schemes": schemes,
            "two_body_python": two_body_tables(scn),
        },
        "learning": learning,
    }

    # gallery: the first draws of the instance family
    rng = np.random.default_rng(GALLERY_SEED)
    gallery_params, gallery_expected = [], []
    family_rows = res["instance_family"]["rows"]
    for i in range(GALLERY_SIZE):
        s = E.random_instance(rng)
        g = game(s)
        need_close(f"gallery case {i} Pc", g["pc_table"][0, 0], family_rows[i]["pc_no_burn"])
        need_close(f"gallery case {i} private cost", g["ce_objective"], family_rows[i]["private_cost"], atol=1e-10)
        gallery_params.append({"id": i + 1, "scenario": dataclasses.asdict(s), "sigma_s": sigma_s(s)})
        gallery_expected.append({"id": i + 1, **g})
    expected["gallery"] = gallery_expected

    params = {
        "source": "conjce.model.BASELINE and conjce.maneuver; see export_matlab.py",
        "units": "SI: m, s, m/s, rad unless the name says otherwise; costs are fractions of satellite value; value and disruption in USD",
        "constants": {"mu_earth": mv.MU_EARTH, "r_earth": mv.R_EARTH, "burn_sign": list(BURN_SIGN)},
        "scenario": dataclasses.asdict(scn),
        "action_labels": list(M.ACTION_LABELS),
        "action_names": list(M.ACTION_NAMES),
        "two_body_check": {
            "ode_reltol": ODE_RELTOL, "ode_abstol": ODE_ABSTOL, "tca_tol_s": TCA_TOL_S,
            "mc_samples": MC_SAMPLES, "mc_seed": MC_SEED,
            "mc_proposal_radius_hbr": MC_PROPOSAL_RADIUS_HBR,
            "sigma_s": sigma_s(scn),
        },
        "gallery": gallery_params,
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "params.json").write_text(dumps(clean(params)) + "\n")
    (OUT / "expected.json").write_text(dumps(clean(expected)) + "\n")
    print(f"wrote {OUT / 'params.json'} and {OUT / 'expected.json'}")


if __name__ == "__main__":
    main()
