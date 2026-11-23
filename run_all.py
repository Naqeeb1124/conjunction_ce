from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from conjce import exact_checks as X
from conjce import experiments as E
from conjce import games as G
from conjce import model as M

ROOT = pathlib.Path(__file__).resolve().parent
PRIVATE = ("private_m0", "private_uniform", "private_targeted_3", "private_targeted_6")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main(make_figures: bool = False) -> dict:
    t_start = time.time()
    scn = M.BASELINE
    c1, c2, obj, pc = E.nominal(scn)
    schemes, families, info = E.baseline_schemes(scn)
    res = {"scenario": E.scenario_dict(scn), "scheme_info": info}

    res["baseline"] = {
        "pc_table": pc, "miss_xi_table": M.miss_xi_table(scn), "cost_op1": c1, "cost_op2": c2,
        "coordinator_cost_table": obj, "maneuver_cost_fraction": M.maneuver_cost_fraction(scn),
        "pure_nash": G.pure_nash(c1, c2),
        "nash": [{"p": p, "q": q, **M.outcome_metrics(scn, G.product_table(p, q))}
                 for p, q in G.support_enumeration(c1, c2)],
        "fixed_rule": M.fixed_rule_profile(scn),
        "pc_vs_k": {str(k): M.pc_table(scn, k) for k in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)},
    }
    res["families"] = {name: [{"mu": f["mu"], "objective": f["objective"], "table": f["x"],
                               **M.outcome_metrics(scn, f["x"])} for f in fam]
                       for name, fam in families.items()}

    res["nominal_comparison"] = E.nominal_comparison(scn)
    public_cost = res["nominal_comparison"]["public"]["coordinator_cost"]

    res["learning_curves"] = {name: E.learning_curve(scn, schemes[name], seed=10 + i)
                              for i, name in enumerate(("public",) + PRIVATE)}
    res["learning_curve_margins"] = {"private_m0": 0.0, "private_uniform": info["mu_uniform_max"],
                                     "private_targeted_3": info["mu_targeted_3"],
                                     "private_targeted_6": info["mu_targeted_6"]}
    best = {name: E.best_margin_per_k(scn, fam, seed=20) for name, fam in families.items()}
    res["best_margin_per_k"] = best
    res["crossover_k"] = {name: E.crossover(rows, public_cost) for name, rows in best.items()}
    res["margin_scan"] = {name: E.margin_scan(scn, fam) for name, fam in families.items()}
    res["two_parameter_check"] = E.two_parameter_check(scn, families["targeted"], grid_n=25)
    res["exact_checks"] = X.run(scn)

    games = M.sample_true_games(scn, 10000, np.random.default_rng(30))
    res["cost_uncertainty"] = E.cost_uncertainty_eval(scn, schemes, games)
    res["cost_uncertainty"]["nash_true_game"] = E.true_game_nash(scn, games)
    res["cost_uncertainty_settings"] = {"n": 10000, "sig_fuel": 0.3, "sig_dis": 0.3, "sig_val": 0.3, "sig_k": 0.1}
    games_low = M.sample_true_games(scn, 10000, np.random.default_rng(31), 0.1, 0.1, 0.1, 0.05)
    res["cost_uncertainty_low"] = E.cost_uncertainty_eval(scn, schemes, games_low)
    res["cost_uncertainty_low"]["nash_true_game"] = E.true_game_nash(scn, games_low)
    res["cost_uncertainty_low_settings"] = {"n": 10000, "sig_fuel": 0.1, "sig_dis": 0.1, "sig_val": 0.1, "sig_k": 0.05}
    res["uncertainty_sweep"] = E.uncertainty_sweep(scn, schemes)
    res["calibration"] = E.calibrate_margin(scn, families)
    res["calibration_sigma_0.1"] = E.calibrate_margin(scn, families, sig_cost=0.1, sig_k=0.05, seed=40)
    res["safety_constraint"] = E.safety_constraint_check(scn)
    res["published_pc_variant"] = E.published_pc_variant(scn, {k: schemes[k] for k in ("public",) + PRIVATE})

    res["learning_with_cost_uncertainty"] = E.learning_with_cost_uncertainty(scn, families, schemes)

    res["instance_family"] = E.private_vs_public_family()

    res["scenario_program"] = E.scenario_program_check(scn)
    res["scenario_program_small_uncertainty"] = E.scenario_program_check(scn, sig_cost=0.05, sig_k=0.02, seed=50)

    res["runtime_seconds"] = time.time() - t_start
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(_jsonable(res), indent=1), encoding="utf-8")
    if make_figures:
        from conjce import figures
        figures.make_all(res, ROOT / "figures")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true", help="also write the PNG figures")
    args = parser.parse_args()
    result = main(make_figures=args.figures)
    print("done in %.1f s; results/results.json written" % result["runtime_seconds"])
