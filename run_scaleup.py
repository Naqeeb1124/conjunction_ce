from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from conjce import batch

ROOT = pathlib.Path(__file__).resolve().parent


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def main(make_figures: bool = False) -> dict:
    try:
        res = batch.run()
    except batch.CheckFailed as err:
        print(f"check failed, no results written: {err}", file=sys.stderr)
        sys.exit(1)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "scaleup.json").write_text(json.dumps(_jsonable(res), indent=1), encoding="utf-8")
    if make_figures:
        from conjce import batch_figures
        batch_figures.make(json.loads((out_dir / "scaleup.json").read_text(encoding="utf-8")),
                           ROOT / "figures" / "scaleup.png")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true", help="also write figures/scaleup.png")
    args = parser.parse_args()
    result = main(make_figures=args.figures)
    print("done in %.1f s; results/scaleup.json written" % result["runtime_seconds"])
