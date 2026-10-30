"""Small parameter sweep on the DEV split around one trial's config.

    python scripts/sweep.py experiments/T3/config.yaml skin.cr "[[133,173],[128,173],[138,173]]" --name skin_cr

Writes experiments/sweeps/<name>.json; the log builder turns it into a table.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eggseg.config import load_trial, set_by_path  # noqa: E402
from eggseg.data import load_split  # noqa: E402
from eggseg.metrics import summarise  # noqa: E402
from eggseg.runner import SPLIT, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("param", help="dotted path inside the pipeline, e.g. components.min_solidity")
    ap.add_argument("values", help="JSON list of values to try")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    trial = load_trial(a.config)
    keys = load_split(SPLIT)["dev"]
    out = {"base": trial["id"], "param": a.param, "split": "dev", "runs": []}
    for v in json.loads(a.values):
        s = summarise(run(set_by_path(trial["pipeline"], a.param, v), keys)[0])
        out["runs"].append({"value": v, "iou_mean": s["iou"]["mean"], "iou_median": s["iou"]["median"],
                            "boundary_f1": s["boundary_f1"]["mean"], "instance_f1": s["instance"]["f1"],
                            "empty": s["empty_predictions"]})
        print(f"{a.param}={v}: IoU {s['iou']['mean']:.3f} (med {s['iou']['median']:.3f})  "
              f"BF1 {s['boundary_f1']['mean']:.3f}  instF1 {s['instance']['f1']:.3f}  empty {s['empty_predictions']}")
    path = ROOT / "experiments" / "sweeps" / f"{a.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
