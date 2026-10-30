"""Ablation of the final pipeline on DEV: switch off one stage at a time.

    python scripts/ablation.py

Run after final_eval.py, as part of the Task 3 analysis, to see what each stage
still contributes in the final pipeline (several were accepted early, on top of a
much simpler method). Dev only; eval is not re-run, and the reported eval numbers
stay those of the frozen pipeline. Writes results/ablation/ablation.json.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from eggseg.config import config_path, load_trial, set_by_path  # noqa: E402
from eggseg.data import load_split  # noqa: E402
from eggseg.metrics import summarise  # noqa: E402
from eggseg.runner import SPLIT, run  # noqa: E402

VARIANTS = [  # (label, dotted param, value)
    ("final pipeline", None, None),
    ("no skin filter", "skin.enabled", False),
    ("no saturation filter", "saturation.enabled", False),
    ("no morphology", "morphology.enabled", False),
    ("no GrabCut", "grabcut.enabled", False),
    ("no fallback pass", "candidate.fallback_min_score", None),
    ("no watershed", "watershed.enabled", False),
]


def main():
    trial = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())["trial"]
    base = load_trial(config_path(trial))["pipeline"]
    dev = load_split(SPLIT)["dev"]
    out = {"base": trial, "split": "dev", "runs": []}
    for label, param, value in VARIANTS:
        cfg = set_by_path(base, param, value) if param else copy.deepcopy(base)  # returns a copy
        assert (cfg != base) == bool(param), label  # a variant must really differ from the base
        rows, _ = run(cfg, dev)
        s = summarise(rows)
        out["runs"].append({"label": label, "param": param, "value": value,
                            "iou_mean": s["iou"]["mean"], "iou_median": s["iou"]["median"],
                            "precision": s["precision"]["mean"], "recall": s["recall"]["mean"],
                            "boundary_f1": s["boundary_f1"]["mean"], "instance_f1": s["instance"]["f1"],
                            "empty": s["empty_predictions"],
                            "seconds_per_image": round(float(np.mean([r["seconds"] for r in rows])), 3)})
        r = out["runs"][-1]
        print(f"{label:>22}: IoU {r['iou_mean']:.3f}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"BF1 {r['boundary_f1']:.3f}  instF1 {r['instance_f1']:.3f}  empty {r['empty']}", flush=True)
    dest = ROOT / "results" / "ablation"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ablation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
