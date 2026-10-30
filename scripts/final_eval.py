"""The ONE run on the eval split (Task 3).

    python scripts/final_eval.py

Reads the chosen trial from configs/final.yaml, runs it on dev and eval, and saves
results/final_eval/{summary.json, dev_per_image.csv, eval_per_image.csv} plus the
predicted instance masks. Refuses to run again once eval results exist, so eval
can't quietly become a tuning set; --i-know-this-reuses-eval overrides that.

Round 2 (improvements designed after looking at the round-1 eval failures) writes to
its own folder and never overwrites round 1:
    python scripts/final_eval.py --config configs/final_round2.yaml --out results/final_eval_round2
The config's `note` is stored in summary.json, so the second use of eval is on record.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from eggseg.config import config_path, load_trial  # noqa: E402
from eggseg.data import load_split  # noqa: E402
from eggseg.metrics import summarise  # noqa: E402
from eggseg.runner import SPLIT, run  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import write_rows  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--out", default="results/final_eval")
    ap.add_argument("--i-know-this-reuses-eval", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out
    if (out / "eval_per_image.csv").exists() and not a.i_know_this_reuses_eval:
        sys.exit(f"{a.out} already has eval results. Eval is touched once per round; refusing to run again.")

    choice = yaml.safe_load((ROOT / a.config).read_text())
    trial = load_trial(config_path(choice["trial"]))
    split = load_split(SPLIT)
    summary = {"trial": trial["id"], "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "config": a.config, "note": choice.get("note", ""), "pipeline": trial["pipeline"]}
    for name in ("dev", "eval"):
        rows, preds = run(trial["pipeline"], split[name], keep_preds=True)
        summary[name] = summarise(rows, worst_k=8)
        summary[name]["seconds_per_image"] = round(float(np.mean([r["seconds"] for r in rows])), 3)
        out.mkdir(parents=True, exist_ok=True)
        write_rows(out / f"{name}_per_image.csv", rows)
        for key, (_, inst) in preds.items():  # instance ids as PNG (0 = background)
            p = out / "preds" / Path(key).parent / (Path(key).stem + ".png")
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), np.clip(inst, 0, 255).astype(np.uint8))
        s = summary[name]
        print(f"{name:>4}: IoU {s['iou']['mean']:.3f} (med {s['iou']['median']:.3f})  Dice {s['dice']['mean']:.3f}  "
              f"P {s['precision']['mean']:.3f}  R {s['recall']['mean']:.3f}  BF1 {s['boundary_f1']['mean']:.3f}  "
              f"instF1 {s['instance']['f1']:.3f}  empty {s['empty_predictions']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
