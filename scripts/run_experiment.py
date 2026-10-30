"""Run one trial on the DEV split and save its results.

    python scripts/run_experiment.py experiments/T3/config.yaml [--worst 6]

Writes experiments/<id>/results.json (summary), per_image.csv, resolved.yaml and,
with --worst N, error maps of the N worst dev images (for deciding the next trial).
There is deliberately no --split option: eval is only read by scripts/final_eval.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from eggseg.config import load_trial  # noqa: E402
from eggseg.data import load_gt, load_rgb, load_split  # noqa: E402
from eggseg.metrics import summarise  # noqa: E402
from eggseg.runner import DATA, GT, SPLIT, run  # noqa: E402
from eggseg.viz import error_panel, save_rgb  # noqa: E402


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--worst", type=int, default=0, help="save error maps of the N worst dev images")
    a = ap.parse_args()

    trial = load_trial(a.config)
    keys = load_split(SPLIT)["dev"]
    rows, preds = run(trial["pipeline"], keys, keep_preds=a.worst > 0)
    summary = summarise(rows)

    out = a.config.parent
    write_rows(out / "per_image.csv", rows)
    (out / "resolved.yaml").write_text(yaml.safe_dump(trial["pipeline"], sort_keys=False))
    result = {"id": trial["id"], "split": "dev", "summary": summary,
              "seconds_per_image": round(sum(r["seconds"] for r in rows) / len(rows), 3)}
    (out / "results.json").write_text(json.dumps(result, indent=2))

    for r in sorted(rows, key=lambda r: r["iou"])[:a.worst]:
        gt_mask, _ = load_gt(GT, r["image"])
        panel = error_panel(load_rgb(DATA / r["image"]), gt_mask, preds[r["image"]][0], f"IoU {r['iou']:.2f}")
        save_rgb(out / "worst" / (Path(r["image"]).stem[:20] + ".jpg"), panel)

    s = summary
    print(f"{trial['id']:>6}  IoU {s['iou']['mean']:.3f} (med {s['iou']['median']:.3f})  Dice {s['dice']['mean']:.3f}  "
          f"P {s['precision']['mean']:.3f}  R {s['recall']['mean']:.3f}  BF1 {s['boundary_f1']['mean']:.3f}  "
          f"instF1 {s['instance']['f1']:.3f}  empty {s['empty_predictions']}  {result['seconds_per_image']}s/img")


if __name__ == "__main__":
    main()
