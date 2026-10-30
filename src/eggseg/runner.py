"""Run the classical pipeline over a list of images and score it against the pseudo-GT."""
import time
from pathlib import Path

from .classical import segment
from .data import load_gt, load_rgb
from .metrics import image_metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "egg_dataset"
GT = ROOT / "ground_truth"
SPLIT = ROOT / "data" / "splits" / "split.csv"


def run(pipeline, keys, keep_preds=False):
    """Returns (per-image rows, {key: (mask, instances)} if keep_preds)."""
    rows, preds = [], {}
    for key in keys:
        rgb = load_rgb(DATA / key)
        t0 = time.perf_counter()
        mask, inst = segment(rgb, pipeline)
        dt = time.perf_counter() - t0
        gt_mask, gt_inst = load_gt(GT, key)
        rows.append({"image": key, **image_metrics(mask, inst, gt_mask, gt_inst), "seconds": round(dt, 3)})
        if keep_preds:
            preds[key] = (mask, inst)
    return rows, preds
