"""Figure for the report: what each stage of the final pipeline does on one DEV image.

    python scripts/stage_figure.py [--image not_damaged_136] [--trial T11]

Re-runs the stages one by one (same functions as classical.segment) and saves
report/figures/pipeline_stages.jpg.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from eggseg import classical as C  # noqa: E402
from eggseg.config import config_path, load_trial  # noqa: E402
from eggseg.data import load_gt, load_rgb, load_split  # noqa: E402
from eggseg.runner import DATA, GT, SPLIT  # noqa: E402
from eggseg.viz import overlay, save_rgb  # noqa: E402

PALETTE = [(42, 120, 214), (235, 104, 52), (60, 170, 90), (170, 80, 200)]


def tile(img, text, size=300):
    t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (size, 24), (0, 0, 0), -1)
    cv2.putText(t, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def colour_instances(rgb, inst):
    out = (rgb * 0.45).astype(np.uint8)
    for k in range(1, int(inst.max()) + 1):
        out[inst == k] = PALETTE[(k - 1) % len(PALETTE)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="not_damaged_136")
    ap.add_argument("--trial", default="T11")
    a = ap.parse_args()
    key = next(k for k in load_split(SPLIT)["dev"] if Path(k).name.startswith(a.image + "_"))
    cfg = load_trial(config_path(a.trial))["pipeline"]
    rgb = load_rgb(DATA / key)

    img = C.preprocess(rgb, cfg["preprocess"])
    allowed = ~C.skin(img, cfg["skin"]["cb"], cfg["skin"]["cr"]) & C.low_saturation(img, cfg["saturation"]["s_max"])
    lch = C.channel(img, cfg["candidate"]["channel"])
    found = C.threshold_search(img, cfg["candidate"], cfg["components"], allowed)
    m = C.morphology(found, cfg["morphology"])
    refined = C.grabcut_refine(rgb, m, cfg["grabcut"])
    inst = C.split_instances(refined, cfg["watershed"])
    gt, _ = load_gt(GT, key)

    gate = rgb.copy()
    gate[~allowed] = (gate[~allowed] * 0.25).astype(np.uint8)
    lvis = cv2.cvtColor(cv2.normalize(lch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    tiles = [tile(rgb, "1. input"),
             tile(gate, "2. colour gate (not skin, low sat.)"),
             tile(lvis, "3. L* channel (thresholded 24x)"),
             tile(overlay(rgb, found, (42, 120, 214)), "4. egg-shaped blobs picked"),
             tile(overlay(rgb, refined, (42, 120, 214)), "5. morphology + GrabCut"),
             tile(colour_instances(rgb, inst), f"6. watershed: {int(inst.max())} instances"),
             tile(overlay(rgb, gt, (60, 170, 90)), "GT (SAM 2.1)")]
    tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    out = ROOT / "report" / "figures" / "pipeline_stages.jpg"
    save_rgb(out, sheet)
    print(key, "->", out)


if __name__ == "__main__":
    main()
