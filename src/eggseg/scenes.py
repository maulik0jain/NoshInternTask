"""Round 3: scene-level decisions, to stop tuning to a few dev scenes.

Round 1 and 2 kept a change if the dev mean IoU over images went up. With 85 dev
images from 14 scenes, a big scene (12 images) counts four times as much as a
small one, and a gain of a few thousandths is within noise. Round 2 passed that
test on dev and did not transfer to eval. Round 3 decides per scene instead.

The rule (fixed before any round-3 run, see git history):
  unit         a scene (a split group): scene IoU = mean IoU of its images,
               score of a pipeline = mean of its scene IoUs (every scene counts once)
  keep a change only if, against the pipeline without it, ALL of:
    1. mean scene delta > 0
    2. scenes helped (delta > +HELP) outnumber scenes hurt (delta < -HELP)
    3. no scene gets worse by more than WORST
    4. scene bootstrap (resample the scenes with replacement, N_BOOT times):
       mean delta > 0 in at least P_MIN of the resamples
  selection    forward: start from the round-1 pipeline, try every change not yet in,
               add the passing change with the largest mean scene delta, repeat until
               none passes. No parameter sweeps: every change keeps a value fixed beforehand.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HELP = 0.01     # scene delta that counts as helped / hurt (ignores numerical no-ops)
WORST = 0.10    # no scene may lose more than this (0.1 = the per-image "real change" used throughout)
N_BOOT = 2000
P_MIN = 0.90
SEED = 0


def scene_groups(split_csv, split="dev"):
    """{image key: group id} for one split."""
    with open(split_csv, newline="", encoding="utf-8") as f:
        return {r["image"]: r["group"] for r in csv.DictReader(f) if r["split"] == split}


def scene_means(iou, groups, scenes=None):
    """iou: {image: IoU}. Returns {scene: mean IoU} over the given scenes (default: all)."""
    acc = defaultdict(list)
    for k, g in groups.items():
        if scenes is None or g in scenes:
            acc[g].append(iou[k])
    return {g: float(np.mean(v)) for g, v in acc.items()}


def score(iou, groups, scenes=None, unit="scene"):
    """Pipeline score: mean of scene means (round 3) or mean over images (rounds 1-2)."""
    if unit == "image":
        return float(np.mean([iou[k] for k, g in groups.items() if scenes is None or g in scenes]))
    return float(np.mean(list(scene_means(iou, groups, scenes).values())))


def compare(base, new, groups, scenes=None):
    """Scene-level comparison of two pipelines' per-image IoU. Returns the rule's numbers."""
    a, b = scene_means(base, groups, scenes), scene_means(new, groups, scenes)
    keys = sorted(a, key=int)
    d = np.array([b[g] - a[g] for g in keys])
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    out = {"scene_delta": float(d.mean()),
           "image_delta": score(new, groups, scenes, "image") - score(base, groups, scenes, "image"),
           "helped": int((d > HELP).sum()), "hurt": int((d < -HELP).sum()),
           "worst": float(d.min()), "p_gain": float((boot > 0).mean()),
           "per_scene": {g: round(float(x), 4) for g, x in zip(keys, d)}}
    out["checks"] = {"mean_up": out["scene_delta"] > 0, "more_helped": out["helped"] > out["hurt"],
                     "no_scene_broken": out["worst"] >= -WORST, "bootstrap": out["p_gain"] >= P_MIN}
    out["passed"] = all(out["checks"].values())
    return out


def forward_select(grid, options, groups, scenes=None, rule="scene"):
    """grid: {frozenset of chosen options: {image: IoU}}; options: {change: [variants]}
    (a change with several variants, e.g. ellipse prior with / without clip, adds at most one).
    rule='scene' = the round-3 rule above; rule='image' = round 1-2 rule (image mean goes up).
    Returns (chosen frozenset, [steps])."""
    chosen, steps = frozenset(), []
    while True:
        tried = []
        for change, variants in options.items():
            if any(v in chosen for v in variants):
                continue
            for v in variants:
                c = compare(grid[chosen], grid[chosen | {v}], groups, scenes)
                ok = c["passed"] if rule == "scene" else c["image_delta"] > 0
                tried.append((v, c, ok))
        passing = [t for t in tried if t[2]]
        steps.append({"chosen_before": sorted(chosen), "tried": [
            {"option": v, "passed": ok, **{k: c[k] for k in ("scene_delta", "image_delta", "helped", "hurt", "worst", "p_gain")},
             "checks": c["checks"]} for v, c, ok in tried]})
        if not passing:
            return chosen, steps
        key = (lambda t: t[1]["scene_delta"]) if rule == "scene" else (lambda t: t[1]["image_delta"])
        best = max(passing, key=key)
        steps[-1]["added"] = best[0]
        chosen = chosen | {best[0]}


def read_iou(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image"]: float(r["iou"]) for r in csv.DictReader(f)}


ROOT = Path(__file__).resolve().parents[2]
