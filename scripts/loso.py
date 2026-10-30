"""Round 3, steps 2-4: choose the changes by the scene rule, and measure how much
the choice overfits with leave-one-scene-out (LOSO). Dev only.

    python scripts/loso.py            # runs the grid once (cached), then the selection
    python scripts/loso.py --rerun    # re-run the grid

Grid: every on/off combination of the round-2 changes, each exactly as in round 2
(value fixed before round 3, no sweeps), plus the new option "ellipse prior with
clip" (GrabCut may not grow past the fitted ellipse x 1.3). 2^4 x 3 = 48 pipelines,
each run once on the 85 dev images -> results/round3/grid_dev.csv.

Selection = forward selection from the round-1 pipeline (src/eggseg/scenes.py),
with either the round-1/2 rule (image mean goes up) or the round-3 scene rule.
LOSO: for each of the 14 dev scenes, run the selection on the other 13 and score
the chosen pipeline on the held-out scene. LOSO IoU estimates how the selection
does on a scene it has never seen; in-sample minus LOSO = how much it overfits.
Writes results/round3/loso.json.
"""
import argparse
import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from eggseg import scenes  # noqa: E402
from eggseg.config import config_path, deep_merge, load_trial  # noqa: E402
from eggseg.runner import SPLIT  # noqa: E402

OUT = ROOT / "results" / "round3"
GRID = OUT / "grid_dev.csv"
BASE = "T11"
CHANGES = {  # option -> changes on top of the round-1 pipeline (values from the round-2 trials)
    "skin_off": {"skin": {"enabled": False}},                                                   # T12
    "ellipse_prior": {"grabcut": {"ellipse_prior": 1.15}},                                      # T13
    "ellipse_clip": {"grabcut": {"ellipse_prior": 1.15, "ellipse_clip": 1.3}},                  # new
    "split": {"candidate": {"split_merged": True, "split_peak_rel": 0.6, "split_min_core_px": 30}},  # T14
    "edge_rank": {"candidate": {"edge_weight": 0.5, "edge_rank_only": True, "edge_tol": 2, "canny": [30, 90]}},  # T15c
    "shape_reject": {"components": {"max_rectangularity": 0.9}, "candidate": {"max_texture": 5.0}},  # T16
}
GROUPS_OF_OPTIONS = {"skin": ["skin_off"], "ellipse": ["ellipse_prior", "ellipse_clip"], "split": ["split"],
                     "edge": ["edge_rank"], "shape": ["shape_reject"]}
# grid points that are exactly an earlier trial: must reproduce its saved per-image IoU
REPRODUCE = {"T11": [], "T12": ["skin_off"], "T13": ["skin_off", "ellipse_prior"],
             "T17": ["skin_off", "ellipse_prior", "split", "edge_rank", "shape_reject"]}
PROCEDURES = {  # name -> (rule, option groups)
    "round-2 rule, round-2 changes": ("image", {k: [o for o in v if o != "ellipse_clip"] for k, v in GROUPS_OF_OPTIONS.items()}),
    "round-3 rule, round-2 changes": ("scene", {k: [o for o in v if o != "ellipse_clip"] for k, v in GROUPS_OF_OPTIONS.items()}),
    "round-2 rule, + clip option": ("image", GROUPS_OF_OPTIONS),
    "round-3 rule, + clip option": ("scene", GROUPS_OF_OPTIONS),
}
FINAL = "round-3 rule, + clip option"


def name(opts):
    return "+".join(sorted(opts)) or "base"


def pipeline(opts):
    p = load_trial(config_path(BASE))["pipeline"]
    for o in sorted(opts):
        p = deep_merge(p, CHANGES[o])
    return p


def all_option_sets():
    for combo in itertools.product(*[[None] + v for v in GROUPS_OF_OPTIONS.values()]):
        yield frozenset(o for o in combo if o)


def _work(args):
    import cv2
    cv2.setNumThreads(1)
    from eggseg.runner import run
    opts, key = args
    rows, _ = run(pipeline(opts), [key])
    return name(opts), rows[0]


def done_configs(keys):
    """Configs already complete in the grid file (lets an interrupted run resume)."""
    if not GRID.exists():
        return set()
    seen = {}
    with open(GRID, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen.setdefault(r["config"], set()).add(r["image"])
    return {c for c, imgs in seen.items() if imgs >= set(keys)}


def run_grid(keys, workers):
    done = done_configs(keys)
    todo = [opts for opts in all_option_sets() if name(opts) not in done]
    print(f"grid: {len(todo)} of 48 pipelines to run on {len(keys)} dev images, {workers} workers", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(opts, k) for opts in todo for k in keys]
    rows, header = [], GRID.exists() and GRID.stat().st_size > 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, r in ex.map(_work, tasks, chunksize=2):  # results come back in task order
            rows.append({"config": n, **r})
            if len(rows) == len(keys):  # one config complete -> append it
                with open(GRID, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    if not header:
                        w.writeheader()
                        header = True
                    w.writerows(rows)
                print(f"  {n}: IoU {np.mean([x['iou'] for x in rows]):.3f}", flush=True)
                rows = []


def load_grid():
    grid = {}
    with open(GRID, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            grid.setdefault(r["config"], {})[r["image"]] = float(r["iou"])
    return {opts: grid[name(opts)] for opts in all_option_sets()}


def image_mean(iou, keys):
    return float(np.mean([iou[k] for k in keys]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="each worker needs ~0.3 GB RAM")
    a = ap.parse_args()
    groups = scenes.scene_groups(SPLIT, "dev")
    keys = sorted(groups)
    if a.rerun and GRID.exists():
        GRID.unlink()
    run_grid(keys, a.workers)
    grid = load_grid()

    out = {"base": BASE, "changes": CHANGES, "rule": {"help": scenes.HELP, "worst": scenes.WORST,
                                                      "n_boot": scenes.N_BOOT, "p_min": scenes.P_MIN},
           "reproduces": {}, "grid": {}, "procedures": {}}
    for trial, opts in REPRODUCE.items():
        saved = scenes.read_iou(ROOT / "experiments" / trial / "per_image.csv")
        same = all(abs(grid[frozenset(opts)][k] - saved[k]) < 1e-3 for k in keys)
        out["reproduces"][trial] = same
        assert same, f"grid point {opts} does not reproduce {trial}"
    for opts, iou in grid.items():
        out["grid"][name(opts)] = {"iou_image": round(image_mean(iou, keys), 4),
                                   "iou_scene": round(scenes.score(iou, groups), 4)}

    all_scenes = sorted(set(groups.values()), key=int)
    for pname, (rule, options) in PROCEDURES.items():
        chosen, steps = scenes.forward_select(grid, options, groups, None, rule)
        held = {}
        folds = []
        for s in all_scenes:
            train = set(all_scenes) - {s}
            c, _ = scenes.forward_select(grid, options, groups, train, rule)
            test = [k for k in keys if groups[k] == s]
            for k in test:
                held[k] = grid[c][k]
            folds.append({"held_out": s, "n": len(test), "chosen": sorted(c),
                          "iou_held_out": round(image_mean(grid[c], test), 4),
                          "iou_chosen_in_sample": round(image_mean(grid[chosen], test), 4)})
        ins_img, loso_img = image_mean(grid[chosen], keys), image_mean(held, keys)
        ins_sc, loso_sc = scenes.score(grid[chosen], groups), scenes.score(held, groups)
        stab = {o: sum(o in f["chosen"] for f in folds) for v in options.values() for o in v}
        out["procedures"][pname] = {
            "rule": rule, "chosen": sorted(chosen), "steps": steps,
            "in_sample": {"iou_image": round(ins_img, 4), "iou_scene": round(ins_sc, 4)},
            "loso": {"iou_image": round(loso_img, 4), "iou_scene": round(loso_sc, 4)},
            "optimism_image": round(ins_img - loso_img, 4), "optimism_scene": round(ins_sc - loso_sc, 4),
            "folds": folds, "stability": stab, "n_folds": len(folds),
            "distinct_choices": len({tuple(f["chosen"]) for f in folds})}
        print(f"{pname:>30}: chosen {name(chosen):<45} in-sample {ins_img:.3f}  LOSO {loso_img:.3f}  "
              f"optimism {ins_img - loso_img:+.3f}  (scene-mean {ins_sc:.3f} / {loso_sc:.3f})  "
              f"{out['procedures'][pname]['distinct_choices']} distinct choices over {len(folds)} folds")
    out["final"] = {"procedure": FINAL, "chosen": out["procedures"][FINAL]["chosen"]}
    (OUT / "loso.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("final round-3 choice:", out["final"]["chosen"])


if __name__ == "__main__":
    main()
