"""The video as a fresh test set: rounds 1-3 on a scene none of them has seen.

    python scripts/task4_holdout.py

Run after task4_sam_video.py. Nothing in rounds 1-3 was decided with this video in
view, so it is the one clean test left after eval was looked at more than once.
Every STEP-th frame that has a SAM egg in it: segmentation only (no tracker), binary
IoU against SAM's mask. Frames of one shot are not independent (same table, same
eggs, same light), so the interval comes from a block bootstrap over 2 s blocks, and
the whole thing counts as one more scene, not hundreds of new images.
Added after the round numbers were seen (diagnosis only, nothing is chosen from it):
the same score along round 3's selection path, one change per step, and how much of
the eggs / the table under them / the rest falls inside round 1's skin colour box.
Writes results/task4/holdout.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from task4_track import VID, detect_all  # noqa: E402

STEP, BLOCK, N_BOOT, SEED = 6, 48, 2000, 0
ROUNDS = {1: "configs/final.yaml", 2: "configs/final_round2.yaml", 3: "configs/final_round3.yaml"}
PATH = ["T11", "T19", "T20", "T21"]  # round 3's forward selection, one change per step


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import yaml
    from eggseg.config import config_path, load_trial
    frames = sorted((VID / "frames").glob("*.jpg"))
    sam = {f.stem: cv2.imread(str(VID / "sam_ids" / (f.stem + ".png")), cv2.IMREAD_UNCHANGED) for f in frames[::STEP]}
    test = [f for f in frames[::STEP] if (sam[f.stem] > 0).any()]
    out = {"step": STEP, "frames": len(test), "block_frames": BLOCK, "rounds": {}}
    rng = np.random.default_rng(SEED)
    block_of = np.array([int(f.stem) // BLOCK for f in test])
    ids = sorted(set(block_of))

    def block_boot(v):  # resample whole 2 s blocks, not frames
        parts = [v[block_of == b] for b in ids]
        return [np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))]).mean() for _ in range(N_BOOT)]

    def video_iou(trial):
        det = VID / f"classical_det_{trial}"
        detect_all(load_trial(config_path(trial))["pipeline"], test, det, 6)
        ious = []
        for f in test:
            p = cv2.imread(str(det / (f.stem + ".png")), cv2.IMREAD_UNCHANGED) > 0
            g = sam[f.stem] > 0
            ious.append(float(np.logical_and(p, g).sum() / np.logical_or(p, g).sum()))
        return np.array(ious)

    def ci(boot):
        return [round(float(np.percentile(boot, 5)), 4), round(float(np.percentile(boot, 95)), 4)]

    per = {}
    for r, cfg in ROUNDS.items():
        trial = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))["trial"]
        ious = per[r] = video_iou(trial)
        boot = block_boot(ious)
        out["rounds"][str(r)] = {"trial": trial, "iou_mean": round(float(ious.mean()), 4),
                                 "iou_median": round(float(np.median(ious)), 4),
                                 "ci90": [round(float(np.percentile(boot, 5)), 4), round(float(np.percentile(boot, 95)), 4)],
                                 "frames_below_half": int((ious < 0.5).sum())}
        print(f"round {r} ({trial}): IoU {ious.mean():.3f} (median {np.median(ious):.3f}) on {len(test)} frames")
    out["deltas"] = {}
    for a, b in ((1, 3), (2, 3), (1, 2)):  # paired: same frames
        d = per[b] - per[a]
        boot = block_boot(d)
        out["deltas"][f"{b}_vs_{a}"] = {"mean": round(float(d.mean()), 4),
                                        "ci90": [round(float(np.percentile(boot, 5)), 4),
                                                 round(float(np.percentile(boot, 95)), 4)]}
    print(json.dumps(out["deltas"]))
    # diagnosis, after the rounds above were seen: which round-3 step costs on the video.
    # Only reported; no pipeline is chosen from it.
    out["round3_path"], prev = [], None
    for trial in PATH:
        v = video_iou(trial)
        row = {"trial": trial, "iou_mean": round(float(v.mean()), 4)}
        if prev is not None:
            d = v - prev
            row["delta"] = round(float(d.mean()), 4)
            row["delta_ci90"] = ci(block_boot(d))
        out["round3_path"].append(row)
        prev = v
        print(row)
    # what the skin box (round 1's colour filter) covers in this video, on the same frames
    from eggseg.classical import preprocess, skin
    p1 = load_trial(config_path(PATH[0]))["pipeline"]
    share = {"egg": [], "below_egg": [], "rest": []}
    for f in test:
        img = preprocess(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), p1["preprocess"])
        sk = skin(img, p1["skin"]["cb"], p1["skin"]["cr"])
        s = sam[f.stem]
        egg, below = s > 0, np.zeros(s.shape, bool)
        for k in np.unique(s)[1:]:  # one egg height under each egg: where its reflection is
            ys, xs = np.nonzero(s == k)
            below[ys.max() + 1: ys.max() + 1 + (ys.max() - ys.min()), xs.min(): xs.max() + 1] = True
        below &= ~egg
        for k, m in (("egg", egg), ("below_egg", below), ("rest", ~egg & ~below)):
            share[k].append(float(sk[m].mean()))
    out["skin_box_share"] = {k: round(float(np.mean(v)), 4) for k, v in share.items()}
    print("skin box share:", out["skin_box_share"])
    (ROOT / "results" / "task4" / "holdout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
