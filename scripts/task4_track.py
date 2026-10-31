"""Task 4: track the eggs in the video with the classical pipeline + a tracker of my own.

    python scripts/task4_track.py --trial T21        # frames from task4_sam_video.py

1. Detection, per frame: the classical pipeline (segment(), same code as Tasks 2-3,
   frames at 1280x720) -> egg instances. Cached in data/video/classical_det/.
2. Tracking (no learning, no appearance model; two white eggs look the same anyway):
   - predict: a track moves with its smoothed velocity
   - match: Hungarian assignment; cost = 1 - IoU(track's last mask shifted by its
     velocity, detection). No overlap but the centroid within GATE egg sizes -> a weak
     match (cost 1 + distance / gate), so a fast spinning egg is still followed
   - a new detection is tentative and gets an id only after CONFIRM frames in a row:
     one-frame false eggs (white text labels, a knuckle) never get an id
   - an unmatched track coasts for up to MAX_MISS frames (an egg under a hand) and,
     if a detection reappears near it, keeps its id
3. data/video/classical_ids/<frame>.png (id maps, confirmed tracks only) and
   results/task4/classical_tracks.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VID = ROOT / "data" / "video"
CONFIRM = 3      # frames in a row before a detection becomes an egg
MAX_MISS = 48    # 2 s at 24 fps: how long a covered egg keeps its id
GATE = 1.5       # weak match if the centroid is within this many egg diameters (+ speed x frames missed)
ALPHA = 0.5      # velocity smoothing
MIN_AREA = 300   # px at 1280x720: smaller blobs are not eggs


def _detect(args):
    cv2.setNumThreads(1)
    from eggseg.classical import segment
    frame_path, out_path, pipeline = args
    rgb = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    _, inst = segment(rgb, pipeline)
    cv2.imwrite(str(out_path), inst.astype(np.uint8))
    return out_path.name


def detect_all(pipeline, frames, out, workers):
    out.mkdir(parents=True, exist_ok=True)
    todo = [(f, out / (f.stem + ".png"), pipeline) for f in frames if not (out / (f.stem + ".png")).exists()]
    print(f"detecting on {len(todo)} of {len(frames)} frames, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(_detect, todo, chunksize=2), 1):
            if i % 120 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def shifted(m, d):
    return cv2.warpAffine(m.astype(np.uint8), np.float32([[1, 0, d[0]], [0, 1, d[1]]]), m.shape[::-1]) > 0


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / (np.logical_or(a, b).sum() + 1e-9)


class Tracker:
    def __init__(self):
        self.tracks, self.next_tid, self.next_id = [], 0, 1

    def step(self, dets):
        """dets: list of bool masks. Returns {egg id: mask} for confirmed tracks seen this frame."""
        cost = np.full((len(self.tracks), len(dets)), 1e3)
        cds = [centroid(d) for d in dets]
        sizes = [2 * np.sqrt(d.sum() / np.pi) for d in dets]
        for i, t in enumerate(self.tracks):
            pred = shifted(t["mask"], t["vel"] * (t["miss"] + 1))
            pc = t["c"] + t["vel"] * (t["miss"] + 1)
            for j, d in enumerate(dets):
                ov = iou(pred, d)
                gate = GATE * max(t["size"], sizes[j]) + np.linalg.norm(t["vel"]) * t["miss"]
                dist = np.linalg.norm(pc - cds[j])
                if ov > 0.05:
                    cost[i, j] = 1 - ov
                elif dist < gate:
                    cost[i, j] = 1 + dist / gate
        rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
        matched_t, matched_d = set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= 2:
                continue
            t = self.tracks[i]
            c = cds[j]
            t["vel"] = ALPHA * (c - t["c"]) / (t["miss"] + 1) + (1 - ALPHA) * t["vel"]
            t.update(mask=dets[j], c=c, size=sizes[j], miss=0, run=t["run"] + 1)
            if t["id"] is None and t["run"] >= CONFIRM:
                t["id"], self.next_id = self.next_id, self.next_id + 1
            matched_t.add(i)
            matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t["miss"] += 1
                t["run"] = 0
        # tentative tracks die at the first miss; confirmed ones coast up to MAX_MISS frames
        self.tracks = [t for t in self.tracks if (t["miss"] == 0 or (t["id"] is not None and t["miss"] <= MAX_MISS))]
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append({"tid": self.next_tid, "id": None, "mask": d, "c": cds[j], "size": sizes[j],
                                    "vel": np.zeros(2), "miss": 0, "run": 1})
                self.next_tid += 1
        return {t["id"]: t["mask"] for t in self.tracks if t["id"] is not None and t["miss"] == 0}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="classical pipeline to run (experiment id)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from eggseg.config import config_path, load_trial
    pipeline = load_trial(config_path(a.trial))["pipeline"]
    frames = sorted((VID / "frames").glob("*.jpg"))
    det_dir, ids_dir = VID / f"classical_det_{a.trial}", VID / f"classical_ids_{a.trial}"
    detect_all(pipeline, frames, det_dir, a.workers)

    ids_dir.mkdir(parents=True, exist_ok=True)
    tr, tracks, n_dets = Tracker(), {}, {}
    for f in frames:
        inst = cv2.imread(str(det_dir / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
        dets = [inst == k for k in np.unique(inst) if k > 0 and (inst == k).sum() >= MIN_AREA]
        out = tr.step(dets)
        idmap = np.zeros(inst.shape, np.uint8)
        for eid, m in out.items():
            idmap[m] = eid
        cv2.imwrite(str(ids_dir / (f.stem + ".png")), idmap)
        fi = int(f.stem)
        n_dets[fi] = len(dets)
        tracks[fi] = {eid: {"area": int(m.sum()), "box": [int(v) for v in cv2.boundingRect(m.astype(np.uint8))]}
                      for eid, m in out.items()}
    res = ROOT / "results" / "task4"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"classical_tracks_{a.trial}.json").write_text(json.dumps({
        "trial": a.trial, "frames": len(frames), "ids": tr.next_id - 1,
        "tracker": {"confirm": CONFIRM, "max_miss": MAX_MISS, "gate": GATE, "alpha": ALPHA, "min_area": MIN_AREA},
        "detections_per_frame": {str(k): v for k, v in n_dets.items()},
        "tracks": {str(k): {str(i): v for i, v in t.items()} for k, t in tracks.items()}}, indent=1), encoding="utf-8")
    print(f"{tr.next_id - 1} egg ids over {len(frames)} frames ({a.trial})")


if __name__ == "__main__":
    main()
