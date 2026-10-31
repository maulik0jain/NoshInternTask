"""Task 4, reference tracks: Grounding DINO + SAM 2.1 video predictor on the egg video.

    python scripts/task4_sam_video.py --video <path to the YouTube mp4>

Same models as Task 1 (text prompt "egg." -> boxes -> SAM 2.1 Hiera Base+), but with
SAM 2.1's video predictor, which carries each egg's mask from frame to frame with a
memory of past frames, so every egg keeps one id.

1. Frames resized to 1280x720 -> data/video/frames/ (gitignored, the video isn't mine to publish)
2. The video predictor holds a chunk of frames on the GPU at a time (all 1440 frames
   don't fit in this laptop's memory). Chunks overlap by one frame: the eggs' masks
   on the last frame of a chunk are the prompts for the next one, so ids carry over.
   At each chunk start Grounding DINO is run again; an egg box no carried mask covers
   becomes a new id (an egg that enters, or one lost behind a hand). While nothing is
   tracked (title cards), the detector scans ahead every few frames for the first egg.
3. Per frame, an id map (0 = background, k = egg k) -> data/video/sam_ids/<frame>.png,
   and per-frame boxes / areas -> results/task4/sam_tracks.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task1_sam2_groundtruth import filter_boxes  # noqa: E402

WORK = (1280, 720)
VID = ROOT / "data" / "video"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=48, help="frames per SAM 2.1 chunk (2 s; GPU memory)")
    p.add_argument("--scan", type=int, default=6, help="while nothing is tracked, run the detector every this many frames")
    p.add_argument("--sam-ckpt", type=Path, default=ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt")
    p.add_argument("--sam-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--prompt", default="egg.")
    p.add_argument("--box-thr", type=float, default=0.30)   # Task 1 settings
    p.add_argument("--text-thr", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--max-box-frac", type=float, default=0.2, help="an egg never covers more of the frame than this")
    p.add_argument("--new-iou", type=float, default=0.3, help="a box overlapping a tracked egg less than this is a new egg")
    return p.parse_args()


def extract_frames(video, out):
    if out.exists() and any(out.iterdir()):
        return len(list(out.glob("*.jpg")))
    out.mkdir(parents=True, exist_ok=True)
    cap, i = cv2.VideoCapture(str(video)), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{i:05d}.jpg"), cv2.resize(f, WORK, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        i += 1
    return i


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect(proc, gdino, rgb, a, dev):
    h, w = rgb.shape[:2]
    inputs = proc(images=Image.fromarray(rgb), text=a.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        det = proc.post_process_grounded_object_detection(
            gdino(**inputs), inputs.input_ids, threshold=a.box_thr, text_threshold=a.text_thr,
            target_sizes=[(h, w)])[0]
    boxes, scores = filter_boxes(det["boxes"].cpu(), det["scores"].cpu(), a.nms_iou, a.max_box_frac * h * w)
    return [b.tolist() for b in boxes], [float(s) for s in scores]


def main():
    a = parse_args()
    from sam2.build_sam import build_sam2_video_predictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames_dir, ids_dir = VID / "frames", VID / "sam_ids"
    n = extract_frames(a.video, frames_dir)
    ids_dir.mkdir(parents=True, exist_ok=True)
    print(f"{n} frames at {WORK[0]}x{WORK[1]}", flush=True)

    proc = AutoProcessor.from_pretrained(a.gdino)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(a.gdino).to(dev).eval()
    predictor = build_sam2_video_predictor(a.sam_cfg, str(a.sam_ckpt), device=dev)

    tmp = VID / "_chunk"
    carry, next_id, tracks, detections = {}, 1, {}, {}
    written = 0  # frames 0 .. written-1 have an id map

    def write_empty(upto):
        nonlocal written
        for f in range(written, upto):
            cv2.imwrite(str(ids_dir / f"{f:05d}.png"), np.zeros(WORK[::-1], np.uint8))
            tracks[f] = {}
        written = max(written, upto)

    def rgb(f):
        return cv2.cvtColor(cv2.imread(str(frames_dir / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB)

    start = 0
    while start < n:
        if carry:
            boxes, scores = detect(proc, gdino, rgb(start), a, dev)
        else:  # nothing tracked: scan ahead with the detector until an egg shows up
            f, boxes, scores = start, [], []
            while f < n:
                boxes, scores = detect(proc, gdino, rgb(f), a, dev)
                if boxes:
                    break
                f += a.scan
            if not boxes:
                write_empty(n)
                break
            start = f
            write_empty(start)
        detections[start] = [{"box": [round(v, 1) for v in b], "score": round(s, 3)} for b, s in zip(boxes, scores)]
        end = min(start + a.chunk, n)  # frames start .. end-1; the next chunk starts at end-1
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for j, f in enumerate(range(start, end)):
            shutil.copyfile(frames_dir / f"{f:05d}.jpg", tmp / f"{j}.jpg")

        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            state = predictor.init_state(str(tmp))
            prompts = dict(carry)
            for b in boxes:
                if all(box_iou(b, box_of(pr) if isinstance(pr, np.ndarray) else pr) < a.new_iou for pr in prompts.values()):
                    prompts[next_id] = b
                    next_id += 1
            for oid, pr in prompts.items():
                if isinstance(pr, np.ndarray):
                    predictor.add_new_mask(state, frame_idx=0, obj_id=oid, mask=pr)
                else:
                    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(pr, np.float32))
            carry = {}
            for fi, oids, logits in predictor.propagate_in_video(state):
                lg = logits[:, 0].float().cpu().numpy()  # (objects, H, W)
                idmap = np.zeros(lg.shape[1:], np.uint8)
                best = lg.argmax(axis=0)
                fg = lg.max(axis=0) > 0
                for k, oid in enumerate(oids):
                    idmap[fg & (best == k)] = oid
                frame = start + fi
                if fi == end - start - 1:
                    carry = {int(oid): (idmap == oid) for oid in oids if (idmap == oid).sum() > 50}
                if frame < written:
                    continue  # the overlap frame, written by the previous chunk
                cv2.imwrite(str(ids_dir / f"{frame:05d}.png"), idmap)
                tracks[frame] = {int(oid): {"area": int((idmap == oid).sum()), "box": box_of(idmap == oid)}
                                 for oid in oids if (idmap == oid).any()}
                written = frame + 1
            predictor.reset_state(state)
        del state
        torch.cuda.empty_cache()
        print(f"  frames {start}-{end - 1}: {len(prompts)} eggs tracked, carry {sorted(carry)}", flush=True)
        if end == n:
            break
        start = end - 1
    shutil.rmtree(tmp, ignore_errors=True)
    out = ROOT / "results" / "task4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sam_tracks.json").write_text(json.dumps({
        "video": a.video.name, "frames": n, "size": WORK, "chunk": a.chunk, "detector": a.gdino, "prompt": a.prompt,
        "box_thr": a.box_thr, "ids": next_id - 1, "detections_at_chunk_start": detections,
        "tracks": {str(k): v for k, v in sorted(tracks.items())}}, indent=1), encoding="utf-8")
    print(f"{next_id - 1} egg ids over {n} frames -> {out / 'sam_tracks.json'}")


if __name__ == "__main__":
    main()
