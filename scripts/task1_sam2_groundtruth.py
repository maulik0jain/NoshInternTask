"""Task 1 - ground-truth egg masks with Grounding DINO (boxes) + SAM 2.1 (masks).

The Roboflow export is folder-format (classification only), so there are no
boxes to prompt SAM with. Grounding DINO turns the text prompt "egg" into
boxes; SAM 2.1 turns each box into a mask.

Outputs (under --out):
  masks/<rel>.png      binary mask, 0/255 (union of all eggs)
  instances/<rel>.png  uint8 instance ids, 0 = background, 1..N = eggs
  overlays/<rel>.jpg   coloured overlay + boxes, for visual review
  annotations.json     per-image boxes, detector/SAM scores, mask areas
  review_sheet.jpg     contact sheet of all overlays
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou, nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parents[1]
COLORS = [(255, 40, 40), (40, 200, 40), (40, 120, 255), (255, 200, 0), (200, 0, 255), (0, 220, 220)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data" / "egg_dataset")
    p.add_argument("--out", type=Path, default=ROOT / "ground_truth")
    p.add_argument("--sam-ckpt", type=Path, default=ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt")
    p.add_argument("--sam-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--prompt", default="egg.")  # "an egg." scored true eggs ~0.29 (missed); "egg." ~0.6
    p.add_argument("--box-thr", type=float, default=0.30)
    p.add_argument("--text-thr", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--overrides", type=Path, default=ROOT / "ground_truth_overrides.json",
                   help="human-review fixes: per image, detector boxes to drop (matched at IoU >= 0.7)")
    p.add_argument("--dup-frac", type=float, default=0.5, help="drop a mask this much covered by a more confident egg")
    p.add_argument("--max-box-frac", type=float, default=0.95, help="drop boxes covering more of the image than this")
    return p.parse_args()


def filter_boxes(boxes, scores, nms_iou, max_area):
    """NMS, drop near-full-image boxes, drop 'group' boxes that enclose >=2 other boxes."""
    if len(boxes) == 0:
        return boxes, scores
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = area <= max_area
    boxes, scores, area = boxes[keep], scores[keep], area[keep]
    keep = nms(boxes, scores, nms_iou)
    boxes, scores, area = boxes[keep], scores[keep], area[keep]

    # Grounding DINO often adds one box around a cluster of eggs.
    group = torch.zeros(len(boxes), dtype=torch.bool)
    for i in range(len(boxes)):
        x0 = torch.maximum(boxes[:, 0], boxes[i, 0])
        y0 = torch.maximum(boxes[:, 1], boxes[i, 1])
        x1 = torch.minimum(boxes[:, 2], boxes[i, 2])
        y1 = torch.minimum(boxes[:, 3], boxes[i, 3])
        inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
        inside = (inter / area > 0.85) & (area < 0.7 * area[i])
        inside[i] = False
        group[i] = inside.sum() >= 2
    return boxes[~group], scores[~group]


def clean_mask(m, box, pad=6, min_frac=0.15):
    """Clip to a slightly padded box, fill holes, drop specks.

    Keeps every component >= min_frac of the largest: a stem or fridge-tray bar
    across an egg splits it into pieces that are all visible egg.
    """
    h, w = m.shape
    x0, y0, x1, y1 = box
    clip = np.zeros_like(m)
    clip[max(0, int(y0) - pad):min(h, int(y1) + pad), max(0, int(x0) - pad):min(w, int(x1) + pad)] = True
    cnts, _ = cv2.findContours((m & clip).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros((h, w), np.uint8)
    if cnts:
        biggest = max(cv2.contourArea(c) for c in cnts)
        keep = [c for c in cnts if cv2.contourArea(c) >= min_frac * biggest]
        cv2.drawContours(out, keep, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def overlay(img, inst, boxes):
    vis = img.copy()
    for k in range(1, inst.max() + 1):
        c = np.array(COLORS[(k - 1) % len(COLORS)], dtype=np.float32)
        sel = inst == k
        vis[sel] = (0.45 * vis[sel] + 0.55 * c).astype(np.uint8)
        cnts, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, tuple(int(v) for v in c), 2)
    for b in boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
    return vis


def review_sheet(paths, out_path, tile=160, cols=10):
    rows = (len(paths) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, p in enumerate(paths):
        im = cv2.resize(cv2.imread(str(p)), (tile, tile))
        cv2.putText(im, str(i), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = im
    cv2.imwrite(str(out_path), sheet)


@torch.inference_mode()
def main():
    a = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    proc = AutoProcessor.from_pretrained(a.gdino)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(a.gdino).to(dev).eval()
    sam = SAM2ImagePredictor(build_sam2(a.sam_cfg, str(a.sam_ckpt), device=dev))

    overrides = json.loads(a.overrides.read_text()) if a.overrides.exists() else {}
    images = sorted(a.data.rglob("*.jpg"))
    records, overlay_paths = [], []
    for ip in images:
        rel = ip.relative_to(a.data)  # names hold extra dots (x_jpg.rf.<hash>.jpg): never use with_suffix
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]

        inputs = proc(images=Image.fromarray(img), text=a.prompt, return_tensors="pt").to(dev)
        det = proc.post_process_grounded_object_detection(
            gdino(**inputs), inputs.input_ids,
            threshold=a.box_thr, text_threshold=a.text_thr, target_sizes=[(h, w)],
        )[0]
        boxes, dscores = filter_boxes(det["boxes"].cpu(), det["scores"].cpu(), a.nms_iou, a.max_box_frac * h * w)
        review = overrides.get(rel.as_posix(), {})
        if review.get("drop_boxes") and len(boxes):
            hit = box_iou(boxes, torch.tensor(review["drop_boxes"], dtype=boxes.dtype)).max(dim=1).values >= 0.7
            boxes, dscores = boxes[~hit], dscores[~hit]

        inst = np.zeros((h, w), np.uint8)
        eggs = []
        if len(boxes):
            sam.set_image(img)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                masks, sscores, _ = sam.predict(box=boxes.numpy(), multimask_output=False)
            masks = masks.reshape(len(boxes), h, w) > 0
            sscores = np.asarray(sscores).reshape(-1)
            cleaned = [clean_mask(masks[j], boxes[j].tolist()) for j in range(len(boxes))]

            # Mask-level duplicate suppression, most confident detection first:
            # a mask mostly covered by an accepted egg is a duplicate/part box
            # (box NMS misses these: their boxes overlap only ~25%).
            kept, taken = [], np.zeros((h, w), bool)
            for j in sorted(range(len(boxes)), key=lambda j: -float(dscores[j])):
                area = cleaned[j].sum()
                if area == 0 or (cleaned[j] & taken).sum() > a.dup_frac * area:
                    continue
                kept.append(j)
                taken |= cleaned[j]
            boxes, dscores, sscores = boxes[kept], dscores[kept], sscores[kept]
            cleaned = [cleaned[j] for j in kept]

            # Ids run 1..N left to right; paint lowest SAM score first so
            # overlapping pixels end up with the most confident egg.
            ids = {j: k for k, j in enumerate(sorted(range(len(boxes)), key=lambda j: float(boxes[j][0])), start=1)}
            for j in sorted(range(len(boxes)), key=lambda j: sscores[j]):
                inst[cleaned[j]] = ids[j]
            for j, k in sorted(ids.items(), key=lambda t: t[1]):
                eggs.append({
                    "id": k,
                    "box_xyxy": [round(float(v), 1) for v in boxes[j]],
                    "det_score": round(float(dscores[j]), 3),
                    "sam_score": round(float(sscores[j]), 3),
                    "area_px": int((inst == k).sum()),
                })

        for sub, arr, ext in (("masks", (inst > 0).astype(np.uint8) * 255, ".png"), ("instances", inst, ".png")):
            p = a.out / sub / rel.with_name(rel.stem + ext)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), arr)
        op = a.out / "overlays" / rel.with_name(rel.stem + ".jpg")
        op.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(op), cv2.cvtColor(overlay(img, inst, boxes.tolist()), cv2.COLOR_RGB2BGR))
        overlay_paths.append(op)

        records.append({"image": rel.as_posix(), "width": w, "height": h,
                        "num_eggs": len(eggs), "eggs": eggs})
        print(f"{rel.as_posix():90s} eggs={len(eggs)}")

    meta = {"detector": a.gdino, "prompt": a.prompt, "box_thr": a.box_thr, "text_thr": a.text_thr,
            "segmenter": a.sam_ckpt.name, "sam_cfg": a.sam_cfg, "overrides": overrides, "images": records}
    (a.out / "annotations.json").write_text(json.dumps(meta, indent=2))
    review_sheet(overlay_paths, a.out / "review_sheet.jpg")
    print(f"\n{len(records)} images, {sum(r['num_eggs'] for r in records)} eggs -> {a.out}")


if __name__ == "__main__":
    main()
