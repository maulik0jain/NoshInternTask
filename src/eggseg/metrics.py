"""Segmentation metrics: per image, then aggregated over a split.

Pixel metrics treat all eggs in an image as one foreground mask. Instance metrics
(match at IoU >= 0.5) show whether touching eggs come out as separate objects,
which binary IoU cannot see.
"""
import math

import cv2
import numpy as np


def _safe_div(a, b):
    return a / b if b > 0 else float("nan")


def pixel_metrics(pred, gt):
    """IoU, Dice, precision, recall for boolean masks.

    Precision is NaN when the prediction is empty (0/0): it isn't "perfect", it's
    undefined. Aggregates use nanmean and report the number of empty predictions.
    """
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    union = tp + fp + fn
    return {
        "iou": _safe_div(tp, union) if union else 1.0,
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn) if union else 1.0,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
    }


def _boundary(mask):
    """1-pixel inner boundary: mask minus its 3x3 erosion."""
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def boundary_f1(pred, gt, tol=2):
    """Boundary F1 (BF score): a boundary pixel counts as matched if the other
    boundary lies within `tol` px. Rewards accurate outlines, which IoU on big
    eggs barely notices."""
    pb, gb = _boundary(pred), _boundary(gt)
    if not pb.any() or not gb.any():
        return 0.0 if (pb.any() or gb.any()) else 1.0
    # distanceTransform measures distance to the nearest ZERO pixel -> zeros on the boundary
    d_to_gt = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    d_to_pred = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    p = float((d_to_gt[pb] <= tol).mean())
    r = float((d_to_pred[gb] <= tol).mean())
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def instance_match(pred_inst, gt_inst, thr=0.5):
    """Greedy one-to-one matching of instances by IoU (largest IoU first).
    Returns (matched, n_pred, n_gt)."""
    pids = [i for i in np.unique(pred_inst) if i != 0]
    gids = [i for i in np.unique(gt_inst) if i != 0]
    pairs = []
    for g in gids:
        gm = gt_inst == g
        for p in np.unique(pred_inst[gm]):
            if p == 0:
                continue
            pm = pred_inst == p
            iou = np.count_nonzero(pm & gm) / np.count_nonzero(pm | gm)
            if iou >= thr:
                pairs.append((iou, p, g))
    used_p, used_g = set(), set()
    for _, p, g in sorted(pairs, reverse=True):
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
    return len(used_g), len(pids), len(gids)


def image_metrics(pred_mask, pred_inst, gt_mask, gt_inst, tol=2):
    m = pixel_metrics(pred_mask, gt_mask)
    m["boundary_f1"] = boundary_f1(pred_mask, gt_mask, tol)
    matched, n_pred, n_gt = instance_match(pred_inst, gt_inst)
    m.update({"n_gt": n_gt, "n_pred": n_pred, "inst_matched": matched,
              "pred_area_frac": float(pred_mask.mean()), "gt_area_frac": float(gt_mask.mean())})
    return m


def summarise(rows, worst_k=5):
    """mean / median per metric, worst-k images by IoU, instance P/R/F1 pooled over the split."""
    out = {"images": len(rows), "empty_predictions": sum(r["n_pred"] == 0 for r in rows)}
    for k in ("iou", "dice", "precision", "recall", "boundary_f1"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": round(float(np.nanmean(v)), 4), "median": round(float(np.nanmedian(v)), 4)}
    tp = sum(r["inst_matched"] for r in rows)
    npred = sum(r["n_pred"] for r in rows)
    ngt = sum(r["n_gt"] for r in rows)
    p, r = _safe_div(tp, npred), _safe_div(tp, ngt)
    out["instance"] = {"precision": round(p, 4) if not math.isnan(p) else None, "recall": round(r, 4),
                       "f1": round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0,
                       "count_exact": round(sum(r_["n_pred"] == r_["n_gt"] for r_ in rows) / len(rows), 4)}
    worst = sorted(rows, key=lambda r_: r_["iou"])[:worst_k]
    out["worst"] = [{"image": w["image"], "iou": round(w["iou"], 4)} for w in worst]
    return out
