"""Task 4 analysis: classical tracks vs the SAM 2.1 tracks, and the annotated videos.

    python scripts/task4_analyze.py --trial T21 --video <mp4>

SAM 2.1's tracks are the reference here (pseudo-GT, as in Task 1). Per frame, the
classical eggs are matched to the SAM eggs (Hungarian on mask IoU, a match needs
IoU >= 0.5), then the usual multi-object tracking counts:
  MOTA = 1 - (missed + false eggs + id switches) / reference eggs
  IDF1 = how well one classical id stays on one SAM egg over the whole video
  id switch = a SAM egg matched to a different classical id than the last time
Writes results/task4/metrics_<trial>.json, timeline_<trial>.png, and three videos
in results/task4/videos/: SAM, classical, and both side by side.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402

from analyze import GRID, SURFACE, style  # noqa: E402
from task4_track import MIN_AREA  # noqa: E402

VID = ROOT / "data" / "video"
OUT = ROOT / "results" / "task4"
MATCH = 0.5
FPS = 24
# one colour per egg id (BGR), fixed order so an id keeps its colour in every video
COLORS = [(214, 120, 42), (52, 104, 235), (60, 170, 60), (180, 80, 200), (40, 190, 220), (120, 120, 120)]
SAM_C, CL_C = "#7a7974", "#2a78d6"


def ids_of(m):
    return [int(k) for k in np.unique(m) if k > 0]


def match(sam, cl):
    """Hungarian on IoU between the eggs of two id maps. Returns [(sam id, classical id, IoU)]."""
    a, b = ids_of(sam), ids_of(cl)
    if not a or not b:
        return []
    m = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        sx = sam == x
        for j, y in enumerate(b):
            cy = cl == y
            inter = np.logical_and(sx, cy).sum()
            m[i, j] = inter / (np.logical_or(sx, cy).sum() + 1e-9) if inter else 0.0
    r, c = linear_sum_assignment(-m)
    return [(a[i], b[j], float(m[i, j])) for i, j in zip(r, c) if m[i, j] >= MATCH]


MISS_WHY = ("merged with the other egg", "mask too big (spills onto table or reflection)", "mask too small (part of the egg)",
            "mask off target", "segmented, but not kept by the tracker", "not segmented")
FALSE_WHY = ("wrong shape of a real egg", "below an egg (reflection position)", "elsewhere")


def errors(sam, cl, det, ms):
    """One cause per missed SAM egg and per false classical egg, from the masks of one frame."""
    a, b = ids_of(sam), ids_of(cl)
    hit_s, hit_c = {x for x, *_ in ms}, {y for _, y, _ in ms}
    S = {x: sam == x for x in a}
    C = {y: cl == y for y in b}
    D = [det == k for k in ids_of(det)]
    miss, false = [], []
    for x in a:
        if x in hit_s:
            continue
        sx, area = S[x], S[x].sum()
        cover = {y: np.logical_and(sx, C[y]).sum() for y in b}
        y = max(cover, key=cover.get) if cover else None
        if y is None or cover[y] == 0:
            found = any(np.logical_and(sx, d).sum() / (np.logical_or(sx, d).sum() + 1e-9) >= MATCH for d in D)
            miss.append(MISS_WHY[4] if found else MISS_WHY[5])
        elif any(np.logical_and(C[y], S[z]).sum() >= 0.3 * S[z].sum() for z in a if z != x) and cover[y] >= 0.3 * area:
            miss.append(MISS_WHY[0])
        elif C[y].sum() >= 1.5 * area:
            miss.append(MISS_WHY[1])
        elif C[y].sum() < area:
            miss.append(MISS_WHY[2])
        else:
            miss.append(MISS_WHY[3])
    for y in b:
        if y in hit_c:
            continue
        cy = C[y]
        if any(np.logical_and(cy, S[x]).sum() >= 0.1 * cy.sum() for x in a):
            false.append(FALSE_WHY[0])
            continue
        ys, xs = np.nonzero(cy)
        below = False
        for x in a:
            sy, sxs = np.nonzero(S[x])
            h = sy.max() - sy.min() + 1
            below |= sxs.min() <= xs.mean() <= sxs.max() and sy.max() < ys.mean() <= sy.max() + 1.5 * h
        false.append(FALSE_WHY[1] if below else FALSE_WHY[2])
    return miss, false


def draw(frame, idmap, title, color_of):
    out = frame.copy()
    over = frame.copy()
    for k in ids_of(idmap):
        m = idmap == k
        col = color_of(k)
        over[m] = col
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 2)
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
        cv2.putText(out, f"egg {k}", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"egg {k}", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    out = cv2.addWeighted(over, 0.4, out, 0.6, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    frames = sorted((VID / "frames").glob("*.jpg"))
    sam_dir, cl_dir, det_dir = VID / "sam_ids", VID / f"classical_ids_{a.trial}", VID / f"classical_det_{a.trial}"
    rows, pairs = [], {}
    why_miss, why_false = {k: [] for k in MISS_WHY}, {k: [] for k in FALSE_WHY}
    gt_total = pred_total = fn = fp = sw = 0
    last = {}
    for f in frames:
        s = cv2.imread(str(sam_dir / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(cl_dir / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
        ms = match(s, c)
        ns, nc = len(ids_of(s)), len(ids_of(c))
        if ns - len(ms) or nc - len(ms):
            d = cv2.imread(str(det_dir / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
            d = np.where(np.isin(d, [k for k in ids_of(d) if (d == k).sum() >= MIN_AREA]), d, 0)
            em, ef = errors(s, c, d, ms)
            for k in em:
                why_miss[k].append(int(f.stem))
            for k in ef:
                why_false[k].append(int(f.stem))
        gt_total, pred_total = gt_total + ns, pred_total + nc
        fn, fp = fn + ns - len(ms), fp + nc - len(ms)
        for si, ci, _ in ms:
            if si in last and last[si] != ci:
                sw += 1
            last[si] = ci
            pairs[(si, ci)] = pairs.get((si, ci), 0) + 1
        sb, cb = s > 0, c > 0
        union = np.logical_or(sb, cb).sum()
        rows.append({"frame": int(f.stem), "sam": ns, "classical": nc, "matched": len(ms),
                     "iou": float(np.logical_and(sb, cb).sum() / union) if union else None,
                     "matched_iou": float(np.mean([x for *_, x in ms])) if ms else None})
    # IDF1: best one-to-one assignment of classical ids to SAM ids over the whole video
    sids, cids = sorted({p[0] for p in pairs}), sorted({p[1] for p in pairs})
    idtp = 0
    if pairs:
        m = np.array([[pairs.get((x, y), 0) for y in cids] for x in sids])
        r, cc = linear_sum_assignment(-m)
        idtp = int(m[r, cc].sum())
    egg_frames = [r for r in rows if r["sam"] > 0]
    out = {"trial": a.trial, "frames": len(rows), "frames_with_eggs": len(egg_frames),
           "egg_scene": [egg_frames[0]["frame"], egg_frames[-1]["frame"]] if egg_frames else None,
           "fps": FPS,
           "reference_eggs": gt_total, "classical_eggs": pred_total, "missed": fn, "false": fp, "id_switches": sw,
           "mota": round(1 - (fn + fp + sw) / max(gt_total, 1), 4),
           "idf1": round(2 * idtp / max(gt_total + pred_total, 1), 4),
           "recall": round((gt_total - fn) / max(gt_total, 1), 4),
           "precision": round((pred_total - fp) / max(pred_total, 1), 4),
           "mean_iou_egg_frames": round(float(np.mean([r["iou"] or 0 for r in egg_frames])), 4) if egg_frames else None,
           "mean_matched_iou": round(float(np.mean([r["matched_iou"] for r in rows if r["matched_iou"]])), 4),
           "count_exact_egg_frames": round(float(np.mean([r["sam"] == r["classical"] for r in egg_frames])), 4),
           "false_eggs_no_egg_frames": sum(r["classical"] for r in rows if r["sam"] == 0),
           "sam_ids": len(sids), "classical_ids_used": len({p[1] for p in pairs}),
           "id_pairs": {f"sam {x} - classical {y}": n for (x, y), n in sorted(pairs.items())}}
    def spans(fr):  # frame list -> [[start s, end s], ...], gaps under 1 s joined
        out_s = []
        for x in fr:
            if out_s and x - out_s[-1][1] <= FPS:
                out_s[-1][1] = x
            else:
                out_s.append([x, x])
        return [[round(p / FPS, 1), round(q / FPS, 1)] for p, q in out_s]
    out["errors"] = {
        "missed": {k: {"n": len(v), "share": round(len(v) / max(fn, 1), 4), "when_s": spans(v)} for k, v in why_miss.items()},
        "false": {k: {"n": len(v), "share": round(len(v) / max(fp, 1), 4), "when_s": spans(v)} for k, v in why_false.items()}}
    trk = json.loads((OUT / f"classical_tracks_{a.trial}.json").read_text(encoding="utf-8"))
    out["classical_ids_total"] = trk["ids"]
    (OUT / f"metrics_{a.trial}.json").write_text(json.dumps({**out, "per_frame": rows}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "id_pairs"}, indent=1))

    # timeline
    style()
    t = np.array([r["frame"] for r in rows]) / FPS
    fig, ax = plt.subplots(2, 1, figsize=(9.6, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    ax[0].step(t, [r["sam"] for r in rows], where="mid", color=SAM_C, linewidth=2, label="SAM 2.1 (reference)")
    ax[0].step(t, [r["classical"] for r in rows], where="mid", color=CL_C, linewidth=1.4, label=f"classical {a.trial} + tracker")
    ax[0].set_ylabel("eggs in frame")
    ax[0].set_yticks(range(0, max(max(r["sam"] for r in rows), max(r["classical"] for r in rows)) + 2))
    ax[0].legend(loc="upper right", frameon=False, fontsize=8)
    ax[1].plot(t, [r["iou"] if r["iou"] is not None else np.nan for r in rows], color=CL_C, linewidth=1.2)
    ax[1].set_ylabel("IoU vs SAM (all eggs)")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("time (s)")
    for x in ax:
        x.grid(color=GRID, linewidth=0.8)
    ax[0].set_title(f"Task 4 video: classical {a.trial} + tracker vs SAM 2.1, every frame", fontsize=11,
                    fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"timeline_{a.trial}.png", dpi=150, facecolor=SURFACE)

    color = lambda k: COLORS[(k - 1) % len(COLORS)]  # noqa: E731
    # classical ids get the colour of the SAM egg they follow most, so the two are easy to compare
    follow = {}
    for (x, y), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        follow.setdefault(y, x)

    # failure sheet: the worst egg frames (lowest IoU vs SAM), at least 1 s apart
    worst, taken = [], []
    for r in sorted(egg_frames, key=lambda r: r["iou"] or 0):
        if all(abs(r["frame"] - t) >= FPS for t in taken):
            worst.append(r)
            taken.append(r["frame"])
        if len(worst) == 6:
            break
    rowsimg = []
    for r in worst:
        f = VID / "frames" / f"{r['frame']:05d}.jpg"
        img = cv2.imread(str(f))
        s = cv2.imread(str(sam_dir / f"{r['frame']:05d}.png"), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(cl_dir / f"{r['frame']:05d}.png"), cv2.IMREAD_UNCHANGED)
        tiles = [draw(img, s, f"SAM  t={r['frame'] / FPS:.1f}s", color),
                 draw(img, c, f"classical  IoU {r['iou'] or 0:.2f}", lambda k: color(follow.get(k, k)))]
        rowsimg.append(np.hstack([cv2.resize(x, (640, 360)) for x in tiles]))
    out["worst_frames"] = [{"frame": r["frame"], "t": round(r["frame"] / FPS, 1), "iou": round(r["iou"] or 0, 3),
                            "sam": r["sam"], "classical": r["classical"]} for r in worst]
    cv2.imwrite(str(OUT / f"worst_frames_{a.trial}.jpg"), np.vstack(rowsimg), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (OUT / f"metrics_{a.trial}.json").write_text(json.dumps({**out, "per_frame": rows}, indent=1), encoding="utf-8")
    if a.no_video:
        return

    # videos: H.264 (OpenCV's FFmpeg), 1280x720, plus a side-by-side at 1280x360
    vd = OUT / "videos"
    vd.mkdir(parents=True, exist_ok=True)
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    ws = {"sam": cv2.VideoWriter(str(vd / "sam2_tracking.mp4"), fourcc, FPS, (w, h)),
          "cl": cv2.VideoWriter(str(vd / f"classical_{a.trial}_tracking.mp4"), fourcc, FPS, (w, h)),
          "both": cv2.VideoWriter(str(vd / f"side_by_side_{a.trial}.mp4"), fourcc, FPS, (w, h // 2))}
    for f, r in zip(frames, rows):
        img = cv2.imread(str(f))
        s = cv2.imread(str(sam_dir / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(cl_dir / (f.stem + ".png")), cv2.IMREAD_UNCHANGED)
        a1 = draw(img, s, f"SAM 2.1 video tracking (reference)   t={r['frame'] / FPS:5.1f}s", color)
        a2 = draw(img, c, f"Classical {a.trial} + tracker   t={r['frame'] / FPS:5.1f}s",
                  lambda k: color(follow.get(k, k)))
        ws["sam"].write(a1)
        ws["cl"].write(a2)
        ws["both"].write(np.hstack([cv2.resize(a1, (w // 2, h // 2)), cv2.resize(a2, (w // 2, h // 2))]))
    for v in ws.values():
        v.release()
    print("videos ->", vd)


if __name__ == "__main__":
    main()
