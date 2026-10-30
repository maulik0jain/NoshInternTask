"""Round 1 vs round 2, from the two saved final-eval runs (nothing is re-run).

    python scripts/compare_rounds.py

Reads results/final_eval (round 1), results/final_eval_round2 and the round-1 failure
labels; writes results/round2/:
  compare.json                 metrics of both rounds, per split; IoU change per round-1 failure cause
  iou_round1_vs_round2.png     per-image IoU, round 1 (x) vs round 2 (y)
  iou_by_cause.png             mean IoU of the round-1 failures, grouped by their cause, both rounds
  changes.jpg                  biggest gains and losses: image | GT | round 1 | round 2 (error maps)
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import read_rows, style, SPLIT_COLOR, INK2, SURFACE  # noqa: E402
from eggseg.data import load_gt, load_rgb  # noqa: E402
from eggseg.runner import DATA, GT  # noqa: E402
from eggseg.viz import error_map, overlay, save_rgb  # noqa: E402

R1, R2 = ROOT / "results" / "final_eval", ROOT / "results" / "final_eval_round2"
LABELS = ROOT / "results" / "analysis" / "failure_labels.csv"
OUT = ROOT / "results" / "round2"
CHANGE = 0.1  # an image counts as better / worse if its IoU moved by more than this
BEFORE, AFTER = "#b9b8b3", "#2a78d6"  # round 1 muted, round 2 highlighted


def pred(run, key):
    p = run / "preds" / Path(key).parent / (Path(key).stem + ".png")
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0


def headline(summary, rows):
    iou = np.array([r["iou"] for r in rows])
    return {"iou_mean": summary["iou"]["mean"], "iou_median": summary["iou"]["median"],
            "dice": summary["dice"]["mean"], "precision": summary["precision"]["mean"],
            "recall": summary["recall"]["mean"], "boundary_f1": summary["boundary_f1"]["mean"],
            "instance_f1": summary["instance"]["f1"], "count_exact": summary["instance"]["count_exact"],
            "empty": summary["empty_predictions"], "n_good": int((iou >= 0.8).sum()), "n_zero": int((iou == 0).sum()),
            "images": summary["images"]}


def tile(img, text, size=240):
    t = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (size, 22), (0, 0, 0), -1)
    cv2.putText(t, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    s1, s2 = (json.loads((r / "summary.json").read_text(encoding="utf-8")) for r in (R1, R2))
    rows = {}
    out = {"round1_trial": s1["trial"], "round2_trial": s2["trial"], "round2_note": s2.get("note", ""),
           "change": CHANGE, "splits": {}}
    for split in ("dev", "eval"):
        a, b = read_rows(R1 / f"{split}_per_image.csv"), read_rows(R2 / f"{split}_per_image.csv")
        d2 = {r["image"]: r for r in b}
        rows[split] = [(r["image"], r["iou"], d2[r["image"]]["iou"]) for r in a]
        delta = np.array([y - x for _, x, y in rows[split]])
        out["splits"][split] = {"round1": headline(s1[split], a), "round2": headline(s2[split], b),
                                "improved": int((delta > CHANGE).sum()), "worse": int((delta < -CHANGE).sum())}

    labels = list(csv.DictReader(open(LABELS, newline="", encoding="utf-8")))
    by_key = {k: (x, y) for split in rows for k, x, y in rows[split]}
    cause = collections.defaultdict(list)
    for l in labels:
        cause[l["cause"]].append(by_key[l["image"]])
    out["round1_failures_by_cause"] = {
        c: {"n": len(v), "iou_round1": round(float(np.mean([x for x, _ in v])), 3),
            "iou_round2": round(float(np.mean([y for _, y in v])), 3)}
        for c, v in sorted(cause.items(), key=lambda kv: -len(kv[1]))}
    out["fixed_round1_failures"] = int(sum(y >= 0.7 for v in cause.values() for _, y in v))
    out["round1_failures"] = len(labels)
    # where did each split's change come from: round-1 failures (IoU < 0.7) vs images that were already fine
    for split in ("dev", "eval"):
        assert sum(x < 0.7 for _, x, _ in rows[split]) == sum(l["split"] == split for l in labels), split
        for name, keep in (("failed", lambda x: x < 0.7), ("fine", lambda x: x >= 0.7)):
            g = [(x, y) for _, x, y in rows[split] if keep(x)]
            out["splits"][split][f"round1_{name}"] = {
                "n": len(g), "iou_round1": round(float(np.mean([x for x, _ in g])), 3),
                "iou_round2": round(float(np.mean([y for _, y in g])), 3),
                "improved": int(sum(y - x > CHANGE for x, y in g)), "worse": int(sum(x - y > CHANGE for x, y in g))}
    (OUT / "compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # scatter: every image, round 1 vs round 2
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--")
    for split in ("dev", "eval"):
        ax.scatter([x for _, x, _ in rows[split]], [y for _, _, y in rows[split]], s=30, color=SPLIT_COLOR[split],
                   alpha=0.85, edgecolors=SURFACE, linewidths=1.0, label=f"{split} (n={len(rows[split])})")
    ax.set_xlabel(f"IoU, round 1 ({s1['trial']})")
    ax.set_ylabel(f"IoU, round 2 ({s2['trial']})")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Per image: above the line = round 2 better")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "iou_round1_vs_round2.png")
    plt.close(fig)

    # grouped bars by round-1 failure cause
    items = list(out["round1_failures_by_cause"].items())[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(items) + 1.0))
    y = np.arange(len(items))
    hgt = 0.36
    ax.barh(y + hgt / 2 + 0.01, [v["iou_round1"] for _, v in items], height=hgt, color=BEFORE, label="round 1")
    ax.barh(y - hgt / 2 - 0.01, [v["iou_round2"] for _, v in items], height=hgt, color=AFTER, label="round 2")
    for yi, (_, v) in zip(y, items):
        ax.text(v["iou_round1"] + 0.01, yi + hgt / 2, f"{v['iou_round1']:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.text(v["iou_round2"] + 0.01, yi - hgt / 2, f"{v['iou_round2']:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [f"{c} (n={v['n']})" for c, v in items])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("mean IoU of the round-1 failures in this group (dev + eval)")
    ax.set_title("Round-1 failures: did round 2 fix them?")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "iou_by_cause.png")
    plt.close(fig)

    # biggest changes, both directions
    allrows = [(k, x, y) for split in rows for k, x, y in rows[split]]
    gains = sorted(allrows, key=lambda r: r[1] - r[2])[:4]
    losses = [r for r in sorted(allrows, key=lambda r: r[2] - r[1])[:2] if r[2] < r[1]]
    panels = []
    for k, x, y in gains + losses:
        rgb = load_rgb(DATA / k)
        gt, _ = load_gt(GT, k)
        name = "#" + k.split("not_damaged_")[1].split("_")[0]
        panels.append(np.hstack([tile(rgb, name), tile(overlay(rgb, gt, (60, 170, 90)), "GT (SAM)"),
                                 tile(error_map(rgb, gt, pred(R1, k)), f"round 1  IoU {x:.2f}"),
                                 tile(error_map(rgb, gt, pred(R2, k)), f"round 2  IoU {y:.2f}")]))
    save_rgb(OUT / "changes.jpg", np.vstack(panels))
    print(json.dumps(out["splits"], indent=1))


if __name__ == "__main__":
    main()
