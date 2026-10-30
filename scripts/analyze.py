"""Task 3 analysis of the final run (reads results/final_eval, never re-runs the method).

    python scripts/analyze.py [--final results/final_eval] [--out results/analysis]

Writes results/analysis/:
  metrics_table.json        dev vs eval
  diagnostics.csv           per image: FP / FN pixels, share of GT egg pixels the colour gate removes
  factors.json              mean IoU by egg count / hand / touching / lighting / colour / background,
                            Spearman correlation of IoU with brightness and egg area
  iou_hist.png, iou_vs_factors.png, iou_by_background.png, failure_taxonomy.png
  worst_eval_{1,2}.jpg      8 worst eval images: image | GT | prediction | error map
  failures/sheet_*.jpg      every image with IoU < FAIL_IOU, for labelling failure causes
  taxonomy.json             counts from failure_labels.csv (hand-made, one primary cause per failure)
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eggseg import classical as C  # noqa: E402
from eggseg.data import load_gt, load_rgb, mean_lightness  # noqa: E402
from eggseg.runner import DATA, GT  # noqa: E402
from eggseg.viz import error_map, error_panel, save_rgb  # noqa: E402

FINAL = ROOT / "results" / "final_eval"
OUT = ROOT / "results" / "analysis"
ATTR = ROOT / "data" / "processed" / "attributes_manual.csv"
FAIL_IOU = 0.7
GATE_HIGH = 0.4  # "the colour gate removed a big part of the egg"

# Reference palette (dataviz skill, light mode): categorical slots 1-2, text inks, surface.
SPLIT_COLOR = {"dev": "#2a78d6", "eval": "#eb6834"}
SINGLE = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150})


def read_rows(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k, v in r.items():
            if k != "image":
                r[k] = float(v) if v not in ("", "nan") else float("nan")
    return rows


def pred_mask(key):
    p = FINAL / "preds" / Path(key).parent / (Path(key).stem + ".png")
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def gate_loss(rgb, gt, cfg):
    """Share of GT egg pixels the colour gate (skin box, saturation) removes before any
    thresholding. Those pixels can only come back through GrabCut."""
    img = C.preprocess(rgb, cfg["preprocess"])
    none = np.zeros(gt.shape, bool)
    sk = C.skin(img, cfg["skin"]["cb"], cfg["skin"]["cr"]) if cfg["skin"]["enabled"] else none
    sat = ~C.low_saturation(img, cfg["saturation"]["s_max"]) if cfg["saturation"]["enabled"] else none
    n = max(int(gt.sum()), 1)
    return sk[gt].sum() / n, sat[gt].sum() / n, (sk | sat)[gt].sum() / n


def metrics_table(summary, rows_by_split):
    out = {}
    for s in ("dev", "eval"):
        m = summary[s]
        iou = np.array([r["iou"] for r in rows_by_split[s]])
        out[s] = {k: m[k] for k in ("iou", "dice", "precision", "recall", "boundary_f1")}
        out[s].update({"instance": m["instance"], "empty_predictions": m["empty_predictions"], "images": m["images"],
                       "n_good": int((iou >= 0.8).sum()), "n_poor": int((iou < 0.5).sum()),  # IoU >= 0.8 / < 0.5
                       "n_zero": int((iou == 0).sum())})
    return out


def factors(rows, attrs):
    out = {}
    for col in ("egg_count", "hand", "touching", "lighting", "egg_colour", "background"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[attrs[r["image"]][col]].append(r["iou"])
        out[col] = {k: {"n": len(v), "iou_mean": round(float(np.mean(v)), 3)} for k, v in sorted(groups.items())}
    iou = np.array([r["iou"] for r in rows])
    gate = np.array([r["gate_loss"] for r in rows])
    out["gate_loss"] = {"median": round(float(np.median(gate)), 3), "high_threshold": GATE_HIGH,
                        "n_high": int((gate >= GATE_HIGH).sum()),
                        "iou_mean_high": round(float(iou[gate >= GATE_HIGH].mean()), 3) if (gate >= GATE_HIGH).any() else None,
                        "iou_mean_low": round(float(iou[gate < GATE_HIGH].mean()), 3)}
    out["spearman_iou_vs_gate_loss"] = round(spearman(gate, iou), 3)
    out["spearman_iou_vs_brightness"] = round(spearman(np.array([r["brightness"] for r in rows]), iou), 3)
    out["spearman_iou_vs_egg_area"] = round(spearman(np.array([r["gt_area_frac"] for r in rows]), iou), 3)
    return out


def plot_hist(rows_by_split):
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    bins = np.linspace(0, 1, 11)
    w = 0.038
    for i, s in enumerate(("dev", "eval")):
        iou = [r["iou"] for r in rows_by_split[s]]
        frac = np.histogram(iou, bins)[0] / len(iou) * 100
        ax.bar(bins[:-1] + 0.05 + (i - 0.5) * (w + 0.004), frac, width=w, color=SPLIT_COLOR[s],
               label=f"{s} (n={len(iou)})", edgecolor=SURFACE, linewidth=0)
    ax.set_xticks(bins)
    ax.set_xlabel("IoU with pseudo-GT")
    ax.set_ylabel("% of images")
    ax.set_title("Per-image IoU, final pipeline")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "iou_hist.png")
    plt.close(fig)


def plot_factors(rows_by_split):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharey=True)
    axes = axes.ravel()
    for s in ("dev", "eval"):
        rows = rows_by_split[s]
        iou = [r["iou"] for r in rows]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(rows)) + (-0.15 if s == "dev" else 0.15)
        kw = dict(s=30, color=SPLIT_COLOR[s], alpha=0.85, edgecolors=SURFACE, linewidths=1.0)
        axes[0].scatter([r["n_gt"] + j for r, j in zip(rows, jitter)], iou, label=s, **kw)
        axes[1].scatter([r["brightness"] for r in rows], iou, **kw)
        axes[2].scatter([r["gt_area_frac"] * 100 for r in rows], iou, **kw)
        axes[3].scatter([r["gate_loss"] * 100 for r in rows], iou, **kw)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xlabel("eggs in image (GT)")
    axes[1].set_xlabel("mean brightness (L*)")
    axes[2].set_xlabel("egg area (% of image, log)")
    axes[2].set_xscale("log")
    axes[3].set_xlabel("% of GT egg pixels removed by the colour gate")
    axes[3].axvline(GATE_HIGH * 100, color=INK2, linewidth=0.8, linestyle="--")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("IoU")
    for ax, t in zip(axes, ("IoU vs egg count", "IoU vs brightness", "IoU vs egg size", "IoU vs colour-gate loss")):
        ax.set_title(t)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2)  # outside the panels: no overlap with points
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "iou_vs_factors.png")
    plt.close(fig)


def plot_background(fac):
    bg = sorted(fac["background"].items(), key=lambda kv: kv[1]["iou_mean"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    y = np.arange(len(bg))
    ax.barh(y, [v["iou_mean"] for _, v in bg], height=0.7, color=SINGLE)
    ax.set_yticks(y, [f"{k.replace('_', ' ')} (n={v['n']})" for k, v in bg])
    for yi, (_, v) in zip(y, bg):
        ax.text(v["iou_mean"] + 0.01, yi, f"{v['iou_mean']:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("mean IoU (dev + eval)")
    ax.set_title("Mean IoU by background")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "iou_by_background.png")
    plt.close(fig)


def plot_taxonomy(tax):
    items = sorted(tax["counts"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(items) + 0.9))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], height=0.7, color=SINGLE)
    ax.set_yticks(y, [k for k, _ in items])
    for yi, (_, v) in zip(y, items):
        ax.text(v + 0.15, yi, str(v), va="center", fontsize=8, color=INK2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"failed images (IoU < {FAIL_IOU}), dev + eval")
    ax.set_title("Failure causes (one primary cause per image)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "failure_taxonomy.png")
    plt.close(fig)


def failure_sheets(fails):
    """Labelled thumbnails (image | error map) of every failure, 12 per sheet."""
    for k in range(0, len(fails), 12):
        tiles = []
        for r in fails[k:k + 12]:
            rgb = load_rgb(DATA / r["image"])
            gt, _ = load_gt(GT, r["image"])
            pair = np.hstack([cv2.resize(rgb, (240, 240)), cv2.resize(error_map(rgb, gt, pred_mask(r["image"])), (240, 240))])
            label = f"{r['split']} {r['image'].split('/')[-1].split('_jpg')[0].replace('not_damaged_', '#')} IoU {r['iou']:.2f}"
            cv2.rectangle(pair, (0, 0), (480, 22), (0, 0, 0), -1)
            cv2.putText(pair, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(pair)
        while len(tiles) % 3:
            tiles.append(np.full_like(tiles[0], 255))
        save_rgb(OUT / "failures" / f"sheet_{k // 12 + 1}.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))


def main():
    global FINAL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="results/final_eval")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    FINAL, OUT = ROOT / a.final, ROOT / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    summary = json.loads((FINAL / "summary.json").read_text())
    attrs = {r["image"]: r for r in csv.DictReader(open(ATTR, newline=""))}
    rows_by_split = {s: read_rows(FINAL / f"{s}_per_image.csv") for s in ("dev", "eval")}
    cfg = summary["pipeline"]
    for s, rows in rows_by_split.items():
        for r in rows:
            rgb = load_rgb(DATA / r["image"])
            gt, _ = load_gt(GT, r["image"])
            pred = pred_mask(r["image"])
            r["split"] = s
            r["brightness"] = mean_lightness(rgb)
            r["fp_px"], r["fn_px"] = int((pred & ~gt).sum()), int((gt & ~pred).sum())
            r["skin_loss"], r["sat_loss"], r["gate_loss"] = (round(float(v), 3) for v in gate_loss(rgb, gt, cfg))
    all_rows = rows_by_split["dev"] + rows_by_split["eval"]
    with open(OUT / "diagnostics.csv", "w", newline="") as f:
        cols = ["image", "split", "iou", "n_gt", "n_pred", "fp_px", "fn_px", "skin_loss", "sat_loss", "gate_loss", "brightness"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    (OUT / "metrics_table.json").write_text(json.dumps(metrics_table(summary, rows_by_split), indent=2))
    fac = {"all": factors(all_rows, attrs), "eval": factors(rows_by_split["eval"], attrs)}
    (OUT / "factors.json").write_text(json.dumps(fac, indent=2))
    plot_hist(rows_by_split)
    plot_factors(rows_by_split)
    plot_background(fac["all"])

    worst = sorted(rows_by_split["eval"], key=lambda r: r["iou"])[:8]
    panels = []
    for r in worst:
        gt, _ = load_gt(GT, r["image"])
        name = r["image"].split("/")[-1].split("_jpg")[0].replace("not_damaged_", "#")
        panels.append(error_panel(load_rgb(DATA / r["image"]), gt, pred_mask(r["image"]), f"{name}  IoU {r['iou']:.2f}", size=300))
    for i in range(0, len(panels), 4):  # 4 per sheet so each fits on a page
        save_rgb(OUT / f"worst_eval_{i // 4 + 1}.jpg", np.vstack(panels[i:i + 4]))

    fails = sorted((r for r in all_rows if r["iou"] < FAIL_IOU), key=lambda r: (r["split"], r["iou"]))
    failure_sheets(fails)
    labels_path = OUT / "failure_labels.csv"
    if labels_path.exists():
        labels = list(csv.DictReader(open(labels_path, newline="")))
        missing = sorted({r["image"] for r in fails} - {l["image"] for l in labels})
        by_split = collections.defaultdict(collections.Counter)
        for l in labels:
            by_split[l["split"]][l["cause"]] += 1
        counts = collections.Counter(l["cause"] for l in labels)
        tax = {"fail_iou_below": FAIL_IOU, "failed_images": len(fails), "labelled": len(labels),
               "unlabelled": missing, "counts": dict(counts.most_common()),
               "by_split": {s: dict(c.most_common()) for s, c in by_split.items()}}
        (OUT / "taxonomy.json").write_text(json.dumps(tax, indent=2))
        plot_taxonomy(tax)
    print(f"failures (IoU < {FAIL_IOU}): {len(fails)}  ->", OUT)


if __name__ == "__main__":
    main()
