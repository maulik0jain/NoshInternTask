"""Why did some images get worse in round 2? Undo one round-2 change at a time.

    python scripts/loss_check.py

Takes every image (dev and eval) whose IoU dropped by more than 0.1 from round 1
to round 2, and re-runs the round-2 pipeline on just those images with one
round-2 change switched back to its round-1 setting. Analysis only: nothing is
tuned or chosen from it, and the reported round-2 numbers stay those in
results/final_eval_round2. Run after compare_rounds.py.
Second part: every image still at IoU 0 in round 2, re-run with the saturation
gate off, to check whether that gate (left on in round 2) is what still blocks it.
Writes results/round2/loss_check.json.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from eggseg.config import config_path, load_trial, set_by_path  # noqa: E402
from eggseg.runner import run  # noqa: E402

R1, R2 = ROOT / "results" / "final_eval", ROOT / "results" / "final_eval_round2"
DIAG = ROOT / "results" / "analysis" / "diagnostics.csv"
LABELS = ROOT / "results" / "analysis" / "failure_labels.csv"
DROP = 0.1
SAT_HIGH = 0.5  # the saturation gate removes at least half of the egg
UNDO = [  # (label, trial that added it, [(dotted param, round-1 value)])
    ("skin filter back on", "T12", [("skin.enabled", True)]),
    ("no ellipse prior", "T13", [("grabcut.ellipse_prior", None)]),
    ("no blob splitting", "T14", [("candidate.split_merged", False)]),
    ("no edge ranking", "T15c", [("candidate.edge_weight", 0.0)]),
    ("no shape/texture reject", "T16", [("components.max_rectangularity", None), ("candidate.max_texture", None)]),
]


def iou_by_image(path):
    with open(path, encoding="utf-8") as f:
        return {r["image"]: float(r["iou"]) for r in csv.DictReader(f)}


def main():
    trial = yaml.safe_load((ROOT / "configs" / "final_round2.yaml").read_text(encoding="utf-8"))["trial"]
    base = load_trial(config_path(trial))["pipeline"]
    losses = []
    for split in ("dev", "eval"):
        a, b = iou_by_image(R1 / f"{split}_per_image.csv"), iou_by_image(R2 / f"{split}_per_image.csv")
        losses += [(k, split, a[k], b[k]) for k in a if a[k] - b[k] > DROP]
    keys = [k for k, *_ in losses]
    out = {"base": trial, "drop": DROP, "images": [], "variants": [u[0] for u in UNDO],
           "variant_trials": {u[0]: u[1] for u in UNDO}}
    per_variant = {}
    for label, _, changes in UNDO:
        cfg = base
        for param, value in changes:
            cfg = set_by_path(cfg, param, value)  # returns a copy
        assert cfg != base, label
        rows, _ = run(cfg, keys)
        per_variant[label] = {r["image"]: r["iou"] for r in rows}
        print(f"{label:>24}: " + "  ".join(f"{r['iou']:.2f}" for r in rows), flush=True)
    for k, split, x, y in losses:
        undo = {label: round(per_variant[label][k], 4) for label, *_ in UNDO}
        best = max(undo, key=undo.get)
        out["images"].append({"image": k, "id": "#" + k.split("not_damaged_")[1].split("_")[0], "split": split,
                              "iou_round1": round(x, 4), "iou_round2": round(y, 4), "undo": undo,
                              "best_undo": best, "best_undo_trial": out["variant_trials"][best],
                              "recovered": round(undo[best] - y, 4)})
        print(f"{out['images'][-1]['id']:>5} {split}: {x:.2f} -> {y:.2f}; best undo '{best}' -> {undo[best]:.2f}")
    out["best_undo_counts"] = {label: sum(i["best_undo"] == label for i in out["images"]) for label, *_ in UNDO}
    out["n_images"] = len(out["images"])

    # images still at IoU 0 after round 2: is it the saturation gate (still on in round 2)?
    # sat_loss = share of the GT egg the saturation gate removes (results/analysis/diagnostics.csv; the
    # gate and preprocessing are the same in round 1 and round 2, checked below)
    r1 = load_trial(config_path(yaml.safe_load((ROOT / "configs" / "final.yaml").read_text(encoding="utf-8"))["trial"]))
    assert r1["pipeline"]["saturation"] == base["saturation"] and r1["pipeline"]["preprocess"] == base["preprocess"]
    with open(DIAG, encoding="utf-8") as f:
        diag = {r["image"]: r for r in csv.DictReader(f)}
    with open(LABELS, encoding="utf-8") as f:
        labels = {r["image"]: r for r in csv.DictReader(f)}
    zeros = [(k, split) for split in ("dev", "eval")
             for k, v in iou_by_image(R2 / f"{split}_per_image.csv").items() if v == 0]
    rows, _ = run(set_by_path(base, "saturation.enabled", False), [k for k, _ in zeros])
    no_sat = {r["image"]: r["iou"] for r in rows}
    out["still_zero"] = [{"id": "#" + k.split("not_damaged_")[1].split("_")[0], "split": split,
                          "sat_loss": float(diag[k]["sat_loss"]), "iou_no_saturation": round(no_sat[k], 4),
                          "cause": labels.get(k, {}).get("cause", "-"), "note": labels.get(k, {}).get("note", "")} for k, split in zeros]
    out["still_zero_by_split"] = {s: sum(z["split"] == s for z in out["still_zero"]) for s in ("dev", "eval")}
    out["still_zero_sat_high"] = sum(z["sat_loss"] >= SAT_HIGH for z in out["still_zero"])
    out["sat_high"] = SAT_HIGH
    for z in out["still_zero"]:
        print(f"{z['id']:>5} {z['split']}: sat_loss {z['sat_loss']:.2f}, IoU without saturation gate {z['iou_no_saturation']:.2f}"
              f"  ({z['cause']})")
    dest = ROOT / "results" / "round2"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "loss_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
