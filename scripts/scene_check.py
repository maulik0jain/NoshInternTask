"""Round 3, step 1: would the round-2 changes pass a scene-level rule? (dev only)

    python scripts/scene_check.py

Re-judges every round-2 decision with the round-3 rule (src/eggseg/scenes.py) from
the per-image results the trials already saved: same comparison as in round 2
(trial vs the trial it was tested on), but per scene instead of per image.
Nothing is re-run and eval is not read. Writes results/round3/scene_check.json
and scene_check.png (per-scene IoU change of each round-2 change).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from analyze import INK2, SURFACE, style  # noqa: E402
from eggseg import scenes  # noqa: E402
from eggseg.runner import SPLIT  # noqa: E402

OUT = ROOT / "results" / "round3"
EXP = ROOT / "experiments"
PAIRS = [  # (label, trial it was tested on, trial)
    ("skin filter off (T12)", "T11", "T12"),
    ("ellipse prior (T13)", "T12", "T13"),
    ("blob splitting (T14)", "T13", "T14"),
    ("edge ranking (T15c)", "T13", "T15c"),
    ("shape / texture reject (T16)", "T13", "T16"),
    ("all together (T17)", "T13", "T17"),
    ("edge weight 0.3 (T18)", "T17", "T18"),
    ("round 2 as a whole (T11 → T18)", "T11", "T18"),
]
BETTER, WORSE, MID = "#2a78d6", "#c8442f", "#f0efec"


def scene_labels(groups):
    """Readable scene names from the hand labels: most common background in the scene."""
    with open(ROOT / "data" / "processed" / "attributes_manual.csv", newline="", encoding="utf-8") as f:
        bg = {r["image"]: r["background"].replace("_", " ") for r in csv.DictReader(f)}
    name = {g: Counter(bg[k] for k, gg in groups.items() if gg == g).most_common(1)[0][0] for g in set(groups.values())}
    dup = Counter(name.values())
    n = Counter(groups.values())
    return {g: f"{v}{' ' + g if dup[v] > 1 else ''} ({n[g]})" for g, v in name.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    groups = scenes.scene_groups(SPLIT, "dev")
    labels = scene_labels(groups)
    out = {"rule": {"help": scenes.HELP, "worst": scenes.WORST, "n_boot": scenes.N_BOOT, "p_min": scenes.P_MIN},
           "n_scenes": len(set(groups.values())), "scene_labels": labels, "changes": []}
    for label, a, b in PAIRS:
        c = scenes.compare(scenes.read_iou(EXP / a / "per_image.csv"), scenes.read_iou(EXP / b / "per_image.csv"), groups)
        out["changes"].append({"label": label, "base": a, "trial": b, **c})
        print(f"{label:>32}: image d {c['image_delta']:+.3f}  scene d {c['scene_delta']:+.3f}  "
              f"helped {c['helped']:>2} hurt {c['hurt']:>2}  worst {c['worst']:+.2f}  P(gain) {c['p_gain']:.2f}  "
              f"-> {'pass' if c['passed'] else 'FAIL'}")
    out["n_passed"] = sum(c["passed"] for c in out["changes"][:5])
    (OUT / "scene_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # heatmap: scenes x the five single changes (+ round 2 as a whole)
    style()
    cols = [c for c in out["changes"] if c["trial"] in ("T12", "T13", "T14", "T15c", "T16")] + [out["changes"][-1]]
    order = sorted(labels, key=lambda g: -sum(1 for v in groups.values() if v == g))
    m = np.array([[c["per_scene"][g] for c in cols] for g in order])
    lim = 0.3
    cmap = LinearSegmentedColormap.from_list("div", [WORSE, MID, BETTER])
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.imshow(np.clip(m, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, "0" if abs(v) < 0.005 else f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.18 else "#1f1e1c")
    ax.set_xticks(range(len(cols)), [c["label"].replace(" (", "\n(") for c in cols], fontsize=8.5)
    ax.set_yticks(range(len(order)), [labels[g] for g in order], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.axvline(len(cols) - 1.5, color=INK2, linewidth=1)
    ax.set_title("Dev, per scene: IoU change from each round-2 change (blue = better)", fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "scene_check.png", dpi=150, facecolor=SURFACE)
    print(f"{out['n_passed']} of 5 round-2 changes pass the scene rule -> {OUT / 'scene_check.json'}")


if __name__ == "__main__":
    main()
