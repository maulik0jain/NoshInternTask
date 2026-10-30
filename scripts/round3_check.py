"""Round 3, last step: did the dev estimate predict eval this time?

    python scripts/round3_check.py

Run after the round-3 eval run (final_eval.py --config configs/final_round3.yaml
--out results/final_eval_round3). For each round: the dev number its decisions
were made on (in-sample), the leave-one-scene-out estimate where there is one
(results/round3/loso.json), and eval with a 90% bootstrap interval over the eval
images. The gap "dev in-sample - eval" is how much the round overfitted its dev
scenes. Nothing here feeds back into any decision.
Writes results/round3/check.json and predicted_vs_eval.png.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import GRID, INK2, SURFACE, style  # noqa: E402

RES = ROOT / "results"
RUNS = {1: RES / "final_eval", 2: RES / "final_eval_round2", 3: RES / "final_eval_round3"}
LOSO_OF = {2: "round-2 rule, round-2 changes", 3: "round-3 rule, + clip option"}  # selection procedure per round
N_BOOT, SEED, CHANGE = 2000, 0, 0.1
SEEN, HELD, EVAL = "#b9b8b3", "#2a78d6", "#eb6834"


def per_image(run, split):
    with open(run / f"{split}_per_image.csv", newline="", encoding="utf-8") as f:
        return {r["image"]: float(r["iou"]) for r in csv.DictReader(f)}


def boot_ci(values, rng):
    v = np.asarray(values)
    m = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return [round(float(np.percentile(m, 5)), 4), round(float(np.percentile(m, 95)), 4)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    loso = json.loads((RES / "round3" / "loso.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "rounds": {}, "eval_deltas": {}}
    ev = {r: per_image(p, "eval") for r, p in RUNS.items()}
    keys = sorted(ev[1])
    for r, p in RUNS.items():
        s = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        dev, em = s["dev"]["iou"]["mean"], s["eval"]["iou"]["mean"]  # same numbers as the trial tables
        e = [ev[r][k] for k in keys]
        row = {"trial": s["trial"], "dev_in_sample": round(dev, 4), "eval": round(em, 4),
               "eval_ci90": boot_ci(e, rng), "gap_dev_minus_eval": round(dev - em, 4)}
        if r in LOSO_OF:
            row["loso"] = loso["procedures"][LOSO_OF[r]]["loso"]["iou_image"]
            row["loso_procedure"] = LOSO_OF[r]
            row["gap_loso_minus_eval"] = round(row["loso"] - row["eval"], 4)
        out["rounds"][str(r)] = row
        print(f"round {r} ({s['trial']}): dev {dev:.3f}  LOSO {row.get('loso', float('nan')):.3f}  "
              f"eval {row['eval']:.3f} [{row['eval_ci90'][0]:.3f}, {row['eval_ci90'][1]:.3f}]")
    for a, b in ((1, 3), (2, 3), (1, 2)):  # paired: same eval images
        d = np.array([ev[b][k] - ev[a][k] for k in keys])
        out["eval_deltas"][f"{b}_vs_{a}"] = {"mean": round(float(d.mean()), 4), "ci90": boot_ci(d, rng),
                                             "better": int((d > CHANGE).sum()), "worse": int((d < -CHANGE).sum())}
    # eval images that moved between round 2 and round 3 (the clip was aimed at #124 / #145-type spill)
    out["moved_2_to_3"] = [{"id": "#" + k.split("not_damaged_")[1].split("_")[0], "round2": round(ev[2][k], 3),
                            "round3": round(ev[3][k], 3)} for k in keys if abs(ev[3][k] - ev[2][k]) > CHANGE]
    (RES / "round3" / "check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.26
    for i, r in enumerate(("1", "2", "3")):
        row = out["rounds"][r]
        bars = [(-w, row["dev_in_sample"], SEEN), (0, row.get("loso"), HELD), (w, row["eval"], EVAL)]
        for dx, v, c in bars:
            if v is None:
                ax.text(i + dx, 0.02, "n/a", ha="center", fontsize=8, color=INK2)
                continue
            ax.bar(i + dx, v, width=w - 0.03, color=c)
            ax.text(i + dx, 0.03, f"{v:.3f}", ha="center", fontsize=8, color="white", fontweight="bold")
        lo, hi = row["eval_ci90"]
        ax.plot([i + w, i + w], [lo, hi], color=INK2, linewidth=1.2)
    ax.set_xticks([0, 1, 2], [f"round {r} ({out['rounds'][r]['trial']})" for r in ("1", "2", "3")])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("mean IoU", color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for c, lab in ((SEEN, "dev, scenes the choices were made on"), (HELD, "dev, held-out scene (LOSO)"),
                   (EVAL, "eval (line = 90% bootstrap interval)")):
        ax.bar([0], [0], color=c, label=lab)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=3, handlelength=1.2, columnspacing=1.0)
    ax.set_title("What dev predicted vs what eval measured", fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(RES / "round3" / "predicted_vs_eval.png", dpi=150, facecolor=SURFACE)
    print(json.dumps(out["eval_deltas"], indent=1))


if __name__ == "__main__":
    main()
