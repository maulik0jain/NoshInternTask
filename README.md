# NoshInternTask

Egg segmentation: pretrained-model ground truth → classical image-processing segmentation → evaluation → video tracking.

**Report:** [report/report.pdf](report/report.pdf)

| Part | Status |
|---|---|
| Task 1 — ground truth (Grounding DINO + SAM 2.1) | done |
| Dev / eval split (4:1, grouped by scene) | done, frozen |
| Task 2 — classical segmentation + experiment log | done ([experiments/LOG.md](experiments/LOG.md)) |
| Task 3 — evaluation & analysis | done ([results/](results/)) |
| Round 2 — the improvements from the Task 3 analysis (T12–T18) | done; round-1 results kept as they were ([results/round2/](results/round2/)) |
| Task 4 — video tracking | next |

## Setup

Python 3.10+ (tested on 3.13), an NVIDIA GPU recommended (tested on RTX 4070 Laptop, 8 GB; CPU works, slower).

```bash
pip install -r requirements.txt
python scripts/download_assets.py   # SAM 2.1 checkpoint -> checkpoints/ (dataset too, if missing)
```

Grounding DINO and DINOv2 weights download from Hugging Face on first run.

## Run

```bash
python scripts/task1_sam2_groundtruth.py   # Task 1 -> ground_truth/
python scripts/make_split.py               # split  -> data/splits/split.csv
```

The outputs of both are already committed. Re-running regenerates them.

### Task 2 — classical segmentation (dev split only)

```bash
python scripts/run_experiment.py experiments/T11/config.yaml --worst 8   # one trial -> experiments/T11/
python scripts/sweep.py experiments/T8/config.yaml grabcut.bg_dilate "[25, 45, 70]" --name T8_bg_dilate
python scripts/build_log.py                # experiments/LOG.md + log.json + progress.png from the trial folders
```

Each trial folder holds `config.yaml` (what changed, why, decision, reason), `resolved.yaml`
(full pipeline parameters), `results.json`, `per_image.csv` and worst-case panels. A trial
`extends` another one and lists only its `changes`; `configs/default.yaml` has every stage off.

### Task 3 — evaluation (the only run that touches eval)

```bash
python scripts/final_eval.py               # trial named in configs/final.yaml -> results/final_eval/
python scripts/analyze.py                  # metrics table, plots, failure sheets -> results/analysis/
python scripts/ablation.py                 # dev only: final pipeline with one stage off -> results/ablation/
python scripts/build_report.py             # report/report_template.html -> report.html + report.pdf
pytest tests/                              # metric sanity checks
```

`final_eval.py` refuses to run a second time once eval results exist in its output folder.
Failure causes in `results/analysis/failure_labels.csv` are labelled by hand from the failure sheets.

### Round 2 — improvements after the analysis

Trials T12–T18 (dev only, same rules as Task 2). The round-1 results above are not touched;
round 2 gets its own, second eval run into a separate folder.

```bash
python scripts/final_eval.py --config configs/final_round2.yaml --out results/final_eval_round2
python scripts/compare_rounds.py           # round 1 vs round 2 tables + plots -> results/round2/
python scripts/loss_check.py               # images that got worse, one round-2 change undone at a time
```

## Layout

```
data/egg_dataset/            110 images (Roboflow "egg" v1, CC BY 4.0)
data/splits/split.csv        frozen dev/eval split (image, source_id, group, num_eggs, split)
data/splits/scene_merges.json  manual same-scene merges used by the split
ground_truth/masks/          binary masks 0/255
ground_truth/instances/      instance ids (0 = background, 1..N = eggs, left to right)
ground_truth/overlays/       overlays for checking
ground_truth/annotations.json  boxes, scores, areas, settings
ground_truth_overrides.json  manual fixes from reviewing the overlays
data/processed/attributes_manual.csv  hand labels per image (hand, background, lighting...) for Task 3
src/eggseg/                  library: classical pipeline, metrics, config loading, plots
configs/                     default pipeline (all stages off) + final.yaml (round 1) + final_round2.yaml
experiments/                 one folder per trial (T0..T18), sweeps/, LOG.md
results/final_eval/          round-1 eval run (the clean one): summary.json, per-image CSVs, predicted masks
results/final_eval_round2/   round-2 eval run (second look at eval)
results/round2/              round 1 vs round 2 comparison, loss check
results/analysis/            Task 3 tables, plots, failure sheets and labels
results/ablation/            dev-only ablation of the final pipeline
scripts/                     entry points
report/                      report source (HTML), figures, PDF
```

## Dataset

[Roboflow Universe — egg](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1), CC BY 4.0.
