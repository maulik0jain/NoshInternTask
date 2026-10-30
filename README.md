# NoshInternTask

Egg segmentation: pretrained-model ground truth → classical image-processing segmentation → evaluation → video tracking.

**Report:** [report/report.pdf](report/report.pdf)

| Part | Status |
|---|---|
| Task 1 — ground truth (Grounding DINO + SAM 2.1) | done |
| Dev / eval split (4:1, grouped by scene) | done, frozen |
| Task 2 — classical segmentation | next |
| Task 3 — evaluation | — |
| Task 4 — video tracking | — |

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
scripts/                     all code
report/                      report source (HTML), figures, PDF
```

## Dataset

[Roboflow Universe — egg](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1), CC BY 4.0.
