"""Phase 0: inspect the raw dataset.

Outputs (results/phase0/):
  inspect.json                 counts, resolutions, filename patterns, annotation check, brightness,
                               and (if data/processed/attributes_manual.csv exists) the variety summary
  contact_sheet_random30.jpg   30 random images (fixed seed)
  all_images_<k>.jpg           every image labelled by source id, used to make the manual attribute labels
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from eggseg.data import IMAGE_EXTS, list_images, load_rgb, mean_lightness, unzip  # noqa: E402
from eggseg.viz import contact_sheet, save_rgb  # noqa: E402

# Files that would mean the export carries labels (COCO, VOC, YOLO, CSV, mask PNGs).
ANNOTATION_HINTS = ("_annotations", ".json", ".xml", ".csv", "labels/", "masks/", "_mask")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=ROOT / "data" / "egg_dataset.zip")
    p.add_argument("--out-data", type=Path, default=ROOT / "data" / "egg_dataset")
    p.add_argument("--attributes", type=Path, default=ROOT / "data" / "processed" / "attributes_manual.csv")
    p.add_argument("--results", type=Path, default=ROOT / "results" / "phase0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def short_id(source):
    """not_damaged_105 -> '105' (all sources here end in a number); falls back to the full name."""
    tail = source.rsplit("_", 1)[-1]
    return tail if tail.isdigit() else source


def zip_inventory(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    exts = collections.Counter(Path(n).suffix.lower() or "<none>" for n in names)
    non_images = [n for n in names if Path(n).suffix.lower() not in IMAGE_EXTS]
    annotation_like = [n for n in non_images if any(h in n.lower() for h in ANNOTATION_HINTS)]
    return {"files": len(names), "extensions": dict(exts), "non_image_files": non_images,
            "annotation_like_files": annotation_like}


def summarise_attributes(path, keys):
    """Counts from the hand-made attribute labels (one row per image)."""
    rows = list(csv.DictReader(open(path, newline="")))
    missing = sorted(set(keys) - {r["image"] for r in rows})
    out = {"labelled_images": len(rows), "unlabelled_images": missing}
    for col in ("egg_count", "egg_colour", "hand", "touching", "lighting", "background"):
        out[col] = dict(sorted(collections.Counter(r[col] for r in rows).items()))
    return out


def main():
    a = parse_args()
    a.results.mkdir(parents=True, exist_ok=True)
    report = {"zip": {"path": a.zip.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(a.zip.read_bytes()).hexdigest(),
                      **zip_inventory(a.zip)}}

    unzip(a.zip, a.out_data)
    recs = list_images(a.out_data)
    imgs = {r.key: load_rgb(r.path) for r in recs}

    sizes = collections.Counter(f"{im.shape[1]}x{im.shape[0]}" for im in imgs.values())
    modes = collections.Counter(Image.open(r.path).mode for r in recs)
    by_source = collections.Counter(r.source for r in recs)
    file_md5 = collections.Counter(hashlib.md5(r.path.read_bytes()).hexdigest() for r in recs)
    light = np.array([mean_lightness(imgs[r.key]) for r in recs])

    report["images"] = {
        "count": len(recs),
        "per_folder": dict(collections.Counter(r.folder for r in recs)),
        "resolutions": dict(sizes),
        "pil_modes": dict(modes),
        "file_kb": {"min": round(min(r.path.stat().st_size for r in recs) / 1024, 1),
                    "median": round(float(np.median([r.path.stat().st_size for r in recs])) / 1024, 1),
                    "max": round(max(r.path.stat().st_size for r in recs) / 1024, 1)},
    }
    report["filenames"] = {
        "match_roboflow_pattern": sum(bool(r.rf_hash) for r in recs),
        "example": recs[0].path.name,
        "unique_sources": len(by_source),
        "sources_with_multiple_images": {s: n for s, n in by_source.items() if n > 1},
        "unique_rf_hashes": len({r.rf_hash for r in recs}),
        "byte_identical_files": sum(n - 1 for n in file_md5.values() if n > 1),
        "source_prefixes": dict(collections.Counter(r.source.rsplit("_", 1)[0] for r in recs)),
    }
    report["brightness_mean_Lstar"] = {k: round(float(v), 1) for k, v in
                                       zip(("min", "p10", "median", "p90", "max"),
                                           np.percentile(light, [0, 10, 50, 90, 100]))}

    rng = random.Random(a.seed)
    sample = sorted(rng.sample(recs, 30), key=lambda r: int(short_id(r.source)) if short_id(r.source).isdigit() else 0)
    save_rgb(a.results / "contact_sheet_random30.jpg",
             contact_sheet([imgs[r.key] for r in sample], [short_id(r.source) for r in sample], cols=6, tile=200))
    report["random30_sources"] = [r.source for r in sample]

    ordered = sorted(recs, key=lambda r: int(short_id(r.source)) if short_id(r.source).isdigit() else 0)
    for k in range(0, len(ordered), 30):
        chunk = ordered[k:k + 30]
        save_rgb(a.results / f"all_images_{k // 30 + 1}.jpg",
                 contact_sheet([imgs[r.key] for r in chunk], [short_id(r.source) for r in chunk], cols=6, tile=220))

    if a.attributes.exists():
        report["variety_manual_labels"] = summarise_attributes(a.attributes, [r.key for r in recs])

    (a.results / "inspect.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
