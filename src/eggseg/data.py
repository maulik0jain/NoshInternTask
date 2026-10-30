"""Dataset access: unzip, list images, parse Roboflow filenames, load images."""
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Roboflow export names look like   not_damaged_105_jpg.rf.6b94b57d4e4cb521abbec55bb95c2a76.jpg
#                                   <original stem>_<original ext>.rf.<32-hex hash>.<ext>
# Roboflow re-encodes every image and appends .rf.<hash>; augmented copies of one
# photo keep the same original stem. So the stem without ".rf.<hash>" (the
# "source") is the unit that must never be split across dev/eval.
RF_NAME = re.compile(
    r"^(?P<source>.+?)_(?P<orig_ext>jpe?g|png)\.rf\.(?P<rf_hash>[0-9a-f]{32})\.(?P<ext>jpe?g|png)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageRecord:
    path: Path        # absolute path on disk
    key: str          # path relative to the dataset root (posix) - the image's id everywhere
    source: str       # original photo name, e.g. "not_damaged_105"
    rf_hash: str      # Roboflow hash, "" if the name doesn't follow the pattern
    folder: str       # Roboflow folder, e.g. "train/egg" (class-folder export)


def unzip(zip_path, out_dir):
    """Extract once; a second call is a no-op if images are already there."""
    out_dir = Path(out_dir)
    if not any(p.suffix.lower() in IMAGE_EXTS for p in out_dir.rglob("*")):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    return out_dir


def parse_name(filename):
    """Split a Roboflow filename into source / hash. Non-matching names keep their stem as source."""
    m = RF_NAME.match(filename)
    if m is None:
        return {"source": Path(filename).stem, "rf_hash": "", "matches_rf_pattern": False}
    return {"source": m["source"], "rf_hash": m["rf_hash"].lower(), "matches_rf_pattern": True}


def list_images(root):
    """All images under root, sorted by key, as ImageRecords."""
    root = Path(root)
    recs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        info = parse_name(p.name)
        recs.append(ImageRecord(path=p, key=p.relative_to(root).as_posix(), source=info["source"],
                                rf_hash=info["rf_hash"], folder=p.parent.relative_to(root).as_posix()))
    return recs


def load_rgb(path):
    """Load an image as uint8 RGB (OpenCV loads BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_split(csv_path):
    """{'dev': [keys], 'eval': [keys]} from the frozen split CSV (key = image path relative to dataset root)."""
    import csv
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["split"], []).append(row["image"])
    return out


def gt_path(gt_root, key, kind="masks"):
    """ground_truth/<kind>/<key with .png>. Path.stem only strips the final .jpg, so the .rf.<hash> part survives."""
    k = Path(key)
    return Path(gt_root) / kind / k.parent / (k.stem + ".png")


def load_gt(gt_root, key):
    """(binary mask bool HxW, instance labels uint8 HxW) for one image."""
    mask = cv2.imread(str(gt_path(gt_root, key, "masks")), cv2.IMREAD_GRAYSCALE)
    inst = cv2.imread(str(gt_path(gt_root, key, "instances")), cv2.IMREAD_GRAYSCALE)
    if mask is None or inst is None:
        raise FileNotFoundError(gt_path(gt_root, key))
    return mask > 127, inst


def mean_lightness(rgb):
    """Mean L* (0-100) of CIE Lab: a perceptual brightness measure, used to bin images by lighting."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)  # uint8 L is scaled to 0-255
    return float(lab[..., 0].mean() * 100.0 / 255.0)
