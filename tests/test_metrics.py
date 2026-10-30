"""Sanity checks for the metrics (run: python -m pytest tests -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eggseg.metrics import boundary_f1, instance_match, pixel_metrics  # noqa: E402


def square(y0, x0, s, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


def test_perfect_prediction():
    gt = square(10, 10, 40)
    m = pixel_metrics(gt, gt)
    assert m["iou"] == m["dice"] == m["precision"] == m["recall"] == 1.0
    assert boundary_f1(gt, gt) == 1.0


def test_empty_prediction():
    gt = square(10, 10, 40)
    m = pixel_metrics(np.zeros_like(gt), gt)
    assert m["iou"] == 0.0 and m["recall"] == 0.0 and np.isnan(m["precision"])


def test_half_overlap():
    gt, pred = square(0, 0, 40), square(0, 20, 40)  # 20x40 overlap, union 60x40
    assert abs(pixel_metrics(pred, gt)["iou"] - 1 / 3) < 1e-9


def test_boundary_tolerance():
    gt = square(10, 10, 40)
    assert boundary_f1(square(11, 11, 40), gt, tol=2) == 1.0   # 1 px shift: inside tolerance
    assert boundary_f1(square(20, 20, 40), gt, tol=2) < 0.5    # 10 px shift: mostly outside


def test_instance_split():
    gt = np.zeros((100, 100), np.int32)
    gt[10:40, 10:40], gt[10:40, 50:80] = 1, 2
    merged = np.zeros((100, 100), np.int32)
    merged[10:40, 10:80] = 1                       # one blob covering both eggs
    assert instance_match(merged, gt) == (0, 1, 2)  # IoU < 0.5 with either egg
    assert instance_match(gt, gt) == (2, 2, 2)
