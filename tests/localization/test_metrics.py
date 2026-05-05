"""Unit tests for leps_localizer metric extensions."""
from __future__ import annotations

import pytest

from src.localization.metrics import (
    DEFAULT_AREA_FRAC_BUCKETS,
    best_of_n_match,
    containment_iou,
    iou_xyxy,
    recall_by_bucket,
)


def test_iou_identical_boxes():
    assert iou_xyxy([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)


def test_iou_disjoint():
    assert iou_xyxy([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_iou_half_overlap():
    iou = iou_xyxy([0, 0, 100, 100], [50, 0, 150, 100])
    assert iou == pytest.approx(0.5 / 1.5)


def test_containment_pred_inside_gt():
    """Tight pred fully inside square gt = full pred containment (1.0)."""
    pred = [10, 10, 90, 90]
    gt = [0, 0, 100, 100]
    assert containment_iou(pred, gt) == pytest.approx(1.0)


def test_containment_partial():
    """Half of pred inside gt = 0.5 pred containment."""
    pred = [0, 0, 100, 100]
    gt = [50, 0, 150, 100]
    assert containment_iou(pred, gt) == pytest.approx(0.5)


def test_containment_handles_zero_gt():
    assert containment_iou([0, 0, 10, 10], [50, 50, 50, 50]) == 0.0


def test_best_of_n_match_hit():
    preds = [
        [0, 0, 50, 50],  # IoU ~0
        [10, 10, 90, 90],  # IoU 0.640... vs gt
    ]
    gt = [0, 0, 100, 100]
    hit, best, idx = best_of_n_match(preds, gt, iou_threshold=0.5)
    assert hit is True
    assert idx == 1
    assert best > 0.5


def test_best_of_n_match_miss_when_all_low_iou():
    preds = [[0, 0, 5, 5], [200, 200, 210, 210]]
    gt = [50, 50, 100, 100]
    hit, best, _ = best_of_n_match(preds, gt, iou_threshold=0.5)
    assert hit is False
    assert best < 0.5


def test_best_of_n_match_empty_preds():
    hit, best, idx = best_of_n_match([], [0, 0, 10, 10])
    assert hit is False
    assert best == 0.0
    assert idx == -1


def test_best_of_n_match_containment_helps_with_square_gt():
    """Tight pred inside large square gt: IoU low (<0.5) but containment ~1."""
    preds = [[40, 40, 60, 60]]
    gt = [0, 0, 100, 100]
    iou_hit, _, _ = best_of_n_match(preds, gt, iou_threshold=0.5)
    cont_hit, _, _ = best_of_n_match(preds, gt, iou_threshold=0.5, use_containment=True)
    assert iou_hit is False
    assert cont_hit is True


def test_recall_by_bucket_default_buckets():
    samples = [
        (0.02, True),
        (0.03, False),
        (0.07, True),
        (0.15, True),
        (0.30, False),
        (0.60, True),
        (0.95, True),
    ]
    res = recall_by_bucket(samples)
    assert len(res) == len(DEFAULT_AREA_FRAC_BUCKETS)
    by_bounds = {(r["lo"], r["hi"]): r for r in res}
    b1 = by_bounds[(0.0, 0.05)]
    assert b1["n"] == 2 and b1["hits"] == 1
    assert b1["recall"] == pytest.approx(0.5)
    b3 = by_bounds[(0.10, 0.25)]
    assert b3["n"] == 1 and b3["hits"] == 1
    assert b3["recall"] == pytest.approx(1.0)
    b4 = by_bounds[(0.25, 0.50)]
    assert b4["n"] == 1 and b4["hits"] == 0


def test_recall_by_bucket_empty_bucket_zero_recall():
    samples = [(0.30, True)]
    res = recall_by_bucket(samples)
    by_bounds = {(r["lo"], r["hi"]): r for r in res}
    empty = by_bounds[(0.0, 0.05)]
    assert empty["n"] == 0
    assert empty["recall"] == 0.0
