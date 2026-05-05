import json
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou


def compute_similarity_matrices(
    gt: tp.Dict[str, tp.List], preds: tp.Dict[str, tp.List]
) -> tp.List[np.ndarray]:
    """For each image in preds, a similarity matrix S is computed. S is NxM, with N=nb
    of ground truths and M=nb of detections. S(i,j) is the IoU between ground truth box
    i and detected box j.

    Parameters
    ----------
    gt : tp.Dict[str, tp.List]
        Ground truth bounding boxes. The dictionary is expected to contain a list for
        each image ID. The first element of the list must be a list of bounding boxes in
        the [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    preds : tp.Dict[str, tp.List]
        Predicted bounding boxes. Same format as 'gt'.
    """
    similarity_matrices = []
    for img_id, img_preds in preds.items():
        bboxes_preds = torch.tensor(img_preds[0])
        bboxes_gt = torch.tensor(gt[img_id][0])
        similarity_matrix = box_iou(bboxes_gt, bboxes_preds)
        similarity_matrices.append(similarity_matrix.numpy())

    return similarity_matrices


def compute_matches_per_image(
    similarity: np.ndarray,
    scores: np.ndarray,
    similarity_thresholds: tp.Sequence[float],
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Greedily computes matches in a single image. Note: if two detections have a
    similarity higher than the threshold with one ground truth, only the one with the
    highest score matches with it (even if it's similarity with the ground truth is
    higher).

    Parameters
    ----------
    similarity : np.ndarray[N, M]
        Similarity matrix, must be postive.
        A high value of similarity[i,j] indicates that the j-th detection is very
        similar to the i-th ground truth, typically in the sense of IoU.
    scores : np.ndarray[M]
        Score of each detection (0 <= score <= 1)
    similarity_thresholds : tp.Sequence[float]
        List of T similarity thresholds at which the matching is performed.
        A match is considered as good if the similarity is greater than or equal to
        the threshold.

    Returns
    -------
    tp.Tuple[np.ndarray, np.ndarray]
        A tuple representing the matching on the image with the following keys:
        gt_matches : np.ndarray[T, N]
        dt_matches : np.ndarray[T, M]

        T is the number of thresholds in 'similarity_thresholds'. gt_matches
        (resp. dt_matches) indicate the index of the corresponding match in the
        detection list (resp. ground truth list). A value of -1 indicates no matching.
    """

    n_gt, n_dt = similarity.shape
    similarity_thresholds = np.asarray(similarity_thresholds)

    gt_matches = -np.ones((len(similarity_thresholds), n_gt), dtype=int)
    dt_matches = -np.ones((len(similarity_thresholds), n_dt), dtype=int)

    if n_gt == 0 or n_dt == 0:
        return gt_matches, dt_matches

    # Sort predictions by descending score
    permutation = np.argsort(-scores)

    # Greedily match predictions with ground truths
    for idx_sim, similarity_threshold in enumerate(similarity_thresholds):
        gt_remaining = np.where(gt_matches[idx_sim] == -1)[0]

        for idx_dt in permutation:
            if len(gt_remaining) == 0:
                # No remaining ground truth
                break

            # Find the best matching gt among remaining gt
            gt_idx = gt_remaining[np.argmax(similarity[gt_remaining, idx_dt])]

            if similarity[gt_idx, idx_dt] >= similarity_threshold:
                gt_matches[idx_sim, gt_idx] = idx_dt
                dt_matches[idx_sim, idx_dt] = gt_idx
                gt_remaining = np.where(gt_matches[idx_sim] == -1)[0]
                continue

    return gt_matches, dt_matches


def compute_matches(
    similarity_matrices: tp.List[np.ndarray],
    scores: tp.List[np.ndarray],
    similarity_thresholds: tp.Sequence[float],
):
    gt_matches, dt_matches = [], []

    for similarity_matrix, scores_per_image in zip(similarity_matrices, scores):
        gt_matches_per_image, dt_matches_per_image = compute_matches_per_image(
            similarity_matrix, scores_per_image, similarity_thresholds
        )
        gt_matches.append(gt_matches_per_image)
        dt_matches.append(dt_matches_per_image)

    return gt_matches, dt_matches


def extract_scores(preds: tp.Dict[str, tp.List]) -> tp.List[np.ndarray]:
    """
    Parameters
    ----------
    preds : tp.Dict[str, tp.List]
        Predicted bounding boxes. The dictionary is expected to contain a list for each
        image ID. The third element of the list must be the list of scores given to each
        bounding box of the image.
    """
    scores = []
    for img_preds in preds.values():
        scores.append(np.array(img_preds[2]))
    return scores


def remove_precision_zigzags(precision: np.ndarray) -> np.ndarray:
    """Turn 'precision' into a non-increasing array.
    This leads to overestimating the AP, but possibly more realistic results.

    Parameters
    ----------
    precision : np.ndarray[T, E]
        precision curve at T thresholds and E elements.

    Returns
    -------
    np.ndarray[T, E]
        The same precision curve but with zigzags removed.
    """

    return np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]


def compute_precision_recall(
    gt_path: str,
    preds_path: str,
    similarity_thresholds: tp.Sequence[float] = [0.5, 0.75],
    return_scores: bool = False,
):
    with open(gt_path) as f:
        gt = json.load(f)
    with open(preds_path) as f:
        preds = json.load(f)

    similarity_matrices = compute_similarity_matrices(gt, preds)
    scores = extract_scores(preds)
    gt_matches, dt_matches = compute_matches(
        similarity_matrices, scores, similarity_thresholds
    )

    # Merge lists of matches
    dt_matches = np.concatenate(dt_matches, axis=1)
    gt_matches = np.concatenate(gt_matches, axis=1)
    scores = np.concatenate(scores)

    # Sanity check
    if scores.shape[0] != dt_matches.shape[1]:
        raise ValueError(
            f"The number of scores ({scores.shape[0]})does not match"
            f"the number of detections ({dt_matches.shape[1]})."
        )

    # Sort detections by descending score
    permutation = np.argsort(-scores)
    dt_matches = dt_matches[:, permutation]

    # Compute precision & recall
    eps = np.finfo(np.float64).eps
    n_gt = gt_matches.shape[1]
    tp = np.cumsum(dt_matches >= 0, axis=1, dtype=float)
    fp = np.cumsum(dt_matches == -1, axis=1, dtype=float)

    precision = tp / (fp + tp + eps)
    precision[fp + tp == 0] = 1
    precision = remove_precision_zigzags(precision)
    recall = tp / (n_gt + eps)

    if return_scores:
        return precision, recall, scores[permutation]

    return precision, recall


# --- leps_localizer metric extensions ---------------------------------------
# Best-of-N matching for single-subject tasks (any pred near the gt = hit) and
# containment-aware metrics for FG-square-gt vs YOLO-tight-rect-pred mismatch.


def iou_xyxy(box_a: tp.Sequence[float], box_b: tp.Sequence[float]) -> float:
    """IoU of two axis-aligned [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def containment_iou(
    pred_xyxy: tp.Sequence[float], gt_xyxy: tp.Sequence[float]
) -> float:
    """Fraction of pred that lands inside gt: |pred ∩ gt| / |pred|.

    Built for the FG square-gt vs YOLO-tight-pred mismatch: a tight pred
    fully inside a fat square gt scores 1.0 (the pred is on the subject,
    even though IoU is low because the gt is bigger than it needs to be).
    """
    inter_x1 = max(pred_xyxy[0], gt_xyxy[0])
    inter_y1 = max(pred_xyxy[1], gt_xyxy[1])
    inter_x2 = min(pred_xyxy[2], gt_xyxy[2])
    inter_y2 = min(pred_xyxy[3], gt_xyxy[3])
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    pred_area = max(0.0, pred_xyxy[2] - pred_xyxy[0]) * max(
        0.0, pred_xyxy[3] - pred_xyxy[1]
    )
    return float(inter / pred_area) if pred_area > 0 else 0.0


def best_of_n_match(
    preds_xyxy: tp.Sequence[tp.Sequence[float]],
    gt_xyxy: tp.Sequence[float],
    *,
    iou_threshold: float = 0.5,
    use_containment: bool = False,
) -> tp.Tuple[bool, float, int]:
    """Single-subject "best of N" reduction.

    Returns (hit, best_score, best_pred_idx). `hit` is True if ANY prediction
    has IoU (or containment, when `use_containment`) ≥ threshold against the
    single gt box.
    """
    if not preds_xyxy:
        return False, 0.0, -1
    scorer = containment_iou if use_containment else iou_xyxy
    scores = [scorer(p, gt_xyxy) for p in preds_xyxy]
    best_idx = int(np.argmax(scores))
    best = float(scores[best_idx])
    return (best >= iou_threshold, best, best_idx)


# Default leps_localizer area-fraction buckets (lower-inclusive, upper-exclusive
# except the last which is fully closed).
DEFAULT_AREA_FRAC_BUCKETS: tp.List[tp.Tuple[float, float]] = [
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.25),
    (0.25, 0.50),
    (0.50, 1.01),
]


def _bucket_index(frac: float, buckets: tp.Sequence[tp.Tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= frac < hi:
            return i
    return -1


def recall_by_bucket(
    samples: tp.Sequence[tp.Tuple[float, bool]],
    *,
    buckets: tp.Optional[tp.Sequence[tp.Tuple[float, float]]] = None,
) -> tp.List[tp.Dict[str, float]]:
    """Per-bucket recall.

    `samples` is a list of `(bbox_area_fraction, hit)` pairs — one per gt.
    Returns one dict per bucket: `{lo, hi, n, hits, recall}`.
    """
    if buckets is None:
        buckets = DEFAULT_AREA_FRAC_BUCKETS
    counts = [{"lo": lo, "hi": hi, "n": 0, "hits": 0} for (lo, hi) in buckets]
    for frac, hit in samples:
        idx = _bucket_index(frac, buckets)
        if idx < 0:
            continue
        counts[idx]["n"] += 1
        if hit:
            counts[idx]["hits"] += 1
    out: tp.List[tp.Dict[str, float]] = []
    for c in counts:
        recall = (c["hits"] / c["n"]) if c["n"] else 0.0
        out.append({**c, "recall": recall})
    return out


def create_precision_recall_fig(
    precision: np.ndarray,
    recall: np.ndarray,
    iou_thresholds: tp.Sequence[float],
    idx: tp.Optional[int] = None,
):
    T = len(iou_thresholds)
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    marked_pr_points = []

    for i in range(T):
        ax.plot(recall[i], precision[i], label=iou_thresholds[i])
        if idx is not None:
            pt = ax.plot(recall[i, idx], precision[i, idx], c="red", marker="x")[0]
            marked_pr_points.append(pt)

    ax.legend(title="IoU threshold")
    ax.grid(visible=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision - Recall curve")

    return fig, marked_pr_points
