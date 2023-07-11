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

