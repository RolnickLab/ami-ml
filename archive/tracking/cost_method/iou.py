import numpy as np

""""
Author        : Aditya Jain
Date created  : 15th March, 2022
About         : Finds intersection over union for a bounding box pair
"""


def intersection_over_union(bb1, bb2):
    """
    Finds intersection over union for a bounding box pair

    Args
    ----------
    bb1 = [x1, y1, x2, y2]
    bb2 = [x1, y1, x2, y2]
    The origin is top-left corner; x1<x2; y1<y2; integer values in the list

    Return
    -----------
    0<=float<=1
    """
    assert bb1[0] < bb1[2], "Issue in bounding box 1 x_annotation"
    assert bb1[1] < bb1[3], "Issue in bounding box 1 y_annotation"
    assert bb2[0] < bb2[2], "Issue in bounding box 2 x_annotation"
    assert bb2[1] < bb2[3], "Issue in bounding box 2 y_annotation"

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    x_min = max(bb1[0], bb2[0])
    x_max = min(bb1[2], bb2[2])
    width = max(0, x_max - x_min + 1)

    y_min = max(bb1[1], bb2[1])
    y_max = min(bb1[3], bb2[3])
    height = max(0, y_max - y_min + 1)

    intersec_area = width * height
    union_area = bb1_area + bb2_area - intersec_area

    iou = np.around(intersec_area / union_area, 2)
    assert 0 <= iou <= 1, "IoU out of bounds"

    return iou
