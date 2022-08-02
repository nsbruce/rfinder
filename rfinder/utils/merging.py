from itertools import combinations
from typing import List

import numpy as np

from rfinder.types import Box


def merge_overlapping(boxes: List[Box]) -> List[Box]:
    """Merges list of boxes when they overlap

    Args:
        boxes (List[Box]): boxes to attempt to merge

    Returns:
        List[Box]: non-overlapping boxes
    """
    if len(boxes) <= 1:
        return boxes

    independent_boxes: List[Box] = []
    overlapping_boxes: List[Box] = []
    for box in boxes:
        if not any(box.overlaps(other) for other in boxes if other != box):
            independent_boxes.append(box)
        else:
            overlapping_boxes.append(box)

    if len(overlapping_boxes) == 0:
        return independent_boxes
    else:
        pairs_idxs = combinations(range(len(overlapping_boxes)), 2)
        merge_groups: List[List[int]] = []
        for i, j in pairs_idxs:
            found_group = False
            if overlapping_boxes[i].overlaps(overlapping_boxes[j]):
                for merge_group in merge_groups:
                    if i in merge_group:
                        merge_group.append(j)
                        found_group = True
                        continue
                    if j in merge_group:
                        merge_group.append(i)
                        found_group = True
                        continue
                if not found_group:
                    merge_groups.append([i, j])

        for merge_group in merge_groups:
            idxs = set(merge_group)
            boxes = [overlapping_boxes[i] for i in idxs]
            merged_box = merge_all(boxes)
            independent_boxes.append(merged_box)

        return merge_overlapping(independent_boxes)


def merge_two(boxA: Box, boxB: Box) -> Box:
    """Merges two boxes and returns the merged box. Confidence from the first box is used.

    Args:
        boxA (Box): First box
        boxB (Box): Second box

    Returns:
        Box: Resulting merged box
    """
    a_x0 = boxA.cx - boxA.w / 2
    a_x1 = boxA.cx + boxA.w / 2
    a_y0 = boxA.cy - boxA.h / 2
    a_y1 = boxA.cy + boxA.h / 2

    b_x0 = boxB.cx - boxB.w / 2
    b_x1 = boxB.cx + boxB.w / 2
    b_y0 = boxB.cy - boxB.h / 2
    b_y1 = boxB.cy + boxB.h / 2

    new_x0 = min(a_x0, b_x0)
    new_x1 = max(a_x1, b_x1)
    new_y0 = min(a_y0, b_y0)
    new_y1 = max(a_y1, b_y1)

    new_w = new_x1 - new_x0
    new_cx = float(np.mean([new_x1, new_x0]))
    new_h = new_y1 - new_y0
    new_cy = float(np.mean([new_y1, new_y0]))

    return Box([boxA.conf, new_cx, new_cy, new_w, new_h])


def merge_all(boxes: List[Box]) -> Box:
    """Merges all boxes in the list and returns the merged box. Confidence from the
    first box is used.

    Args:
        boxes (List[Box]): List of boxes to merge

    Returns:
        Box: Resulting merged box
    """
    if len(boxes) <= 1:
        return boxes[0]
    else:
        return merge_two(boxes[0], merge_all(boxes[1:]))
