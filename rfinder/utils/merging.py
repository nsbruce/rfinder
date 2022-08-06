from itertools import combinations
from typing import List

import numpy as np
from rtree import index  # type: ignore

from rfinder.types import Box


def merge_overlapping(boxes: List[Box]) -> List[Box]:
    """Merges list of boxes when they overlap using an algorithm that Nick invented and
    works but not quickly.

    Args:
        boxes (List[Box]): boxes to attempt to merge

    Returns:
        List[Box]: non-overlapping boxes
    """
    if len(boxes) <= 1:
        return boxes

    independent_boxes: List[Box] = []
    overlapping_boxes: List[Box] = []
    print("Checking for overlapping boxes", len(boxes))
    for box in boxes:
        if not any(box.overlaps(other) for other in boxes if other != box):
            independent_boxes.append(box)
        else:
            overlapping_boxes.append(box)
    print("checked for overlaps", len(overlapping_boxes), len(independent_boxes))

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


# def merge_with_cv(boxes: List[Box]) -> List[Box]:
#     """Merges boxes using OpenCV. HOWEVER THIS TAKES AVERAGE INSTEAD OF EXTENT

#     Args:
#         boxes (List[Box]): List of boxes to merge

#     Returns:
#         List[Box]: Resulting merged boxes
#     """
#     arr = [box.as_list()[1:] for box in boxes]
#     merged = cv2.groupRectangles(arr, groupThreshold=1, eps=0.1)[0]
#     merged = np.insert(merged, 0, 1.0, axis=1)
#     print(merged)
#     result = []
#     for m in merged:
#         b = Box(m)
#         result.append(b)
#     return result


# def merge_via_sweep(boxes: List[Box]) -> List[Box]:
#     result = []
#     # arr = np.empty((len(boxes), 5))
#     # for i, box in enumerate(boxes):
#     #     arr[i] = box.as_list(limits=True)
#     # print("now have an arr instead of a list of boxes")
#     # print(arr)
#     # arr = arr[arr[:, 1].argsort()]
#     # print("sorted arr by x0")
#     # print(arr)

#     boxes.sort(key=lambda b: b.cx - b.w / 2)
#     # current_x = arr[0, 1]

# return []


def merge_via_rtree(boxes: List[Box]) -> List[Box]:
    """Merges list of boxes when they overlap using an rtree structure as described in
    https://stackoverflow.com/a/48565371/8844897.

    Args:
        boxes (List[Box]): boxes to attempt to merge

    Returns:
        List[Box]: non-overlapping boxes
    """
    idx = index.Index()
    unique_id = 0

    def insert_box(idx: index.Index, box: List[float], index: int) -> int:
        results = list(idx.intersection(box, objects=True))
        if len(results) == 0:
            idx.insert(index, box)
            return index + 1
        else:
            x0, y0, x1, y1 = box
            for r in results:
                rect = r.bbox
                idx.delete(r.id, rect)
                x0 = min(x0, rect[0])
                y0 = min(y0, rect[1])
                x1 = max(x1, rect[2])
                y1 = max(y1, rect[3])
            return insert_box(idx, [x0, y0, x1, y1], index)

    for b in boxes:
        limits = b.as_list(limits=True)[1:]
        unique_id = insert_box(idx, limits, unique_id)

    boxes = []
    full_bounds = idx.bounds
    all_bounded = list(idx.intersection(full_bounds, objects=True))
    for n in all_bounded:
        bounds = n.bbox
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        boxes.append(Box([1.0, cx, cy, w, h]))

    return boxes


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
