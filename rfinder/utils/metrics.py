import numpy as np

from rfinder.types import Box


def IOU(a: Box, b: Box) -> float:
    """Returns the intersection over union for two boxes.

    Args:
        a (Box): First box
        b (Box): Second box

    Returns:
        iou (float): Intersection over union of the two boxes
    """

    I_box = a.get_intersection(b)

    if I_box.w <= 0 or I_box.h <= 0:  # if boxes don't intersect
        return 0.0

    intersection = I_box.area()

    union = a.area() + b.area() - intersection

    return intersection / union


def center_distance(a: Box, b: Box) -> float:
    """
    Returns the distance between the centers of two boxes.

    Args:
        a (Box): First box
        b (Box): Second box

    Returns:
        distance (float): Distance between the centers of the two boxes
    """

    return float(np.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2))
