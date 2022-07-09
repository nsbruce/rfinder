from rfinder.types import Box


def IOU(a: Box, b: Box) -> float:
    """
    Returns the intersection over union for two boxes.
    Parameters:
        a: rt.Box
        b: rt.Box
    Returns:
        iou: float
    """

    I_box = a.get_intersection(b)

    if I_box.w <= 0 or I_box.h <= 0:  # if boxes don't intersect
        return 0.0

    intersection = I_box.area()

    union = a.area() + b.area() - intersection

    return intersection / union
