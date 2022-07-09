from typing import List, Optional

import matplotlib.patches as patch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from rfinder.types import Box


def add_center_dot_patch(
    ax: plt.Axes, boxes: List[Box], color: Optional[str] = "black"
) -> None:
    """
    Adds dot on center of bounding boxes to existing axes
    Parameters:
        ax: matplotlib.pyplot.Axes
            An axes object on which to plot
        boxes: list of rt.Box objects
        color: string [default: 'black']
            Color to draw the dot in
    Returns:
        None
    """
    for box in boxes:
        ax.add_patch(patch.Circle((box.cx, box.cy), 0.5, ec=color, fc=color))


def add_rect_patch(
    ax: plt.Axes, boxes: List[Box], color: Optional[str] = "black", **kwargs: int
) -> None:
    """
    Adds bounding boxes to existing axes
    Parameters:
        ax: matplotlib.pyplot.Axes
            An axes object on which to plot
        boxes: list of rt.Box objects
        color: string [default: 'black']
            Color to draw the dot in
    Returns:
        None
    """
    for box in boxes:
        conf, cx, cy, w, h = box.as_list()
        ax.add_patch(
            patch.Rectangle(
                (cx - w / 2, cy - h / 2),
                height=h,
                width=w,
                ec=color,
                fc="none",
                **kwargs,
            )
        )
