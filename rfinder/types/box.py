from typing import List, Optional


class Box:
    """Class to represent a bounding box"""

    def __init__(self, box: List[float]) -> None:
        """
        Instantiates box using either a list of valuees
        Parameters:
            box: lists of floats
                Box values as [conf, cx, cy, w, h]
        Returns:
            None
        """
        assert box is not None, (
            "Need to instantiate with either box param or all of conf, cx, cy, w, h"
            " params"
        )

        if box is not None:
            self._update_box(box)

    def _update_box(self, box: List[float]) -> None:
        """
        Update entire box from list
        Parameters:
            box: list of floats
                Box values as [conf, cx, cy, w, h]
        Returns:
            None
        """
        self.conf, self.cx, self.cy, self.w, self.h = box

    def isAboveThreshold(self, threshold: float) -> bool:
        """
        Boolean for whether confidence is above or below configured threshold.
        Parameters:
            threshold: float
                Value to compare box confidence with
        Returns:
            boolean
                True if confidence > threshold else False
        """
        return self.conf > threshold

    def shift(self, x: Optional[float] = None, y: Optional[float] = None) -> "Box":
        """
        Shift box position in x and/or y
        Parameters:
            x: float
                Value to shift in x
            y: float
                Value to shift in y
        Returns:
            None
        """
        if x is not None:
            self.cx += x
        if y is not None:
            self.cy += y

        return self

    def scale(
        self,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        w: Optional[float] = None,
        h: Optional[float] = None,
    ) -> "Box":
        """
        Scales box dimensions and position
        Parameters:
            cx: float
                Value to scale cx by
            cy: float
                Value to scale cy by
            w: float
                Value to scale w by
            h: float
                Value to scale h by
        Returns:
            None
        """
        if cx is not None:
            self.cx *= cx
        if cy is not None:
            self.cy *= cy
        if w is not None:
            self.w *= w
        if h is not None:
            self.h *= h

        return self

    def as_list(self) -> List[float]:
        """
        Returns box elements as a list
        """
        return [self.conf, self.cx, self.cy, self.w, self.h]

    def area(self) -> float:
        """
        Returns the area of the box
        """
        if self.w < 0 and self.h < 0:
            return 0
        return self.w * self.h

    def merge(self, merge_box: "Box") -> "Box":
        """
        Merges the box with a passed in box.
        """
        my_x0 = self.cx - self.w / 2
        my_x1 = self.cx + self.w / 2
        my_y0 = self.cy - self.h / 2
        my_y1 = self.cy + self.h / 2

        other_x0 = merge_box.cx - merge_box.w / 2
        other_x1 = merge_box.cx + merge_box.w / 2
        other_y0 = merge_box.cy - merge_box.h / 2
        other_y1 = merge_box.cy + merge_box.h / 2

        new_x0 = min(my_x0, other_x0)
        new_x1 = max(my_x1, other_x1)
        new_y0 = min(my_y0, other_y0)
        new_y1 = max(my_y1, other_y1)

        self.w = new_x1 - new_x0
        self.cx = new_x0 + self.w / 2
        self.h = new_y1 - new_y0
        self.cy = new_y0 + self.h / 2

        return self
