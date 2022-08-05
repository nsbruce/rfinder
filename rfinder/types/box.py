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
        cx_scale: Optional[float] = None,
        cy_scale: Optional[float] = None,
        w_scale: Optional[float] = None,
        h_scale: Optional[float] = None,
        all_scale: Optional[float] = None,
    ) -> "Box":
        """Scales box coordinates and dimensions

        Args:
            cx_scale (Optional[float], optional): Value to scale cx by. Defaults to
            None.
            cy_scale (Optional[float], optional): Value to scale cy by. Defaults to
            None.
            w_scale (Optional[float], optional): Value to scale w by. Defaults to None.
            h_scale (Optional[float], optional): Value to scale h by. Defaults to None.
            all_scale (Optional[float], optional): Value to scale everything by.
            Overrides other arguments. Defaults to None.

        Returns:
            Box: The scaled box
        """
        if all_scale is not None:
            cx_scale = cy_scale = w_scale = h_scale = all_scale
        if cx_scale is not None:
            self.cx *= cx_scale
        if cy_scale is not None:
            self.cy *= cy_scale
        if w_scale is not None:
            self.w *= w_scale
        if h_scale is not None:
            self.h *= h_scale

        return self

    def as_list(self, limits: bool = False) -> List[float]:
        """Returns box as a list. When `limits` argument is False (default), box is
        [conf, cx, cy, w, h] when `limits` argument is True, box is
        [conf, x_min, y_min, x_max, y_max].

        Args:
            limits (bool, optional): Whether to return limits instead of center
            coordinates and width/height. Defaults to False.

        Returns:
            List[float]: The box values.
        """
        if limits:
            return [
                self.conf,
                self.cx - self.w / 2,
                self.cy - self.h / 2,
                self.cx + self.w / 2,
                self.cy + self.h / 2,
            ]
        return [self.conf, self.cx, self.cy, self.w, self.h]

    def area(self) -> float:
        """
        Returns the area of the box
        """
        if self.w < 0 and self.h < 0:
            return 0
        return self.w * self.h

    def merge(self, merge_box: "Box") -> None:
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

        self._update_box(
            [
                self.conf,
                new_x0 + self.w / 2,
                new_y0 + self.h / 2,
                new_x1 - new_x0,
                new_y1 - new_y0,
            ]
        )

    def overlaps(self, other: "Box") -> bool:
        """Checks whether the box intersects a passed in box.

        Args:
            other (Box): Box to check for intersection

        Returns:
            bool: whether or not the two boxes intersect
        """

        return self.get_intersection(other).area() > 0

    def get_intersection(self, other: "Box") -> "Box":
        """Returns intersection box between two boxes

        Args:
            other (Box): Box to get intersection with
        Returns:
            Box: the intersection as a Box
        """

        x1_a = self.cx - self.w / 2
        x2_a = x1_a + self.w
        y1_a = self.cy - self.h / 2
        y2_a = y1_a + self.h

        x1_b = other.cx - other.w / 2
        x2_b = x1_b + other.w
        y1_b = other.cy - other.h / 2
        y2_b = y1_b + other.h

        x1_I = max(x1_a, x1_b)
        y1_I = max(y1_a, y1_b)
        x2_I = min(x2_a, x2_b)
        y2_I = min(y2_a, y2_b)

        w_I = x2_I - x1_I
        h_I = y2_I - y1_I

        I_box = Box([1, x1_I + w_I / 2, y1_I + h_I / 2, w_I, h_I])

        return I_box
