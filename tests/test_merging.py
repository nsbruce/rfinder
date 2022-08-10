from typing import Callable, List

import pytest

from rfinder.types import Box
from rfinder.utils.merging import merge_overlapping, merge_via_rtree

merge_functions = [merge_overlapping, merge_via_rtree]

TMergeFunction = Callable[[List[Box]], List[Box]]


@pytest.mark.parametrize("merge_function", merge_functions)
class TestMerging:
    def test_two_independent_boxes(self, merge_function: TMergeFunction) -> None:
        boxes = [Box([1.0, 2, 2, 2, 2]), Box([1.0, 30, 30, 2, 2])]

        merged: List[Box] = merge_function(boxes)

        assert len(merged) == len(boxes), (
            f"Merged result is not the correct length. Expected length {len(boxes)} but"
            f" got {len(merged)}"
        )

        flattened_boxes = [b.as_list() for b in boxes].sort()
        merged_boxes = [b.as_list() for b in merged].sort()

        assert (
            flattened_boxes == merged_boxes
        ), "Nothing should have been changed but something was"

    def test_two_overlapping_boxes(self, merge_function: TMergeFunction) -> None:
        boxes = [Box([1.0, 15, 15, 9, 9]), Box([1.0, 17, 17, 9, 9])]

        merged = merge_function(boxes)

        assert (
            merged[0].as_list() == Box([1.0, 16, 16, 11, 11]).as_list()
        ), "Merged result is not correct"

    def test_three_overlapping_boxes_1(self, merge_function: TMergeFunction) -> None:
        boxes = [Box([1, 8, 8, 2, 2]), Box([1, 24, 24, 2, 2]), Box([1, 16, 16, 16, 16])]
        expected = [Box([1, 16, 16, 18, 18])]

        merged = merge_function(boxes)

        assert (
            merged[0].as_list() == expected[0].as_list()
        ), "Merged result is not correct"

    def test_three_overlapping_boxes_2(self, merge_function: TMergeFunction) -> None:
        boxes = [
            Box([1, 16, 17, 10, 10]),
            Box([1, 16, 15, 10, 10]),
            Box([1, 16, 16, 10, 10]),
        ]
        expected = [Box([1, 16, 16, 10, 12])]

        merged = merge_function(boxes)

        assert (
            merged[0].as_list() == expected[0].as_list()
        ), "Merged result is not correct"

    def test_four_recursively_overlapping_boxes(
        self, merge_function: TMergeFunction
    ) -> None:
        boxes = [
            Box([1, 4, 16, 4, 22]),
            Box([1, 9, 6, 10, 4]),
            Box([1, 14, 3, 4, 4]),
            Box([1, 16, 19, 8, 4]),
        ]
        expected = [Box([1, 11, 14, 18, 26])]

        merged = merge_function(boxes)

        assert (
            merged[0].as_list() == expected[0].as_list()
        ), "Merged result is not correct"
