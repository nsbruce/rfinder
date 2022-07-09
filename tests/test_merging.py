from rfinder.types import Box
from rfinder.utils.merging import merge_overlapping


def test_two_independent_boxes() -> None:
    boxes = [Box([1.0, 2, 2, 2, 2]), Box([1.0, 30, 30, 2, 2])]

    merged = merge_overlapping(boxes)
    assert boxes == merged, "These should not have been merged but were"


def test_two_overlapping_boxes() -> None:
    boxes = [Box([1.0, 15, 15, 9, 9]), Box([1.0, 17, 17, 9, 9])]

    merged = merge_overlapping(boxes)

    assert (
        merged[0].as_list() == Box([1.0, 16, 16, 11, 11]).as_list()
    ), "Merged result is not correct"


def test_three_overlapping_boxes_1() -> None:
    boxes = [Box([1, 8, 8, 2, 2]), Box([1, 24, 24, 2, 2]), Box([1, 16, 16, 16, 16])]
    expected = [Box([1, 16, 16, 18, 18])]

    merged = merge_overlapping(boxes)

    assert merged[0].as_list() == expected[0].as_list(), "Merged result is not correct"


def test_three_overlapping_boxes_2() -> None:
    boxes = [
        Box([1, 16, 17, 10, 10]),
        Box([1, 16, 15, 10, 10]),
        Box([1, 16, 16, 10, 10]),
    ]
    expected = [Box([1, 16, 16, 10, 12])]

    merged = merge_overlapping(boxes)

    assert merged[0].as_list() == expected[0].as_list(), "Merged result is not correct"
