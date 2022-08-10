from rfinder.stream.utils import list_split, place_boxes
from rfinder.types import Box


def test_basic_box_placement() -> None:
    tile_dim = 32
    tile_overlap = 16
    channel_bw = 1.0
    f0 = 0.0
    t_int = 1.0
    t0 = 0.0

    boxes = [
        [
            Box([1.0, 16.0, 16.0, 4.0, 4.0]),
        ],
        [
            Box([1.0, 1.0, 16.0, 2.0, 4.0]),
        ],
        [
            Box([1.0, 24.0, 28.0, 12.0, 4.0]),
        ],
    ]
    expected = [
        Box([1.0, 16.0, 16.0, 4.0, 4.0]),
        Box([1.0, 17.0, 16.0, 2.0, 4.0]),
        Box([1.0, 56.0, 28.0, 12.0, 4.0]),
    ]

    placed = place_boxes(boxes, tile_dim, tile_overlap, channel_bw, f0, t_int, t0)

    expected_list = [box.as_list() for box in expected]
    placed_list = [box.as_list() for box in placed]

    assert placed_list == expected_list, "Placed boxes are not correct"


def test_basic_box_placement_inverted() -> None:
    tile_dim = 32
    tile_overlap = 16
    channel_bw = 1.0
    f0 = 0.0
    t_int = 1.0
    t0 = 0.0

    boxes = [
        [
            Box([1.0, 16, 16, 4.0, 4.0]),
        ],
        [
            Box([1.0, 1.0, 16.0, 2.0, 4.0]),
        ],
        [
            Box([1.0, 24.0, 28.0, 12.0, 4.0]),
        ],
    ]
    expected = [
        Box([1.0, 16.0, 16.0, 4.0, 4.0]),
        Box([1.0, 17.0, 16.0, 2.0, 4.0]),
        Box([1.0, 56.0, 4.0, 12.0, 4.0]),
    ]

    placed = place_boxes(
        boxes, tile_dim, tile_overlap, channel_bw, f0, t_int, t0, invert_y=True
    )

    expected_list = [box.as_list() for box in expected]
    placed_list = [box.as_list() for box in placed]

    assert placed_list == expected_list, "Placed boxes are not correct"


def test_scaled_box_placement() -> None:
    tile_dim = 32
    tile_overlap = 16
    channel_bw = 2.0
    f0 = 100.0
    t_int = 0.75
    t0 = 10.25

    boxes = [
        [
            Box([1.0, 16.0, 16.0, 4.0, 4.0]),
        ],
        [
            Box([1.0, 1.0, 16.0, 2.0, 4.0]),
        ],
        [
            Box([1.0, 24.0, 28.0, 12.0, 4.0]),
        ],
    ]
    expected = [
        Box([1.0, 132.0, 22.25, 8.0, 3.0]),
        Box([1.0, 134.0, 22.25, 4.0, 3.0]),
        Box([1.0, 212.0, 31.25, 24.0, 3.0]),
    ]

    placed = place_boxes(boxes, tile_dim, tile_overlap, channel_bw, f0, t_int, t0)

    expected_list = [box.as_list() for box in expected]
    placed_list = [box.as_list() for box in placed]

    assert placed_list == expected_list, "Placed boxes are not correct"


def test_scaled_box_placement_inverted() -> None:
    tile_dim = 32
    tile_overlap = 16
    channel_bw = 2.0
    f0 = 100.0
    t_int = 0.75
    t0 = 10.25

    boxes = [
        [
            Box([1.0, 16.0, 16.0, 4.0, 4.0]),
        ],
        [
            Box([1.0, 1.0, 16.0, 2.0, 4.0]),
        ],
        [
            Box([1.0, 24.0, 28.0, 12.0, 4.0]),
        ],
    ]
    expected = [
        Box([1.0, 132.0, 22.25, 8.0, 3.0]),
        Box([1.0, 134.0, 22.25, 4.0, 3.0]),
        Box([1.0, 212.0, 13.25, 24.0, 3.0]),
    ]

    placed = place_boxes(
        boxes, tile_dim, tile_overlap, channel_bw, f0, t_int, t0, invert_y=True
    )

    expected_list = [box.as_list() for box in expected]
    placed_list = [box.as_list() for box in placed]

    assert placed_list == expected_list, "Placed boxes are not correct"


def test_list_splitter() -> None:
    input = [1, 2, 3, 4, 5, 6, 7, 8]
    n = 3
    expected = [[1, 2, 3], [4, 5, 6], [7, 8]]

    assert list_split(input, n) == expected, "List split is not correct"
