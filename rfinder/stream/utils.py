from typing import List

from rfinder.types import Box


def place_boxes(
    boxes: List[List[Box]],
    tile_dim: int,
    tile_overlap: int,
    channel_bw: float,
    f0: float,
    t_int: float,
    t_0: float,
    invert_y: bool = False,
) -> List[Box]:
    """Rescale and shift boxes to their correct location in time and frequency.

    Args:
        boxes (List[List[Box]]): List of bounding boxes for each tile
        tile_dim (int): Tile dimension in pixels
        tile_overlap (int): Tile overlap in pixels
        channel_bw (float): Channel bandwidth in Hz
        f0 (float): Start frequency of the waterfall in Hz
        t_int (float): Integration time in some float unit
        t_0 (float): Start time of the waterfall in the same float unit as t_int
        invert_y (bool): Whether to invert the y-axis

    Returns:
        List[Box]: List of bounding boxes for the waterfall
    """

    output: List[Box] = []

    for i, tile in enumerate(boxes):
        for box in tile:
            # Scale box in frequency
            box.scale(cx_scale=channel_bw, w_scale=channel_bw)
            # Shift box in frequency
            box.shift(x=f0 + i * (tile_dim - tile_overlap) * channel_bw)
            if invert_y:
                box.cy = tile_dim - box.cy
            # Scale box in time
            box.scale(cy_scale=t_int, h_scale=t_int)
            # Shift box in time
            box.shift(y=t_0)

            output.append(box)

    return output
