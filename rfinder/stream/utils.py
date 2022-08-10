from typing import List, TypeVar

import numpy as np

from rfinder.types import Box

T = TypeVar("T")


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


def list_split(li: List[T], n: int) -> List[List[T]]:
    """A list version of numpy's array_split. Will try to split the input list into n
    equal sized lists if possible. If not, will return a list of lists of length n with
    the last list containing the remaining elements of the input list.

    Args:
        li (List[T]): List to split
        n (int): Number of desired output lists

    Returns:
        List[List[T]]: List of n lists
    """
    sublistLen = int(np.ceil(len(li) / n))
    list_of_sublists = []
    for i in range(n - 1):
        list_of_sublists.append(li[i * sublistLen : i * sublistLen + sublistLen])
    list_of_sublists.append(li[(n - 1) * sublistLen :])
    return list_of_sublists


def list_split_overlapping_frequency(
    li: List[Box], n: int, overlap: float
) -> List[List[T]]:

    # Sort list by center frequency
    li.sort(key=lambda x: x.cx)

    return []
