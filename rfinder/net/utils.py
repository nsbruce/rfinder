from typing import List

import numpy as np
import numpy.typing as npt

from rfinder.environment import load_env
from rfinder.types import Box

env = load_env()


def prepare_tiles(tiles: List[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
    """Takes list of 2D tiles and flattens to input into the network

    Args:
        pixels (List[npt.NDArray[np.float_]]): 2D tiles

    Returns:
        List[npt.NDArray[np.float_]]: Flattened tiles
    """
    return np.array(tiles).reshape((len(tiles), -1))


def prepare_boxes(boxes: List[List[Box]]) -> npt.NDArray[np.float_]:
    """Takes a list of boxes and turns it into a flattened array of homogenous
    dimension that the network expects

    Args:
        boxes (List[List[Box]]): Bounding boxes

    Returns:
        List[List[np.float_]]: Bounding boxes as flattened lists
    """
    output = np.zeros((len(boxes), int(env["MAX_BLOBS_PER_TILE"]) * 4))
    for i, tile in enumerate(boxes):
        for j, box in enumerate(tile):
            output[i][j * 4 : j * 4 + 4] = box.as_list()[1:]
    return output / int(env["TILE_DIM"])
