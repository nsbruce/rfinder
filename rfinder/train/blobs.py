from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage.filters import gaussian_filter  # type:ignore

from rfinder.environment import load_env
from rfinder.types import Box
from rfinder.utils.merging import merge_overlapping

env = load_env()


def generate_training_set(N: int) -> Tuple[List[List[Box]], List[npt.NDArray[np.float_]]]:
    """Generates training dataset full of blobs

    Args:
        N (int): The number of tiles to generate

    Returns:
        List[Tuple[Box, np.NDArray[np.float_]]]: A list of tuples, each containing a
        list of bounding boxes and a numpy array of the tile's "pixels"
    """

    min_blob_dim = 4
    max_blob_dim = tile_dim = int(env["TILE_DIM"])

    all_boxes = []
    all_pixels = []

    for _ in range(N):
        boxes: List[Box] = []
        pixels = np.random.normal(0, 0.2, (tile_dim, tile_dim))

        for _ in range(np.random.randint(0, int(env["MAX_BLOBS_PER_TILE"]) + 1)):

            blob_height, blob_width = np.random.randint(
                min_blob_dim, max_blob_dim + 1, size=2
            )  # length/width of the source

            # generate coordinates of the lowest left corner
            min_x, min_y = np.random.randint(0, tile_dim + 1, size=2)

            # add the rest of the source
            max_x = min(min_x + blob_width, tile_dim - 1)
            max_y = min(min_y + blob_height, tile_dim - 1)

            if max_x - min_x < min_blob_dim or max_y - min_y < min_blob_dim:
                continue

            pixels[min_x:max_x, min_y:max_y] = 1

            # create the label
            blob_box = Box(
                [
                    1.0,  # conf
                    np.mean([max_x, min_x])-0.5,  # cx
                    np.mean([max_y, min_y])-0.5,  # cy
                    max_x - min_x,  # w
                    max_y - min_y,  # h
                ]
            )
            boxes.append(blob_box)

        pixels = np.transpose(pixels)
        pixels = gaussian_filter(pixels, 0.8)

        boxes = merge_overlapping(boxes)

        all_boxes.append(boxes)
        all_pixels.append(pixels)

    assert len(all_boxes) == len(all_pixels), "Number of boxes and pixels must match"

    return all_boxes, all_pixels
