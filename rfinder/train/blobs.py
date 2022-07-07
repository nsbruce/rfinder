from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage.filters import gaussian_filter  # type:ignore

from rfinder.environment import load_env
from rfinder.types import Box

env = load_env()


def generate_training_set(N: int) -> List[Tuple[List[Box], npt.NDArray[np.float_]]]:
    """Generates training dataset full of blobs

    Args:
        N (int): The number of tiles to generate

    Returns:
        List[Tuple[Box, np.NDArray[np.float_]]]: A list of tuples, each containing a list of bounding boxes and a
        numpy array of the tile's "pixels"
    """

    min_blob_dim = 4
    max_blob_dim = tile_dim = int(env["TILE_DIM"])

    result = []

    for n in range(N):
        boxes = []
        # TODO not sure whether to start with zeros or noise
        # pixels = np.zeros((tile_dim, tile_dim))
        pixels = np.random.normal(0, 0.2, (tile_dim, tile_dim))  # mean  # std dev

        for i in range(np.random.randint(0, int(env["MAX_BLOBS_PER_TILE"]) + 1)):
            # tmp = np.zeros((tile_dim, tile_dim))

            blob_length, blob_width = np.random.randint(
                min_blob_dim, max_blob_dim + 1, size=2
            )  # length/width of the source

            diag_ratio = np.random.randint(
                -6, 6
            )  # i.e a one to one ratio means perfectly diagonal

            # generate coordinates of the lowest left corner
            x, y = np.random.randint(0, tile_dim + 1, size=2)

            # add the rest of the source
            min_x = min_y = tile_dim
            max_x = max_y = 0
            for j in range(blob_length):
                for k in range(blob_width):
                    try:
                        pixels[x + j, y + k + (j // diag_ratio)] = 1
                        max_x = max(max_x, x + j)
                        min_x = min(min_x, x + j)
                        max_y = max(max_y, y + k + (j // diag_ratio))
                        min_y = min(min_y, y + k + (j // diag_ratio))
                    except IndexError:
                        pass

            # create the label
            boxes.append(
                Box(
                    [
                        1.0,  # conf
                        np.mean([max_x, min_x]),  # cx
                        np.mean([max_y, min_y]),  # cy
                        max_x - min_x,  # w
                        max_y - min_y,  # h
                    ]
                )
            )
        pixels = np.transpose(pixels)
        pixels = gaussian_filter(pixels, 0.8)
        result.append((boxes, pixels))

    return result
