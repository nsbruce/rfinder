import pickle
from pathlib import Path
from typing import List, Tuple
from warnings import warn

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter  # type:ignore

from rfinder.environment import load_env
from rfinder.net import Network
from rfinder.types import Box
from rfinder.utils.merging import merge_via_rtree

env = load_env()


def generate_training_set(
    N: int,
) -> Tuple[List[List[Box]], List[npt.NDArray[np.float_]]]:
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
            min_x = np.random.randint(0, tile_dim - blob_width + 1)
            min_y = np.random.randint(0, tile_dim - blob_height + 1)

            # add the rest of the source
            max_x = min_x + blob_width
            max_y = min_y + blob_height

            # TODO using np.random.uniform(1, 5) drastically increases loss
            # ? first index is "rows", second is "columns"
            pixels[min_y:max_y, min_x:max_x] = 1

            # create the label such that 0,0 is bottom left for y
            blob_box = Box(
                [
                    1.0,  # conf
                    np.mean([max_x, min_x]) - 0.5,  # cx
                    tile_dim - np.mean([max_y, min_y]) - 0.5,  # cy
                    max_x - min_x,  # w
                    max_y - min_y,  # h
                ]
            )
            boxes.append(blob_box)

        # pixels = np.transpose(pixels)
        pixels = gaussian_filter(pixels, 0.8)
        pixels = np.clip(pixels, 0, 5)

        boxes = merge_via_rtree(boxes)

        all_boxes.append(boxes)
        all_pixels.append(pixels)

    assert len(all_boxes) == len(all_pixels), "Number of boxes and pixels must match"

    return all_boxes, all_pixels


def generate_blob_balanced_training_set(
    N: int,
) -> Tuple[List[List[Box]], List[npt.NDArray[np.float_]]]:
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

    num_blobs_counter = np.zeros(int(env["MAX_BLOBS_PER_TILE"]) + 1)
    tiles_per_blob_count = N / (int(env["MAX_BLOBS_PER_TILE"]) + 1)

    if not tiles_per_blob_count.is_integer():
        warn(
            "N is not divisible by the max number of blobs per tile (remember 0!). The"
            " data set will not be perfectly balanced."
        )

    print(f"Tiles per blob count: {tiles_per_blob_count}")

    # for _ in range(N):
    while min(num_blobs_counter) < tiles_per_blob_count:
        boxes: List[Box] = []
        pixels = np.random.normal(0, 0.2, (tile_dim, tile_dim))

        needed_blobs_count = []
        for i in range(len(num_blobs_counter)):
            if num_blobs_counter[i] < tiles_per_blob_count:
                needed_blobs_count.append(i)

        # for _ in range(np.random.randint(0, int(env["MAX_BLOBS_PER_TILE"]) + 1)):
        for _ in range(needed_blobs_count[-1]):
            blob_height, blob_width = np.random.randint(
                min_blob_dim, max_blob_dim + 1, size=2
            )  # length/width of the source

            # generate coordinates of the lowest left corner
            min_x = np.random.randint(0, tile_dim - blob_width + 1)
            min_y = np.random.randint(0, tile_dim - blob_height + 1)

            # add the rest of the source
            max_x = min_x + blob_width
            max_y = min_y + blob_height

            # TODO using np.random.uniform(1, 5) drastically increases loss
            # ? first index is "rows", second is "columns"
            pixels[min_y:max_y, min_x:max_x] = 1

            # create the label such that 0,0 is bottom left for y
            blob_box = Box(
                [
                    1.0,  # conf
                    np.mean([max_x, min_x]) - 0.5,  # cx
                    tile_dim - np.mean([max_y, min_y]) - 0.5,  # cy
                    max_x - min_x,  # w
                    max_y - min_y,  # h
                ]
            )
            boxes.append(blob_box)

        # pixels = np.transpose(pixels)
        pixels = gaussian_filter(pixels, 0.8)
        pixels = np.clip(pixels, 0, 5)

        boxes = merge_via_rtree(boxes)

        num_blobs = len(boxes)
        if num_blobs_counter[num_blobs] < tiles_per_blob_count:
            num_blobs_counter[num_blobs] += 1
            all_boxes.append(boxes)
            all_pixels.append(pixels)

        print("NUM BLOBS COUNTER", num_blobs_counter, end="\r")

    assert len(all_boxes) == len(all_pixels), "Number of boxes and pixels must match"
    print("NUM OF EACH BLOB COUNT", num_blobs_counter)
    print("TOTAL SIZE", len(all_boxes))

    return all_boxes, all_pixels


def main() -> None:
    boxes, pixels = generate_training_set(10000)
    # dset = Path(__file__).parent.parent.parent / "training_data" / "3maxblobs_80000tiles_balanced_dataset.pkl"
    # boxes, pixels = load_dataset(dset)

    net = Network()

    net.train(pixels, boxes, num_epochs=50)
    net.save()


def save_balanced_dataset() -> None:
    num_tiles = 80000
    boxes, pixels = generate_blob_balanced_training_set(num_tiles)

    save_dir = Path(__file__).parent.parent.parent / "training_data"
    if not save_dir.exists():
        save_dir.mkdir()

    fname = (
        "training_data/"
        + env["MAX_BLOBS_PER_TILE"]
        + "maxblobs_"
        + str(num_tiles)
        + "tiles_balanced_dataset.pkl"
    )

    with open(fname, "wb") as f:
        pickle.dump((boxes, pixels), f)


def load_dataset(file: Path) -> Tuple[List[List[Box]], List[npt.NDArray[np.float_]]]:
    with open(file, 'rb') as f:
        boxes, tiles = pickle.load(f)

    return boxes, tiles


if __name__ == "__main__":
    main()
