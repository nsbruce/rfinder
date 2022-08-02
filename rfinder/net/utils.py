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


def flatten_boxes(boxes: List[List[Box]]) -> npt.NDArray[np.float_]:
    """Takes a list of boxes and turns it into a flattened array of homogenous
    dimension that the network expects

    Args:
        boxes (List[List[Box]]): Bounding boxes

    Returns:
        List[List[np.float_]]: Bounding boxes with shape (num_tiles,
        max_blobs_per_tile * 5)
    """
    output = np.zeros((len(boxes), int(env["MAX_BLOBS_PER_TILE"]) * 5))
    for i, tile in enumerate(boxes):
        for j, box in enumerate(tile):
            output[i][j * 5 : j * 5 + 5] = box.scale(
                all_scale=int(env["TILE_DIM"]) ** -1
            ).as_list()
    return output


def stack_boxes(boxes: List[List[Box]]) -> npt.NDArray[np.float_]:
    """Takes a list of boxes and turns it into a higher dimensional array that the
    network expects

    Args:
        boxes (List[List[Box]]): Bounding boxes

    Returns:
        List[List[np.float_]]: Bounding boxes with shape (num_tiles,
        max_blobs_per_tile, 5)
    """
    output = np.zeros((len(boxes), int(env["MAX_BLOBS_PER_TILE"]), 5))
    for i, tile in enumerate(boxes):
        for j, box in enumerate(tile):
            output[i, j, :] = box.scale(all_scale=int(env["TILE_DIM"]) ** -1).as_list()
    return output


def flattened_predictions_to_boxes(
    predictions: npt.NDArray[np.float_],
) -> List[List[Box]]:
    """Takes network output predictions and turns them into a list of boxes for each tile

    Args:
        predictions (npt.NDArray[np.float_]): Network output predictions

    Returns:
        List[List[Box]]: List of bounding boxes for each tile
    """
    boxes: List[List[Box]] = []
    for i in range(predictions.shape[0]):
        boxes.append([])
        for j in range(predictions.shape[1] // 5):
            boxes[i].append(
                Box(predictions[i][j * 5 : j * 5 + 5]).scale(
                    all_scale=int(env["TILE_DIM"])
                )
            )
    return boxes


def stacked_predictions_to_boxes(
    predictions: npt.NDArray[np.float_],
) -> List[List[Box]]:
    """Takes network output predictions and turns them into a list of boxes for each tile

    Args:
        predictions (npt.NDArray[np.float_]): Network output predictions

    Returns:
        List[List[Box]]: List of bounding boxes for each tile
    """
    boxes: List[List[Box]] = []
    for i in range(predictions.shape[0]):
        boxes.append([])
        for j in range(predictions.shape[1]):
            boxes[i].append(
                Box(predictions[i, j, :].tolist()).scale(all_scale=int(env["TILE_DIM"]))
            )
    return boxes


def filter_preds(
    predictions: List[List[Box]], minimum_confidence: float
) -> List[List[Box]]:
    """Takes a list of boxes and filters out boxes with confidence less than that provided

    Args:
        predictions (List[List[Box]]): Bounding boxes
        minimum_confidence (float): Minimum confidence to keep a box

    Returns:
        List[List[Box]]: Filtered bounding boxes
    """
    return [
        list(filter(lambda box: box.conf >= minimum_confidence, tile))
        for tile in predictions
    ]


preprocess_boxes = flatten_boxes
postprocess_preds = flattened_predictions_to_boxes
