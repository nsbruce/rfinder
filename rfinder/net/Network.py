from typing import List

import numpy as np
import numpy.typing as npt
from keras.layers import Activation, Dense, Dropout  # type:ignore
from keras.models import Sequential  # type:ignore
from keras.losses import Loss  # type:ignore

from rfinder.environment import load_env
from rfinder.net.utils import prepare_boxes, prepare_tiles
from rfinder.types import Box


class Network:
    """Network to predict bounding boxes from 2D images"""

    def __init__(self) -> None:
        self.env = load_env()

        self.model = Sequential(
            [
                Dense(256, input_dim=((int(self.env["TILE_DIM"]) ** 2))),
                Activation("relu"),
                Dropout(0.5),
                Dense(int(self.env["MAX_BLOBS_PER_TILE"]) * 4),
            ]
        )

        self.model.compile("adadelta", "mse")

    def train(
        self,
        tiles: List[npt.NDArray[np.float_]],
        boxes: List[List[Box]],
        num_epochs: int = 50000,
    ) -> None:
        """Trains the network on a set of tiles and bounding boxes

        Args:
            tiles (List[npt.NDArray[np.float_]]): List of 2d tiles
            boxes (List[List[Box]]): List of bounding boxes for each tile
            num_epochs (int): Number of epochs to train for (default: 50000)
        """
        all_Y = prepare_boxes(boxes)
        all_X = prepare_tiles(tiles)
        split_idx = int(len(all_X) * 0.8)
        train_X, test_X = np.split(all_X, [split_idx])
        train_Y, test_Y = np.split(all_Y, [split_idx])

        self.model.fit(
            train_X,
            train_Y,
            epochs=num_epochs,
            shuffle=True,
            validation_data=(test_X, test_Y),
        )


class CustomLoss(Loss):  # type:ignore
    """Loss function for bounding box IOU"""

    def call(self, y_true: npt.NDArray[np.float_], y_pred: npt.NDArray[np.float_]) -> float:

        return 0.0
