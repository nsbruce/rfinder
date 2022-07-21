from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type:ignore
from keras.layers import Activation, Dense, Dropout, Input  # type:ignore
from keras.models import Sequential, load_model  # type:ignore

from rfinder.environment import load_env
from rfinder.net.utils import postprocess_preds, prepare_tiles, preprocess_boxes
from rfinder.types import Box


class Network:
    """Network to predict bounding boxes from 2D images"""

    def __init__(self) -> None:
        self.env = load_env()

        self.batch_size = 32

        self.model = Sequential(
            [
                Input(
                    shape=(int(self.env["TILE_DIM"]) ** 2), batch_size=self.batch_size
                ),
                Dense(256),
                # Dropout(0.25),
                # Dense(256),
                # Dropout(0.25),
                # Dense(256),
                Activation("relu"),
                Dropout(0.25),
                Dense(int(self.env["MAX_BLOBS_PER_TILE"]) * 5),
            ]
        )

        self.default_path = Path(__file__).parent.parent.parent / "models"
        self.default_name = "default"

        # lossy = MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.SUM)
        self.model.compile(
            optimizer='adam',
            loss=self.loss,
            # loss='mean_absolute_percentage_error'
            # loss=lossy,
        )

    def prepare(
        self,
        tiles: Optional[List[npt.NDArray[np.float_]]] = None,
        boxes: Optional[List[List[Box]]] = None,
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Convert the data from 2d tile and list of boxes format to flat numpy arrays

        Args:
            tiles (List[npt.NDArray[np.float_]]): List of 2d tiles
            boxes (List[List[Box]]): List of bounding boxes for each tile

        Returns:
            Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]: Tuple of homogenous
            sized numpy arrays
        """

        Y = preprocess_boxes(boxes) if boxes else np.array([])
        X = prepare_tiles(tiles) if tiles else np.array([])
        return Y, X

    def train(
        self,
        tiles: List[npt.NDArray[np.float_]],
        boxes: List[List[Box]],
        num_epochs: int = 50,
    ) -> None:
        """Trains the network on a set of tiles and bounding boxes

        Args:
            tiles (List[npt.NDArray[np.float_]]): List of 2d tiles
            boxes (List[List[Box]]): List of bounding boxes for each tile
            num_epochs (int): Number of epochs to train for (default: 50000)
        """
        all_Y, all_X = self.prepare(tiles, boxes)
        split_idx = int(len(all_X) * 0.8)
        train_X, test_X = np.split(all_X, [split_idx])
        train_Y, test_Y = np.split(all_Y, [split_idx])

        self.model.fit(
            train_X,
            train_Y,
            epochs=num_epochs,
            shuffle=True,
            validation_data=(test_X, test_Y),
            batch_size=self.batch_size,
        )

    def predict(
        self,
        tiles: List[npt.NDArray[np.float_]],
    ) -> List[List[Box]]:
        """Predict bounding boxes for a set of tiles

        Args:
            tiles (List[npt.NDArray[np.float_]]): List of 2d tiles

        Returns:
            List[List[Box]]: List of bounding boxes for each tile
        """
        _, all_X = self.prepare(tiles=tiles)
        predictions = self.model.predict(all_X)
        return postprocess_preds(predictions)

    def save(self, name: Optional[str] = None) -> None:
        """Save the model

        Returns:
            None: Does not return anything
        """
        if name is None:
            name = self.default_name

        if not self.default_path.exists():
            self.default_path.mkdir()

        self.model.save(self.default_path / name)

    def load(self, name: Optional[str] = None) -> None:
        """Load the model

        Returns:
            None: Does not return anything
        """
        if name is None:
            name = self.default_name

        if not self.default_path.exists():
            self.default_path.mkdir()

        self.model = load_model(
            self.default_path / name,
            custom_objects={'loss': self.loss}
        )

    def loss(self,  y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Custom loss function for bounding boxes which uses IOU, center-to-center
        distance, and MSE between confidences

        Args:
            y_true (npt.NDArray[np.float_]): Thelabels of size (batch_size,
            max_num_boxes*5)
            y_pred (npt.NDArray[np.float_]): The predictions of size (batch_size,
            max_num_boxes*5)

        Returns:
            npt.NDArray[np.float_]: The loss for this batch with shape (batch_size, 1)
        """

        # basic
        # loss = tf.math.divide_no_nan(tf.abs(y_pred - y_true), y_true)
        # loss = tf.math.reduce_sum(loss, axis=1)
        # return loss

        # confidence focused
        y_true = tf.reshape(y_true, (-1, int(self.env["MAX_BLOBS_PER_TILE"]), 5))
        y_pred = tf.reshape(y_pred, (-1, int(self.env["MAX_BLOBS_PER_TILE"]), 5))

        tmp1 = tf.square(y_pred[:, :, 1] - y_true[:, :, 1])
        tmp2 = tf.square(y_pred[:, :, 2] - y_true[:, :, 2])
        loss = tf.reduce_sum(tf.multiply(y_true[:, :, 0], tmp1+tmp2))

        tmp1 = tf.square(y_pred[:, :, 3] - y_true[:, :, 3])
        tmp2 = tf.square(y_pred[:, :, 4] - y_true[:, :, 4])
        loss += tf.reduce_sum(tf.multiply(y_true[:, :, 0], tmp1+tmp2))
        loss += tf.reduce_sum(tf.square(y_true[:, :, 0] - y_pred[:, :, 0]))

        return loss

        # initial
        # y_true = tf.reshape(y_true, (-1, int(self.env["MAX_BLOBS_PER_TILE"]), 5))
        # y_pred = tf.reshape(y_pred, (-1, int(self.env["MAX_BLOBS_PER_TILE"]), 5))

        # # loss defined by confidence
        # loss = tf.square(y_pred[:, :, 0] - y_true[:, :, 0])

        # # loss defined by center-to-center distance
        # loss += tf.sqrt(
        #     tf.square(y_pred[:, :, 1] - y_true[:, :, 1])
        #     +
        #     tf.square(y_pred[:, :, 2] - y_true[:, :, 2])
        # )

        # # loss defined by height
        # loss += tf.math.divide_no_nan(
        #         tf.abs(y_pred[:, :, 3] - y_true[:, :, 3]),
        #         y_true[:, :, 3]
        #     )

        # # loss defined by width
        # loss += tf.math.divide_no_nan(
        #         tf.abs(y_pred[:, :, 4] - y_true[:, :, 4]),
        #         y_true[:, :, 4]
        #     )

        # return loss
