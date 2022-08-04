from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from skimage.util import view_as_windows  # type: ignore

import rfinder.plot as rplt
from rfinder.environment import load_env
from rfinder.net.Network import Network
from rfinder.stream.utils import place_boxes
from rfinder.types import Box
from rfinder.utils.merging import merge_overlapping


class WaterfallBuffer:
    def __init__(
        self, t_int: float, channel_bw: float, f0: float, num_chans: int
    ) -> None:

        #  Class arguments
        self.t_int = t_int
        self.channel_bw = channel_bw
        self.f0 = f0

        #  Environment variables
        self.env = load_env()
        self.tile_dim = int(self.env["TILE_DIM"])
        self.tile_overlap = int(self.env["TILE_OVERLAP"])
        self.boxes_per_tile = int(self.env["MAX_BLOBS_PER_TILE"])

        #  Predictor
        self.net = Network()
        self.net.load()

        #  Buffers
        self.prebuffer_idxs: int = 0
        self.prebuffer = np.zeros((self.tile_dim, num_chans))
        # self.prediction_input = np.zeros_like(self.prebuffer)
        self.prebuffer_timestamps = np.zeros(self.tile_dim)

        #  Results
        self.closedDetections: List[Box] = []
        self.onGoingDetections: List[Box] = []

    def getClosedDetections(self, clear: bool = False) -> List[Box]:
        """Return the closed detections and optionally clear them from RFInder.

        Args:
            clear (bool, optional): Whether to clear the list while returning the
            closed detections. Defaults to False.

        Returns:
            List[Box]: List of detections whos end time has passed and can no longer
            be merged with.
        """
        if clear:
            to_return = self.closedDetections
            self.closedDetections = []
            return to_return
        else:
            return self.closedDetections

    def add_integration(
        self, integration: npt.NDArray[np.float_], timestamp: float
    ) -> None:
        """# TODO update
        Add an integration to the RFInder buffer for prediction and triggers a
        prediction if the buffer is full.

        Args:
            integration (npt.NDArray[np.float_]): Integration to add.
        """
        # Add to buffer
        self.prebuffer[self.prebuffer_idxs, :] = integration
        self.prebuffer_timestamps[self.prebuffer_idxs] = timestamp
        self.prebuffer_idxs += 1

        if self.prebuffer_idxs == self.tile_dim:
            self.prebuffer_idxs -= self.tile_overlap
            prediction_input = self.prebuffer.copy()
            prediction_t0 = self.prebuffer_timestamps[0].copy()
            # prediction_timestamps = self.prebuffer_timestamps.copy()

            self.prebuffer = np.roll(self.prebuffer, -self.tile_overlap, axis=0)
            self.prebuffer_timestamps = np.roll(
                self.prebuffer_timestamps, -self.tile_overlap
            )

            # Predict
            self.predict(prediction_input, prediction_t0)

    def predict(
        self,
        prediction_input: npt.NDArray[np.float_],
        prediction_t0: float,
    ) -> None:

        # tile array: since the array is 2D, the windowing function fxpects to tile in
        # 2 directions but in our case only one fits. Hence we collapse the result into
        # the first dimension with [0].
        tiles = view_as_windows(
            arr_in=prediction_input,
            window_shape=(self.tile_dim, self.tile_dim),
            step=self.tile_overlap,
        )[0]

        tiles = [tiles[i, :, :] for i in range(tiles.shape[0])]

        # TODO check edge cases

        # predict and get boxes
        predictions = self.net.predict(tiles)
        # print("predictions in buffer ")
        for tile, preds in zip(tiles, predictions):
            fig, ax = plt.subplots()
            rplt.tile(ax, tile, preds)
            # for pred in preds:
            #     print(pred.as_list())
            plt.show(block=False)

        # add boxes to onGoingDetections
        self.onGoingDetections.extend(
            # put boxes into correct time-frequency location
            place_boxes(
                boxes=predictions,
                tile_dim=self.tile_dim,
                tile_overlap=self.tile_overlap,
                channel_bw=self.channel_bw,
                f0=self.f0,
                t_int=self.t_int,
                t0=prediction_t0,
            )
        )

        # merge boxes in onGoingDetections
        self.onGoingDetections = merge_overlapping(self.onGoingDetections)

        # filter closed boxes from onGoingDetections into closedDetections
        self.sort_detections(prediction_t0 + self.t_int * self.tile_overlap)

    def sort_detections(self, t_end: float) -> None:
        for box in self.onGoingDetections:
            if box.cy + box.h / 2 > t_end:
                self.closedDetections.append(box)
                self.onGoingDetections.remove(box)
