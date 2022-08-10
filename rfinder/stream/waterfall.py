import time
from multiprocessing import Pool
from os import cpu_count
from typing import List

import numpy as np
import numpy.typing as npt

# from matplotlib import pyplot as plt
from skimage.util import view_as_windows  # type: ignore

# import rfinder.plot as rplt
from rfinder.environment import load_env
from rfinder.net.Network import Network
from rfinder.stream.utils import list_split, place_boxes
from rfinder.types import Box
from rfinder.utils.merging import merge_via_rtree


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
        self.prebuffer_idx: int = self.tile_dim - 1
        self.prebuffer = np.zeros((self.tile_dim, num_chans))
        # self.prediction_input = np.zeros_like(self.prebuffer)
        self.prebuffer_timestamps = np.zeros(self.tile_dim)

        #  Results
        self.closedDetections: List[Box] = []
        self.onGoingDetections: List[Box] = []

        # Processes
        _ = cpu_count()
        self.num_processes: int = _ if _ is not None else 1

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
        self.prebuffer[self.prebuffer_idx, :] = integration
        self.prebuffer_timestamps[self.prebuffer_idx] = timestamp
        self.prebuffer_idx -= 1

        if self.prebuffer_idx == -1:
            self.prebuffer_idx += self.tile_overlap
            prediction_input = self.prebuffer
            prediction_t0 = self.prebuffer_timestamps[-1]
            self.prebuffer = np.roll(self.prebuffer, self.tile_overlap, axis=0)
            self.prebuffer_timestamps = np.roll(
                self.prebuffer_timestamps, self.tile_overlap
            )
            self.__update_buffer__(prediction_input, prediction_t0)

    def __update_buffer__(
        self, prediction_input: npt.NDArray[np.float_], prediction_t0: float
    ) -> None:

        START = time.time()

        # Predict
        predictions = self.__predict__(prediction_input)
        print("predicted", time.time() - START)
        print("onGoingDetections", len(self.onGoingDetections))

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
                t_0=prediction_t0,
            )
        )
        print("placed tiles", time.time() - START)
        print("onGoingDetections", len(self.onGoingDetections))
        # sort detections
        self.onGoingDetections.sort(key=lambda x: x.cx)
        print("sorted by cx", time.time() - START)

        # merge boxes in onGoingDetections
        print(
            "merging boxes",
            len(self.onGoingDetections),
            "detections",
            time.time() - START,
        )

        with Pool(self.num_processes) as pool:
            # split detections into ~evenly sized chunks for parallel processing
            split_detections = list_split(self.onGoingDetections, self.num_processes)

            # empty onGoingDetections so it can be filled with the merged detections
            self.onGoingDetections = []
            print("onGoingDetections", len(self.onGoingDetections))

            print("len(split_detections)", len(split_detections))

            for li in split_detections:
                print("merging", len(li))

            some_merged = pool.map(merge_via_rtree, split_detections)
            print("len(some_merged)", len(some_merged))

            for li in some_merged:
                print("merged", len(li))

            boundary_size = 5  # number of boxes at each end of list to keep
            boundary_detections: List[
                Box
            ] = []  # store merged detections near each boundary

            # get leftmost results
            for i in range(1, len(some_merged)):
                # sort results by left edge
                some_merged[i].sort(key=lambda x: x.cx-x.w/2)
                boundary_detections.extend(some_merged[i][:boundary_size])
                del some_merged[i][:boundary_size]

                if i == 0:
                    # The rest of the first slice can be dumped into onGoingDetections
                    self.onGoingDetections.extend(some_merged[i])
            
            # get rightmost results
            for i in range(len(some_merged)-1):
                # sort results by right edge
                some_merged[i].sort(key=lambda x: x.cx+x.w/2)
                boundary_detections.extend(some_merged[i][-boundary_size:])
                del some_merged[i][-boundary_size:]

                self.onGoingDetections.extend(some_merged[i])
            
            print("len(boundary_detections)", len(boundary_detections))
            print("len(onGoingDetections)", len(self.onGoingDetections))

            self.onGoingDetections.extend(merge_via_rtree(boundary_detections))


            # # first slice has no left boundary
            # self.onGoingDetections.extend(some_merged[0][:-boundary_size])
            # boundary_detections.append(
            #     some_merged[0][-boundary_size:] + some_merged[1][:boundary_size]
            # )

            # # other slices have two boundaries
            # for i in range(1, len(some_merged) - 2):
            #     self.onGoingDetections.extend(
            #         some_merged[i][boundary_size:-boundary_size]
            #     )
            #     boundary_detections.append(
            #         some_merged[i][-boundary_size:] + some_merged[i - 1][:boundary_size]
            #     )

            # # last slice has no right boundary
            # self.onGoingDetections.extend(some_merged[-1][boundary_size:])

            # for li in boundary_detections:
            #     print("merging 2", len(li))

            # rest_merged = pool.map(merge_via_rtree, boundary_detections)
            # for li in rest_merged:
            #     print("merged 2", len(li))
            #     self.onGoingDetections.extend(li)

        print(
            "merged boxes",
            len(self.onGoingDetections),
            "detections",
            time.time() - START,
        )

        # filter closed boxes from onGoingDetections into closedDetections
        self.__sort_detections__(prediction_t0 + self.t_int * self.tile_overlap)
        print("sorted detections", time.time() - START)

    def __predict__(
        self,
        prediction_input: npt.NDArray[np.float_],
    ) -> List[List[Box]]:

        # tile array: since the array is 2D, the windowing function fxpects to tile in
        # 2 directions but in our case only one fits. Hence we collapse the result into
        # the first dimension with [0].
        tiles = view_as_windows(
            arr_in=prediction_input,
            window_shape=(self.tile_dim, self.tile_dim),
            step=self.tile_overlap,
        )[0]

        tiles = [tiles[i, :, :] for i in range(tiles.shape[0])]

        # predict and get boxes
        return self.net.predict(tiles)

    def __sort_detections__(self, t_end: float) -> None:
        """Consider all boxes in onGoingDetections and if their end time is before the
        t_end argument, move the box into closedDetections.

        Args:
            t_end (float): The time after which a box can no longer be merged with.
        """
        for box in self.onGoingDetections:
            if box.cy + box.h / 2 < t_end:
                self.onGoingDetections.remove(box)
                self.closedDetections.append(box)
