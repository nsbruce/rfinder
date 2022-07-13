from typing import List
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np
import numpy.typing as npt

from rfinder.environment import load_env
from rfinder.types import Box

class Network():
    """Network to predict bounding boxes from 2D images
    """
    def __init__(self) -> None:
        self.env = load_env()

        self.model = Sequential([
            Dense(256, input_dim=((self.env["TILE_DIM"] * self.env["TILE_DIM"]))),
            Activation('relu'),
            Dropout(0.5),
            Dense(4)
        ])

        self.model.compile('adadelta', 'mse')


    def prepare_tiles(self, tiles: List[npt.NDArray[np.float_]]) -> List[npt.NDArray[np.float_]]:
        """Takes list of 2D tiles and flattens to input into the network

        Args:
            pixels (List[npt.NDArray[np.float_]]): 2D tiles

        Returns:
            List[npt.NDArray[np.float_]]: Flattened tiles
        """
        return np.array(tiles).reshape(len(tiles, -1))
    
    def prepare_boxes(self, boxes: List[List[Box]]) -> List[List[np.float_]]:
        """Takes a list of boxes and turns it into a flattened array of homogenous
        dimension that the network expects

        Args:
            boxes (List[List[Box]]): Bounding boxes

        Returns:
            List[List[np.float_]]: Bounding boxes as flattened lists
        """
        output = []
        for i, tile in enumerate(boxes):
            output.append(np.zeros((len(tile) * 4)))
            for j, box in enumerate(tile):
                output[i][j*4:j*4+4] = box.as_list()[1:]
        return output
