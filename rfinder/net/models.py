from keras.models import Sequential  # type:ignore
from keras.layers import Activation, Dense, Dropout, Input  # type:ignore

from rfinder.environment import load_env

env = load_env()


def single_blob_detector() -> Sequential:
    """Trains a network to detect a single blob"""
    # This works well for 32x32 tiles with 1 blob each
    return Sequential(
        [
            Input(
                shape=(
                    int(env["TILE_DIM"]) ** 2
                )  # , batch_size=self.batch_size
            ),
            Dense(1024),  # started with 256
            Activation("sigmoid"),  # started with 'relu
            Dropout(0.1),  # started with 0.25
            Dense(int(env["MAX_BLOBS_PER_TILE"]) * 5),
            Activation("sigmoid"), # everything should be between 0 and 1
        ]
    )


def build_model() -> Sequential:
    """Builds the model with the given hyperparameters

    Args:
        hparams (dict): Dictionary of hyperparameters

    Returns:
        None: Does not return anything
    """

    # return Sequential(
    #     [
    #         # TODO
    #         # Flatten(input_shape=(
    #         #     int(self.env["TILE_DIM"]), int(self.env["TILE_DIM"]))
    #         # ),
    #         Input(
    #             shape=(
    #                 int(self.env["TILE_DIM"]) ** 2
    #             )  # , batch_size=self.batch_size
    #         ),
    #         Dense(1024),  # started with 256
    #         Activation("relu"),  # started with 'relu
    #         # Dropout(0.1),  # started with 0.25
    #         Dense(256),  # started with 256
    #         Activation("relu"),  # started with 'relu
    #         # Dropout(0.25),  # started with 0.25
    #         Dense(64),
    #         Activation("relu"),
    #         Dense(int(self.env["MAX_BLOBS_PER_TILE"]) * 5),
    #         Activation("sigmoid"), # everything should be between 0 and 1
    #     ]
    # )


    # return Sequential(
    #     [
    #         Input(shape=(int(self.env["TILE_DIM"]) ** 2,)),
    #         Conv2d()
    #     ]
    # )
