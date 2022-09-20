import shutil
from pathlib import Path

# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
import keras_tuner
import numpy as np
import tensorflow as tf  # type:ignore
from keras.layers import Activation, Dense, Dropout, Input  # type:ignore
from keras.models import Sequential, load_model  # type:ignore
from tensorboard.plugins.hparams import api as hp  # type:ignore

from rfinder.environment import load_env
from rfinder.net import Network
from rfinder.net.losses import normalized_sqrt_err
from rfinder.net.utils import (
    filter_preds,
    postprocess_preds,
    prepare_tiles,
    preprocess_boxes,
)
from rfinder.train.blobs import generate_training_set


def tune() -> None:
    if Path("logs/hparam_tuning").exists():
        shutil.rmtree("logs/hparam_tuning")

    HP_NUM_UNITS = hp.HParam("num_units", hp.Discrete([32, 64, 128, 256, 512]))
    HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5]))
    METRIC_ACCURACY = "accuracy"

    with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
        )

    train_boxes, train_pixels = generate_training_set(50000)
    test_boxes, test_pixels = generate_training_set(1000)

    session = 0

    with open("logs/nick.logs", "w") as f:
        for num_units in HP_NUM_UNITS.domain.values:
            for dropout_rate in HP_DROPOUT.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                }
                run_name = f"run-{session}"
                f.write(run_name + "\n")
                print(f"--- Starting trial: {run_name}")
                run_hparams = {h.name: hparams[h] for h in hparams}
                f.write(str(run_hparams) + "\n")
                print(run_hparams)

                net = Network()

                net.model = net.build_model(
                    num_units=hparams[HP_NUM_UNITS], dropout=hparams[HP_DROPOUT]
                )
                net.compile()
                net.train(
                    tiles=train_pixels,
                    boxes=train_boxes,
                    num_epochs=50,
                    hparams=hparams,
                )

                loss, accuracy = net.evaluate(tiles=test_pixels, boxes=test_boxes)
                f.writelines([f"loss: {loss}" + "\n", f"accuracy: {accuracy}" + "\n\n"])

                with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
                    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
                session += 1


def tune2() -> None:
    # net = Network()
    env = load_env()
    boxes, tiles = generate_training_set(10000)

    def build_model(hp: hp) -> Sequential:
        model = Sequential(
            [
                Input(shape=(int(env["TILE_DIM"]) ** 2)),
                Dense(
                    units=hp.Choice("units", [32, 64, 128, 256, 512, 1024]),
                    activation=hp.Choice(
                        "activation", values=["relu", "sigmoid", "linear"]
                    ),
                ),
                Dropout(hp.Choice("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])),
                Dense(int(env["MAX_BLOBS_PER_TILE"]) * 5),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=normalized_sqrt_err,
        )
        return model

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=50,
    )

    Y = preprocess_boxes(boxes) if boxes else np.array([])
    X = prepare_tiles(tiles) if tiles else np.array([])
    split_idx = int(len(X) * 0.8)
    train_X, test_X = np.split(X, [split_idx])
    train_Y, test_Y = np.split(Y, [split_idx])

    tuner.search(train_X, train_Y, epochs=30, validation_data=(test_X, test_Y))
    tuner.results_summary(num_trials=3)


if __name__ == "__main__":
    tune2()
