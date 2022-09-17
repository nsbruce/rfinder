import tensorflow as tf  # type:ignore
from tensorboard.plugins.hparams import api as hp  # type:ignore

from rfinder.net import Network
from rfinder.train.blobs import generate_training_set


def tune() -> None:
    HP_NUM_UNITS = hp.HParam("num_units", hp.Discrete([32, 64]))  # , 128, 256, 512]))
    HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.25, 0.5))
    METRIC_ACCURACY = "accuracy"

    with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
        )

    all_boxes, all_pixels = generate_training_set(1000)

    session = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
            }
            run_name = f"run-{session}"
            print(f"--- Starting trial: {run_name}")
            print({h.name: hparams[h] for h in hparams})

            net = Network()

            net.model = net.build_model(
                num_units=hparams[HP_NUM_UNITS], dropout=hparams[HP_DROPOUT]
            )
            net.compile()
            net.train(tiles=all_pixels, boxes=all_boxes, num_epochs=1, hparams=hparams)
            session += 1


def main() -> None:
    tune()
