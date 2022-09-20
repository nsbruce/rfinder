import sys

import tensorflow as tf  # type:ignore


def sqrt_err(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Loss function for bounding boxes which uses squared error

    Args:
        y_true (npt.NDArray[np.float_]): Thelabels of size (batch_size,
        max_num_boxes*5)
        y_pred (npt.NDArray[np.float_]): The predictions of size (batch_size,
        max_num_boxes*5)

    Returns:
        npt.NDArray[np.float_]: The loss for this batch with shape (batch_size, 1)
    """

    x = tf.square(y_pred - y_true)
    coords_err = tf.sqrt(x[:, 1::5] + x[:, 2::5])
    loss = tf.reduce_sum(tf.multiply(y_true[:, 0::5], coords_err))
    dims_err = tf.sqrt(x[:, 3::5] + x[:, 4::5])
    loss += tf.reduce_sum(tf.multiply(y_true[:, 0::5], dims_err))
    loss += tf.reduce_sum(x[:, 0::5])
    return loss


def normalized_sqrt_err(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Loss function for bounding boxes which uses squared error

    Args:
        y_true (npt.NDArray[np.float_]): The labels of size (batch_size,
        max_num_boxes*5)
        y_pred (npt.NDArray[np.float_]): The predictions of size (batch_size,
        max_num_boxes*5)

    Returns:
        npt.NDArray[np.float_]: The loss for this batch with shape (batch_size, 1)
    """

    # assert y_true.shape[1] == y_pred.shape[1], f"y_true must have the same label size but are instead {y_true.shape[1]} and {y_pred.shape[1]} respectively"
    # assert len(y_true[:,0]) == len(y_pred[:,0]), f"y_true and y_pred must have thes same batch dimension but are instead {len(y_true[:,0])} and {len(y_pred[:,0])} respectively"
    # TODO I think I have to assume that the batch size is the same for both y_true and y_pred despite being unable to assert it. When I print tensors they are the same size but when I inspect the shapes one (true I think) is None while the other is correctß

    # tf.print('t', y_true[:,0], output_stream=sys.stdout, summarize=-1)
    # tf.print('p', y_pred[:,0], output_stream=sys.stdout, summarize=-1)

    squared = tf.square(y_pred - y_true)
    coords_err = tf.sqrt(squared[:, 1::5] + squared[:, 2::5])
    loss = tf.reduce_sum(tf.multiply(y_true[:, 0::5], coords_err))

    pct = tf.math.divide_no_nan(tf.abs(y_pred - y_true), y_true)
    dims_err = pct[:, 3::5] + pct[:, 4::5]

    loss += tf.reduce_sum(tf.multiply(y_true[:, 0::5], dims_err))
    loss += tf.reduce_sum(squared[:, 0::5])
    return loss
