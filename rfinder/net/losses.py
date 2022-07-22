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

    x = tf.square(y_pred-y_true)
    coords_err = tf.sqrt(x[:, 1::5] + x[:, 2::5])
    loss = tf.reduce_sum(tf.multiply(y_true[:, 0::5], coords_err))
    dims_err = tf.sqrt(x[:, 3::5] + x[:, 4::5])
    loss += tf.reduce_sum(tf.multiply(y_true[:, 0::5], dims_err))
    loss += tf.reduce_sum(x[:, 0::5])
    return loss


def normalized_sqrt_err(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Loss function for bounding boxes which uses squared error

    Args:
        y_true (npt.NDArray[np.float_]): Thelabels of size (batch_size,
        max_num_boxes*5)
        y_pred (npt.NDArray[np.float_]): The predictions of size (batch_size,
        max_num_boxes*5)

    Returns:
        npt.NDArray[np.float_]: The loss for this batch with shape (batch_size, 1)
    """

    squared = tf.square(y_pred - y_true)
    coords_err = tf.sqrt(squared[:, 1::5] + squared[:, 2::5])
    loss = tf.reduce_sum(tf.multiply(y_true[:, 0::5], coords_err))

    pct = tf.math.divide_no_nan(tf.abs(y_pred - y_true), y_true)
    dims_err = pct[:, 3::5] + pct[:, 4::5]

    loss += tf.reduce_sum(tf.multiply(y_true[:, 0::5], dims_err))
    loss += tf.reduce_sum(squared[:, 0::5])
    return loss


