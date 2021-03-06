"""Loss functions."""

import tensorflow as tf
import semver
import numpy as np


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    abs_diff = tf.abs(y_pred - y_true)
    #if tf.less_equal(abs_diff, max_grad) is not None:
    #    return abs_diff**2 / 2
    #else:
    #    return max_grad * (abs_diff - max_grad / 2)
    loss = tf.where(abs_diff <= max_grad, abs_diff ** 2 / 2, max_grad * (abs_diff - max_grad / 2))
    return loss

def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad))
