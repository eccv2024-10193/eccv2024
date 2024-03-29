from __future__ import absolute_import

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import ZeroPadding3D

def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format

def spatial_2d_padding(x, padding=((1, 1), (1, 1)), mode='REFLECT', data_format=None):
    """ Pads the 2nd and 3rd dimensions of a 4D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 4D tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [[0, 0],
               list(padding[0]),
               list(padding[1]),
               [0, 0]]
    # pattern = K.transpose_shape(pattern, data_format, spatial_axes=(1, 2))
    return tf.pad(x, pattern, mode)


class ReflectionPadding2D(ZeroPadding2D):

    def call(self, inputs):
        return spatial_2d_padding(inputs,
                                  padding=self.padding,
                                  data_format=self.data_format)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), mode='REFLECT', data_format=None):
    """ Pads the 2nd, 3rd and 4th dimensions of a 5D tensor.

    # Arguments
        x: Tensor or variable.
        padding: Tuple of 3 tuples, padding pattern.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A padded 5D tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    assert len(padding) == 3
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert len(padding[2]) == 2
    data_format = normalize_data_format(data_format)

    pattern = [[0, 0],
               list(padding[0]),
               list(padding[1]),
               list(padding[2]),
               [0, 0]]
    # pattern = K.transpose_shape(pattern, data_format, spatial_axes=(1, 2, 3))
    return tf.pad(x, pattern, mode)


class ReflectionPadding3D(ZeroPadding3D):

    def call(self, inputs):
        return spatial_3d_padding(inputs,
                                  padding=self.padding,
                                  data_format=self.data_format)
