"""Runtime-compatible port of the pinned author's FfiHades (Apache-2.0).

Source: RAMjET commit afe6f6cf8f0b96106f3eec48f68a1d7f1f0aefb9.
Changes: explicit numerical defaults, shape validation, named layers, and a
separate raw-logit accessor. No dense-head or mask-channel adaptation is made.
See METHODS.md and THIRD_PARTY_NOTICES.md.
"""
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np


def _leaky_relu():
    # Keras 3 renamed alpha to negative_slope; the numerical value is unchanged.
    if tuple(int(v) for v in tf.__version__.split('.')[:2]) >= (2, 16):
        return keras.layers.LeakyReLU(negative_slope=0.01)
    return keras.layers.LeakyReLU(alpha=0.01)


@keras.utils.register_keras_serializable(package="cnn_replication")
class LightCurveNetworkBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, pooling_size, dropout_rate=0.1,
                 batch_normalization=True, spatial=True, **kwargs):
        super().__init__(**kwargs)
        self.settings = dict(filters=filters, kernel_size=kernel_size,
                             pooling_size=pooling_size, dropout_rate=dropout_rate,
                             batch_normalization=batch_normalization, spatial=spatial)
        self.convolution = keras.layers.Conv1D(
            filters, kernel_size, strides=1, padding="valid", data_format="channels_last",
            dilation_rate=1, groups=1, use_bias=True,
            activation=_leaky_relu(),
            kernel_initializer="glorot_uniform", bias_initializer="zeros", name="convolution")
        self.dropout = None
        if dropout_rate > 0:
            cls = keras.layers.SpatialDropout1D if spatial else keras.layers.Dropout
            self.dropout = cls(dropout_rate, name="dropout")
        self.max_pooling = (keras.layers.MaxPooling1D(pooling_size, strides=pooling_size,
                            padding="valid", data_format="channels_last", name="pool")
                            if pooling_size > 1 else None)
        self.batch_normalization = None
        self.batch_normalization_input_reshape = None
        self.batch_normalization_output_reshape = None
        if batch_normalization:
            self.batch_normalization = keras.layers.BatchNormalization(
                axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=False,
                beta_initializer="zeros", moving_mean_initializer="zeros",
                moving_variance_initializer="ones", name="bn")
            if not spatial:
                self.batch_normalization_input_reshape = keras.layers.Reshape((-1,))
                self.batch_normalization_output_reshape = keras.layers.Reshape((-1, filters))

    def call(self, inputs, training=False):
        x = self.convolution(inputs)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        if self.max_pooling is not None:
            x = self.max_pooling(x)
        if self.batch_normalization is not None:
            if self.batch_normalization_input_reshape is not None:
                x = self.batch_normalization_input_reshape(x)
            x = self.batch_normalization(x, training=training)
            if self.batch_normalization_output_reshape is not None:
                x = self.batch_normalization_output_reshape(x)
        return x

    def get_config(self):
        return dict(super().get_config(), **self.settings)


@keras.utils.register_keras_serializable(package="cnn_replication")
class FfiHades(keras.Model):
    """One flux channel, exactly 1,000 positions, and the original convolutional head."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, filters in enumerate((8, 8, 16, 32, 64, 128, 128, 128, 20)):
            setattr(self, "block%d" % i, LightCurveNetworkBlock(
                filters, 3, 2 if i < 6 else 1,
                dropout_rate=0 if i == 0 else 0.1,
                batch_normalization=i != 0, spatial=i != 8, name="block%d" % i))
        self.block9 = LightCurveNetworkBlock(20, 7, 1, name="block9")
        self.block10 = LightCurveNetworkBlock(20, 1, 1, dropout_rate=0,
                                             batch_normalization=False, name="block10")
        self.prediction_layer = keras.layers.Conv1D(
            1, 1, activation=tf.sigmoid, padding="valid", strides=1,
            kernel_initializer="glorot_uniform", bias_initializer="zeros", name="prediction")
        self.reshape = keras.layers.Reshape((1,))

    @property
    def blocks(self):
        return tuple(getattr(self, "block%d" % i) for i in range(11))

    def features(self, inputs, training=False):
        x = tf.ensure_shape(tf.convert_to_tensor(inputs, dtype=tf.float32), (None, 1000, 1))
        tf.debugging.assert_all_finite(x, "Nonfinite CNN input")
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def call(self, inputs, training=False):
        return self.reshape(self.prediction_layer(self.features(inputs, training=training)))

    def logits(self, inputs, training=False):
        """Compute the pre-sigmoid tensor directly, without inverting probabilities.

        This is a separate forward pass. Scientific scoring must use training=False;
        two training-mode calls would draw different dropout masks and update BN twice.
        """
        x = self.features(inputs, training=training)
        z = tf.nn.conv1d(x, self.prediction_layer.kernel, stride=1, padding="VALID")
        z = tf.nn.bias_add(z, self.prediction_layer.bias)
        return self.reshape(z)


def build_model(seed=2026092801):
    if hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)
    else:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    model = FfiHades(name="ffi_hades_replication")
    model(tf.zeros((1, 1000, 1)), training=False)
    return model


def build_optimizer():
    return keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999,
                                 epsilon=1e-7, amsgrad=False)


def source_loss():
    """Author's probability-output BCE. Logit-based reporting is a separate diagnostic."""
    return keras.losses.BinaryCrossentropy(from_logits=False)
