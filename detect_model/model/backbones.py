from model.custom_layers import BlazeBlock
import tensorflow as tf


def blaze_backbone(input_dim):
    inputs = tf.keras.layers.Input(shape=input_dim)
    x_0 = tf.keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = tf.keras.layers.BatchNormalization()(x_0)
    x_0 = tf.keras.layers.Activation(tf.nn.relu)(x_0)

    x_0 = BlazeBlock(x_0, filters_1=24)
    x_0 = BlazeBlock(x_0, filters_1=24)
    x_0 = BlazeBlock(x_0, filters_1=24, strides=2)
    x_0 = BlazeBlock(x_0, filters_1=48)
    x_0 = BlazeBlock(x_0, filters_1=48)

    x_0 = BlazeBlock(x_0, filters_1=24, filters_2=96, strides=2)
    x_0 = BlazeBlock(x_0, filters_1=24, filters_2=96)
    x_0 = BlazeBlock(x_0, filters_1=24, filters_2=96)
    x_1 = BlazeBlock(x_0, filters_1=24, filters_2=96, strides=2)
    x_1 = BlazeBlock(x_1, filters_1=24, filters_2=96)
    x_1 = BlazeBlock(x_1, filters_1=24, filters_2=96)
    return tf.keras.models.Model(inputs, [x_0, x_1])


def blaze_mediapipe_backbone(input_dim):
    inputs = tf.keras.layers.Input(shape=input_dim)
    x_0 = tf.keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = tf.keras.layers.BatchNormalization()(x_0)
    x_0 = tf.keras.layers.Activation(tf.nn.relu)(x_0)

    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=24)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=28)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=32, strides=2)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=36)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=40)

    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=48, strides=2)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=56)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=64)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=72)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=80)
    x_0 = BlazeBlock(x_0, kernel_size=3, filters_1=88)
    x_1 = BlazeBlock(x_0, kernel_size=3, filters_1=96, strides=2)
    x_1 = BlazeBlock(x_1, kernel_size=3, filters_1=96)
    x_1 = BlazeBlock(x_1, kernel_size=3, filters_1=96)
    x_1 = BlazeBlock(x_1, kernel_size=3, filters_1=96)
    x_1 = BlazeBlock(x_1, kernel_size=3, filters_1=96)
    return tf.keras.models.Model(inputs, [x_0, x_1])
