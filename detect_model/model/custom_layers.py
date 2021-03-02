import tensorflow as tf


def ChannelPad(x, pad_num):
    assert pad_num >= 0
    if pad_num == 0:
        return x
    padding = tf.zeros_like(x)
    padding = padding[..., :pad_num]
    return tf.concat([x, padding], axis=-1)


def BlazeBlock(x, filters_1, filters_2=0, kernel_size=5,
               strides=1, padding='same'):
    assert strides in [1, 2]
    activation = tf.keras.layers.Activation(tf.nn.relu)
    use_pool = (strides == 2)
    use_double_block = (filters_2 != 0)

    dw_conv_1 = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(
            filters=filters_1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        ),
        tf.keras.layers.BatchNormalization()
    ])
    if use_double_block:
        dw_conv_2 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                filters=filters_2,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
            ),
            tf.keras.layers.BatchNormalization()
        ])

    x_0 = dw_conv_1(x)

    if use_double_block:
        x_0 = activation(x_0)
        x_0 = dw_conv_2(x_0)
    if use_pool:
        x = tf.keras.layers.MaxPool2D()(x)

    pad_num = x_0.shape[-1] - x.shape[-1]
    x = ChannelPad(x, pad_num)
    x_0 = tf.keras.layers.Add()([x_0, x])
    return activation(x_0)
