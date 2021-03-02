from model.backbones import blaze_backbone
import tensorflow as tf


class Blazeface():
    def __init__(self, backbone, input_dim=(128, 128, 3)):
        self.anchor_num = [2, 6]
        self.cell_size = [16, 8]
        self.input_dim = input_dim
        self.backbone = backbone(input_dim)

    def build_model(self):
        backbone = self.backbone
        feat_map_0, feat_map_1 = backbone.output

        x_0 = tf.keras.layers.Conv2D(filters=self.anchor_num[0], kernel_size=3,
                                     padding='same', activation='sigmoid')(feat_map_0)
        x_0 = tf.keras.layers.Reshape(
            [self.cell_size[0]**2 * self.anchor_num[0], 1])(x_0)

        x_1 = tf.keras.layers.Conv2D(filters=self.anchor_num[1], kernel_size=3,
                                     padding='same', activation='sigmoid')(feat_map_1)
        x_1 = tf.keras.layers.Reshape(
            [self.cell_size[1]**2 * self.anchor_num[1], 1])(x_1)
        confidences = tf.concat([x_0, x_1], axis=1)

        x_2 = tf.keras.layers.Conv2D(filters=self.anchor_num[0] * 4, kernel_size=3,
                                     padding='same')(feat_map_0)
        x_2 = tf.keras.layers.Reshape(
            [self.cell_size[0]**2 * self.anchor_num[0], 4])(x_2)

        x_3 = tf.keras.layers.Conv2D(filters=self.anchor_num[1] * 4, kernel_size=3,
                                     padding='same')(feat_map_1)
        x_3 = tf.keras.layers.Reshape(
            [self.cell_size[1]**2 * self.anchor_num[1], 4])(x_3)
        bboxes = tf.concat([x_2, x_3], axis=1)

        predictions = tf.concat([confidences, bboxes], axis=-1)
        return tf.keras.models.Model(backbone.inputs, predictions)


class BlazefaceFPN():
    def __init__(self, backbone, input_dim=(128, 128, 3)):
        self.anchor_num = [2, 6]
        self.cell_size = [16, 8]
        self.input_dim = input_dim
        self.backbone = backbone(input_dim)

    def build_model(self):
        backbone = self.backbone

        feat_map_0, feat_map_1 = backbone.output
        feat_map_0 = tf.keras.layers.Conv2D(96, 1, padding='same')(feat_map_0)
        feat_map_1 = tf.keras.layers.Conv2D(96, 1, padding='same')(feat_map_1)

        feat_map_1_upscaled = tf.keras.layers.UpSampling2D()(feat_map_1)
        feat_map_0 = feat_map_0 + feat_map_1_upscaled

        x_0 = tf.keras.layers.Conv2D(filters=self.anchor_num[0], kernel_size=3,
                                     padding='same', activation='sigmoid')(feat_map_0)
        x_0 = tf.keras.layers.Reshape(
            [self.cell_size[0]**2 * self.anchor_num[0], 1])(x_0)

        x_1 = tf.keras.layers.Conv2D(filters=self.anchor_num[1], kernel_size=3,
                                     padding='same', activation='sigmoid')(feat_map_1)
        x_1 = tf.keras.layers.Reshape(
            [self.cell_size[1]**2 * self.anchor_num[1], 1])(x_1)
        confidences = tf.concat([x_0, x_1], axis=1)

        x_2 = tf.keras.layers.Conv2D(filters=self.anchor_num[0] * 4, kernel_size=3,
                                     padding='same')(feat_map_0)
        x_2 = tf.keras.layers.Reshape(
            [self.cell_size[0]**2 * self.anchor_num[0], 4])(x_2)

        x_3 = tf.keras.layers.Conv2D(filters=self.anchor_num[1] * 4, kernel_size=3,
                                     padding='same')(feat_map_1)
        x_3 = tf.keras.layers.Reshape(
            [self.cell_size[1]**2 * self.anchor_num[1], 4])(x_3)
        bboxes = tf.concat([x_2, x_3], axis=1)

        predictions = tf.concat([confidences, bboxes], axis=-1)
        return tf.keras.models.Model(backbone.inputs, predictions)
