import tensorflow_addons as tfa
import tensorflow as tf
import random
import math


def random_flip(image, labels, prob=0.5):
    if random.random() < prob:
        image = tf.image.flip_left_right(image)
        for label in labels:
            label[0] = 1.0 - label[0]
    return image, labels


def random_rotate(image, labels, max_factor=math.pi/4, prob=0.5):
    if random.random() < prob:
        factor = (random.random() - 0.5) * 2 * max_factor
        # counter-clockwise
        image = tfa.image.rotate(image, -factor, 'BILINEAR')
        for label in labels:
            norm = math.sqrt((label[0] - 0.5) ** 2.0 + (label[1] - 0.5) ** 2.0)
            angle = math.atan2(label[1] - 0.5, label[0] - 0.5)
            angle = angle + factor

            label[0] = norm * math.cos(angle) + 0.5
            label[1] = norm * math.sin(angle) + 0.5
            label[2:3] = label[2:3] * math.sin(math.pi/4 + abs(factor)) * 1.414
    return image, labels


def random_brightness(image, max_factor=0.25, prob=0.5):
    if random.random() < prob:
        factor = (random.random() - 0.5) * 2 * max_factor
        image = tf.image.adjust_brightness(image, factor)
    return image
