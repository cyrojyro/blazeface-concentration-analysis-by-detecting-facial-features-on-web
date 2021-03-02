from utils.utils import prediction_to_bbox
from utils.losses import *
import tensorflow as tf


def l_loss(true, pred, hard_mining_ratio=3):
    num_pos = tf.reduce_sum(true[..., 0], axis=-1)
    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                       tf.ones_like(num_pos))
    return smooth_l1_loss(true, pred) / num_pos


def cp_loss(true, pred, hard_mining_ratio=3):
    num_box = tf.cast(tf.keras.backend.shape(true)[1], dtype=tf.float32)
    num_pos = tf.reduce_sum(true[..., 0], axis=-1)
    num_neg = tf.minimum(hard_mining_ratio * num_pos, num_box - num_pos)

    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                       tf.ones_like(num_pos))
    ce_pos, ce_neg = cross_entrophy_loss(true[..., 0], pred[..., 0], num_neg)
    return ce_pos / num_pos


def cn_loss(true, pred, hard_mining_ratio=3):
    num_box = tf.cast(tf.keras.backend.shape(true)[1], dtype=tf.float32)
    num_pos = tf.reduce_sum(true[..., 0], axis=-1)
    num_neg = tf.minimum(hard_mining_ratio * num_pos, num_box - num_pos)

    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                       tf.ones_like(num_pos))
    ce_pos, ce_neg = cross_entrophy_loss(true[..., 0], pred[..., 0], num_neg)
    return ce_neg / num_pos
