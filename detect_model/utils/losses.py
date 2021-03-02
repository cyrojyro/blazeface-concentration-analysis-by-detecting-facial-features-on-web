import tensorflow as tf


def smooth_l1_loss(true, pred):
    """
    true, pred: [num_batch, num_box, 5(confidence, cx, cy, w, h)]\n
    returns: [num_batch] loss, responsible anchor box only
    """
    abs_loss = tf.abs(true[..., 1:5] - pred[..., 1:5])
    sq_loss = 0.5 * (true[..., 1:5] - pred[..., 1:5])**2
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)

    sum_loss = tf.reduce_sum(l1_loss, -1)
    pos_loss = tf.reduce_sum(true[..., 0] * sum_loss, axis=1)

    return pos_loss


def cross_entrophy_loss(true, pred, num_neg):
    """
    true, pred: [num_batch, num_box] confidence\n
    returns: [num_batch] loss, negative hard mined (batch mean)
    """
    # prevent under/overflow
    pred = tf.maximum(tf.minimum(pred, 1 - 1e-15), 1e-15)
    neg_pred = tf.maximum(tf.minimum(1 - pred, 1 - 1e-15), 1e-15)
    conf_loss = -(true * tf.math.log(pred) +
                  (1 - true) * tf.math.log(neg_pred))
    pos_conf_loss = tf.reduce_sum(conf_loss * true, axis=-1)

    # use mean over batch
    num_neg_batch = tf.cast(tf.reduce_mean(num_neg), dtype=tf.int32)
    kth, _ = tf.math.top_k(pred * (1 - true), k=num_neg_batch)
    hard_mined = tf.where(pred >= tf.expand_dims(kth[..., -1], -1),
                          tf.ones_like(pred), tf.zeros_like(pred))
    neg_conf_loss = tf.reduce_sum(conf_loss * hard_mined * (1 - true), axis=-1)
    return pos_conf_loss, neg_conf_loss


def multiboxLoss(true, pred, hard_mining_ratio=3):
    num_box = tf.cast(tf.keras.backend.shape(true)[1], dtype=tf.float32)
    num_pos = tf.reduce_sum(true[..., 0], axis=-1)
    num_neg = tf.minimum(hard_mining_ratio * num_pos, num_box - num_pos)

    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                       tf.ones_like(num_pos))
    smooth_l1 = smooth_l1_loss(true, pred)
    ce_pos, ce_neg = cross_entrophy_loss(true[..., 0], pred[..., 0], num_neg)
    total_loss = smooth_l1 + ce_pos + ce_neg
    return total_loss / num_pos
