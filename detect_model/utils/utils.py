import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import time


def drawplt(image, label, target_w, target_h):
    """
    image: [image_width, image_height, 3], numpy array\n
    label: [num_face, 4(cx, cy, w, h)]
    """
    fig, ax = plt.subplots(1)
    ax = plt.imshow(image)
    for l in label:
        rect = patches.Rectangle(((l[0] - l[2] / 2) * target_w, (l[1] - l[3] / 2) * target_h),
                                 l[2] * target_w, l[3] * target_h, linewidth=1,
                                 edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()


def normalize_image(image):
    return image / 127.5 - 1.0


def tie_resolution(prediction, threshold=0.3, match_iou=0.3):
    """
    prediction: [num_batch, num_box, 5(conf, cx, cy, w, h)]\n
    threshold: minimum confidence to preserve box\n
    match_iou: iou threshold of same object\n
    returns: [num_box, 4(cx, cy, w, h)]
    """
    resolved = np.empty([0, 4])
    saved_boxes = prediction[prediction[..., 0] > threshold]
    sorted = tf.argsort(saved_boxes[..., 0], axis=-1, direction="DESCENDING")
    used = np.zeros_like(sorted)

    for element in sorted:
        if(used[element]):
            continue
        ious = calc_iou_batch(saved_boxes[element, 1:5], saved_boxes[..., 1:5])
        selected = ious > match_iou
        used = np.logical_or(used, selected)

        overlapping_boxes = saved_boxes[np.ravel(selected)]
        weighted_boxes = np.expand_dims(overlapping_boxes[..., 0], -1) * \
            overlapping_boxes[..., 1:5] / np.sum(overlapping_boxes[..., 0])
        weighted_box = np.sum(weighted_boxes, axis=0)
        resolved = np.vstack([resolved, weighted_box])

    return resolved


def prediction_to_bbox(prediction, anchors):
    """
    prediction: [num_batch, num_box, 5(conf, cx, cy, w, h)]\n
    anchors: [num_box, 4(cx, cy, w, h)]\n
    returns: [num_batch, num_box, 5(conf, cx, cy, w, h)]
    """
    center = prediction[..., 1:3] * anchors[..., 2:4] + anchors[..., 0:2]
    width_height = anchors[..., 2:4] * tf.exp(prediction[..., 3:5])
    prediction[..., 1:3] = center
    prediction[..., 3:5] = width_height
    return prediction


def calc_iou_batch(box, batch):
    """
    box: [4(cx, cy, w, h)]\n
    batch: [num_box, 4(cx, cy, w, h)]\n
    returns: [num_box] calculated IOUs
    """
    tf.convert_to_tensor(batch)
    box_xmax = box[0] + (box[2] / 2)
    box_xmin = box[0] - (box[2] / 2)
    box_ymax = box[1] + (box[3] / 2)
    box_ymin = box[1] - (box[3] / 2)
    box_space = (box_ymax - box_ymin) * (box_xmax - box_xmin)

    batch_xmax = batch[..., 0] + (batch[..., 2] / 2)
    batch_xmin = batch[..., 0] - (batch[..., 2] / 2)
    batch_ymax = batch[..., 1] + (batch[..., 3] / 2)
    batch_ymin = batch[..., 1] - (batch[..., 3] / 2)
    batch_space = (batch_ymax - batch_ymin) * (batch_xmax - batch_xmin)

    i_xmin = tf.maximum(box_xmin, batch_xmin)
    i_xmax = tf.minimum(box_xmax, batch_xmax)
    i_ymin = tf.maximum(box_ymin, batch_ymin)
    i_ymax = tf.minimum(box_ymax, batch_ymax)

    i_w = tf.maximum(i_xmax - i_xmin, 0)
    i_h = tf.maximum(i_ymax - i_ymin, 0)
    intersection = i_w * i_h

    return intersection / (box_space + batch_space - intersection)


def calc_iou(box_1, box_2):
    box_1_xmax = box_1[0] + (box_1[2] / 2)
    box_1_xmin = box_1[0] - (box_1[2] / 2)
    box_1_ymax = box_1[1] + (box_1[3] / 2)
    box_1_ymin = box_1[1] - (box_1[3] / 2)
    box_1_space = (box_1_ymax - box_1_ymin) * (box_1_xmax - box_1_xmin)

    box_2_xmax = box_2[0] + (box_2[2] / 2)
    box_2_xmin = box_2[0] - (box_2[2] / 2)
    box_2_ymax = box_2[1] + (box_2[3] / 2)
    box_2_ymin = box_2[1] - (box_2[3] / 2)
    box_2_space = (box_2_ymax - box_2_ymin) * (box_2_xmax - box_2_xmin)

    i_xmin = max(box_1_xmin, box_2_xmin)
    i_xmax = min(box_1_xmax, box_2_xmax)
    i_ymin = max(box_1_ymin, box_2_ymin)
    i_ymax = min(box_1_ymax, box_2_ymax)

    i_w = max(i_xmax - i_xmin, 0)
    i_h = max(i_ymax - i_ymin, 0)
    intersection = i_w * i_h

    return intersection / (box_1_space + box_2_space - intersection)
