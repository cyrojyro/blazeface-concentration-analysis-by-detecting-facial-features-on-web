from utils.utils import calc_iou_batch, normalize_image
from utils.augmentation import *
from PIL import Image
import tensorflow as tf
import numpy as np
import sys
import os


def read_image(image_path, target_w, target_h):
    image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='rgb', target_size=(target_h, target_w),
        interpolation='bilinear')
    return np.asarray(image)


def print_progress(num):
    sys.stdout.write('\rProcessing: ' + str(num))
    sys.stdout.flush()
    return num + 1


def load_widerface(gt_dirs, train_dir, target_w, target_h,
                   min_face_ratio=0.04, filter_entire_img=True, max_size=150000):
    """
    loads widerface dataset from directory. filter out images with small faces.\n
    returns: [num_picture, image_width, image_height, 3], [num_picture, num_gt, 4]\n
    4(cx, cy, w, h - normalized(0 ~ 1) location relative to resized image)
    """
    images, labels = [], []
    print('Processing dataset...')
    for gt_dir in gt_dirs:
        print('working on ' + gt_dir)
        with open(gt_dir, 'r') as f:
            process_num = 1
            while True:
                if(len(images) >= max_size):
                    break

                process_num = print_progress(process_num)
                image_name = f.readline().strip("\n ")

                # break if end of line
                if not image_name:
                    break

                num_bbox = int(f.readline())

                # continue if no label exists in image
                if num_bbox == 0:
                    f.readline()
                    continue

                # get size of image
                image_path = os.path.join(train_dir, image_name)
                image = Image.open(image_path)
                image_w, image_h = image.size
                image.close()

                # scale label relative to image size
                label = []
                filter_flag = False
                for bbox in range(num_bbox):
                    gt_str = f.readline().strip('\n ').split(' ')
                    gt = [int(i) for i in gt_str]
                    gt[0], gt[2] = (gt[0] + gt[2] / 2) / \
                        image_w,  gt[2] * 1.0 / image_w
                    gt[1], gt[3] = (gt[1] + gt[3] / 2) / \
                        image_h,  gt[3] * 1.0 / image_h

                    # filter out invalid or small gt boxes
                    # no heavy blur, occlusion, atypical pose
                    if (gt[2] * gt[3] > min_face_ratio and gt[4] == 0
                            and gt[7] != 1 and gt[8] == 0 and gt[9] != 1):
                        label.append(gt[:4])
                    else:
                        filter_flag = True

                if filter_flag and filter_entire_img:
                    continue

                image = read_image(image_path, target_w, target_h)

                if len(label) > 0:
                    images.append(image)
                    labels.append(label)

            print('\nLoaded:', len(images))

    return images, labels


def generate_gt(labels, anchors, iou_threshold=0.5, verbose=False):
    """
    labels: [num_labels, num_gt, 4]]\n
    anchors: [num_box, 4(cx, cy, w, h)]\n
    returns: [num_labels, num_boxes, 5(responsibility, tcx, tcy, tw, th)]
    """
    process_num = 0
    num_boxes = anchors.shape[0]
    gts = []

    for label in labels:
        if verbose:
            process_num = print_progress(process_num)

        gt = np.zeros([num_boxes, 5], dtype=np.float32)
        for box in label:
            ious = np.array(calc_iou_batch(box, anchors), dtype=np.float32)
            # Match higher than threshold
            maxarg = ious > iou_threshold
            # Match best jaccard(IOU) overlap
            maxarg[tf.argmax(ious)] = True
            # translate to target
            gt[maxarg, 0] = 1.0
            gt[maxarg, 1:3] = (box[0:2] - anchors[maxarg, 0:2]
                               ) / anchors[maxarg, 2:4]
            gt[maxarg, 3:5] = np.log(box[2:4] / anchors[maxarg, 2:4])
        gts.append(gt)

    if verbose:
        print('\nLoaded:', len(gts))
    return gts


def dataloader(images, labels, anchors, batch_size=64, augment=True):
    """
    images: [num_images, image_width, image_height, 3]\n
    labels: [num_labels, num_gt, 4(tcx, tcy, tw, th)]\n
    returns: ([batch_size, image_width, image_height, 3],
              [batch_size, num_boxes, 5(conf, tcx, tcy, tw, th)])
    """
    data_keys = np.arange(len(images))
    while True:
        selected_keys = np.random.choice(
            data_keys, replace=False, size=batch_size)

        image_batch = []
        label_batch = []
        for key in selected_keys:
            image = np.array(images[key], dtype=np.float32)
            label = np.array(labels[key], dtype=np.float32)

            # do augmentation
            if augment:
                image, label = random_flip(image, label)
                image, label = random_rotate(image, label)
                image = random_brightness(image)

            image = np.array(image, dtype=np.float32)
            image = normalize_image(image)
            image_batch.append(image)
            label_batch.append(label)

        gt_batch = generate_gt(label_batch, anchors)
        yield (np.array(image_batch, dtype=np.float32),
               np.array(gt_batch, dtype=np.float32))
