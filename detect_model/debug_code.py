from utils.widerface_loader import *
from utils.augmentation import *
from utils.utils import *
import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import sys
import os

GT_DIRS = ["/home/cyrojyro/hddrive/wider_face_split/fddb_gt.txt"]
DATA_DIR = "/home/cyrojyro/hddrive/WIDER_train/images"

img_width = 128
img_height = 128
parser = argparse.ArgumentParser(
    description="Debug, Testbed")
parser.add_argument('--model', type=str, required=True,
                    help='Model directory')


def load_widerface_dynamic(gt_dirs, train_dir, min_face_ratio=0.04,
                           filter_entire_img=True):
    """
    loads widerface dataset from directory. filter out images with small faces.\n
    returns: [num_picture] image_dirs, [num_picture, num_gt, 4]\n
    4(cx, cy, w, h - normalized(0 ~ 1) location relative to resized image)
    """
    images, labels = [], []
    print('Processing dataset...')
    for gt_dir in gt_dirs:
        with open(gt_dir, 'r') as f:
            process_num = 1
            while True:
                process_num = print_progress(process_num)
                image_name = f.readline().strip("\n ")

                # break if end of line
                if not image_name:
                    break

                num_bbox = int(f.readline())

                # continue if no gt exists in image
                if num_bbox == 0:
                    f.readline()
                    continue

                image_path = os.path.join(train_dir, image_name)
                image = Image.open(image_path)
                image_w, image_h = image.size

                # scale gt
                label = []
                filter_flag = False
                for bbox in range(num_bbox):
                    gt_str = f.readline().strip('\n ').split(' ')
                    gt = [int(i) for i in gt_str]
                    gt[0], gt[2] = (gt[0] + gt[2] / 2) / \
                        image_w,  gt[2] / image_w
                    gt[1], gt[3] = (gt[1] + gt[3] / 2) / \
                        image_h,  gt[3] / image_h

                    # filter out invalid or small gt boxes
                    if (gt[2] * gt[3] > min_face_ratio and gt[4] == 0
                            and gt[7] != 1 and gt[8] == 0 and gt[9] != 1):
                        label.append(gt[:4])
                    else:
                        filter_flag = True

                if filter_flag and filter_entire_img:
                    continue

                if len(label) > 0:
                    images.append(image_path)
                    labels.append(label)

            print('\nLoaded: ', len(images))

    return images, labels


def dataloader_dynamic(image_urls, labels, anchors, target_w, target_h, batch_size=64):
    """
    image_urls: [num_images] contains image url\n
    labels: [num_labels, num_gt, 4]]\n
    returns: ([batch_size, image_width, image_height, 3], [batch_size, num_boxes, 5])\n
    this function dynamically loads image from image_url, and make gt from labels.
    """
    data_keys = np.arange(len(image_urls))
    while True:
        selected_keys = np.random.choice(
            data_keys, replace=False, size=batch_size)

        image_batch = []
        label_batch = []
        for key in selected_keys:
            image = read_image(image_urls[key], target_w, target_h)
            label = np.array(labels[key], dtype=np.float32)

            # do augmentation
            image, label = random_flip(image, label)
            image, label = random_rotate(image, label)
            image = random_brightness(image, prob=1)

            image = np.array(image, dtype=np.float32)
            image = np.array(image)
            image = normalize_image(image)

            image_batch.append(image)
            label_batch.append(label)

        gt_batch = generate_gt(label_batch, anchors)
        yield (np.array(image_batch, dtype=np.float32),
               np.array(gt_batch, dtype=np.float32))


def test_model_on_dataset(args):
    anchors = np.load(os.path.join("./", "anchors.npy"))
    model = tf.keras.models.load_model(
        args.model, compile=False)
    image_urls, labels = load_widerface_dynamic(GT_DIRS, DATA_DIR)
    dataloader = dataloader_dynamic(
        image_urls, labels, anchors, img_width, img_height)

    for _ in range(10):
        a, gt = next(dataloader)
        # test gt translation
        p = prediction_to_bbox(gt, anchors)
        for j in range(64):
            ress = p[j]
            ress = ress[ress[..., 0] == 1]

            prediction = model(np.expand_dims(a[j], 0))
            prediction = np.array(prediction, dtype=np.float32)
            bbox = prediction_to_bbox(prediction, anchors)[0]
            bbox = bbox[bbox[..., 0] > 0.5]
            resolved_boxes = tie_resolution(bbox, 0.5, 0.2)


if __name__ == "__main__":
    args = parser.parse_args()
    test_model_on_dataset(args)
