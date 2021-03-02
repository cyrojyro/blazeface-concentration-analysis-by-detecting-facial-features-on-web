from utils.widerface_loader import read_image
from utils.utils import *
import tensorflow as tf
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser(
    description="Test model using arbitrary image")
parser.add_argument('--width', type=int, default=128,
                    help='Target image width')
parser.add_argument('--height', type=int, default=128,
                    help='Target image height')
parser.add_argument('--anchor', type=str, default='./',
                    help='Path of generated anchors')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold of detection')
parser.add_argument('--tie_threshold', type=float, default=0.2,
                    help='Tie threshold of predicted boxes')
parser.add_argument('--image_dir', type=str, default='./sample_images',
                    help='Image directory')
parser.add_argument('--cpu', action="store_true",
                    help="Use CPU for inference")
parser.add_argument('--model', type=str, required=True,
                    help='Model directory')

if __name__ == "__main__":
    """
    inference on every image in image_dir
    """
    args = parser.parse_args()
    anchors = np.load(os.path.join(args.anchor, "anchors.npy"))

    model = tf.keras.models.load_model(
        args.model, compile=False)

    physical_devices = tf.config.list_physical_devices('GPU')
    num_gpu = len(physical_devices)
    print("Available GPUs:", num_gpu)

    device = '/CPU:0' if (args.cpu or num_gpu == 0) else '/GPU:0'
    print('using ' + device)

    with tf.device(device):
        for file_name in os.listdir(args.image_dir):
            f = os.path.join(args.image_dir, file_name)
            image = read_image(os.path.join(f), args.width, args.height)
            image = read_image(
                os.path.join(f), args.width, args.height)
            image_normalized = normalize_image(image)
            image_normalized = tf.convert_to_tensor(image_normalized)
            image_normalized = tf.expand_dims(image_normalized, 0)

            t1 = time.time()
            prediction = model(image_normalized)
            prediction = np.array(prediction, dtype=np.float32)
            bbox = prediction_to_bbox(prediction, anchors)[0]
            bbox = bbox[bbox[..., 0] > args.threshold]
            resolved_boxes = tie_resolution(
                bbox, args.threshold, args.tie_threshold)
            print("time taken(ms): ", (time.time() - t1)*1000)

            drawplt(image, resolved_boxes, args.width, args.height)
            drawplt(image, bbox[..., 1:5], args.width, args.height)
