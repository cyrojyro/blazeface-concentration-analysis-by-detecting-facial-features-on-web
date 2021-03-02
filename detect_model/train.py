from utils.widerface_loader import *
from model.backbones import *
from utils.metrics import *
from model.models import *
from utils.losses import *
import tensorflow as tf
import numpy as np
import configobj
import argparse
import os

CONFIG_FILE = 'train_config.ini'

parser = argparse.ArgumentParser(
    description="Test model using arbitrary image")
parser.add_argument('--model', type=str, default='',
                    help='Model directory to resume training')
parser.add_argument('--epoch', type=int, default=0,
                    help='Resume epoch')


def preprocess_config():
    config = configobj.ConfigObj(CONFIG_FILE)
    cfg = config['DEFAULT']
    return cfg


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = preprocess_config()
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))

    gts = cfg['wider_gts']
    gts = [gts] if isinstance(gts, str) else [gt for gt in gts]

    # [num_picture, image_width, image_height, 3], [num_picture, num_gt, 4]
    images, labels = load_widerface(gts, cfg['wider_train'], cfg.as_int('input_w'),
                                    cfg.as_int('input_h'), max_size=cfg.as_int('max_dset'))

    # [num_box(896), 4]
    anchors = np.load(os.path.join(cfg['anchor_path'], "anchors.npy"))

    # split validation set
    val_num = int(len(images) * cfg.as_float('validation_ratio'))

    images, labels = images[val_num:], labels[val_num:]
    data_loader = dataloader(images, labels, anchors,
                             batch_size=cfg.as_int('batch_size'))

    images_val, labels_val = images[:val_num], labels[:val_num]
    data_loader_val = dataloader(images_val, labels_val, anchors,
                                 batch_size=cfg.as_int('batch_size'), augment=False)

    if args.model:
        model = tf.keras.models.load_model(args.model, custom_objects={
                                           'multiboxLoss': multiboxLoss,
                                           'cn_loss': cn_loss, 'cp_loss': cp_loss,
                                           'l_loss': l_loss})
    else:
        model = Blazeface(input_dim=(
            cfg.as_int('input_w'), cfg.as_int('input_h'), 3),
            backbone=blaze_mediapipe_backbone).build_model()

    loss = multiboxLoss
    optim = tf.keras.optimizers.Adam(
        learning_rate=cfg.as_float('learning_rate'), amsgrad=True)
    model.compile(loss=loss, optimizer=optim,
                  metrics=[cp_loss, cn_loss, l_loss])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=cfg.as_float('early_stop_patience'))
    rdlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=cfg.as_float(
                                                    'rdlr_factor'),
                                                patience=cfg.as_float(
                                                    'rdlr_patience'),
                                                min_lr=cfg.as_float('rdlr_min'))
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            cfg['model_path'], cfg['model_name'] + '{epoch:03d}.hdf5'),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    val_steps = int(cfg.as_int('steps_per_epoch') *
                    cfg.as_float('validation_ratio'))
    res = model.fit(x=data_loader,
                    validation_data=data_loader_val,
                    validation_steps=val_steps,
                    initial_epoch=args.epoch,
                    epochs=cfg.as_int('epochs'),
                    steps_per_epoch=cfg.as_int('steps_per_epoch'),
                    callbacks=[ckpt, early_stop, rdlr],
                    verbose=1)
