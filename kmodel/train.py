import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import albumentations as alb
from .albumentations2 import HueSaturationValue2, RandomGamma2, RandomBrightnessContrast2
from joblib import Memory
from .model import create_model
from .data import get_data, Dataset, Dataloder
from .PlotLosses import PlotLosses
from .kutils import get_contours

TRIM_GPU = False
if TRIM_GPU:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)


def visualize(title, **images):
    """PLot images in one row."""
    img_filtered = {key: value for (key, value) in images.items() if value is not None}
    n = len(img_filtered)
    fig = plt.figure(figsize=(16, 16))
    for i, (name, img) in enumerate(img_filtered.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(img)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# def round_clip_0_1(x, **kwargs):
#     return x.round().clip(0, 1)


def get_training_augmentation(conf, is_stub=False):
    train_transform = [

        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(always_apply=False, p=0.5),
        #
        # scale_limit ((float, float) or float) – scaling factor range. If scale_limit is a single float value,
        # the range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).
        # rotate_limit ((int, int) or int) – rotation range. If rotate_limit is a single int value,
        # the range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        # shift_limit ((float, float) or float) – shift factor range for both height and width. If shift_limit is
        # a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper
        # bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        # alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.5, p=1, border_mode=0),
        alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=90, p=1, border_mode=0, interpolation=cv2.INTER_LANCZOS4),

        # Pad side of the image / max if side is less than desired number.
        alb.PadIfNeeded(min_height=conf.img_wh, min_width=conf.img_wh, always_apply=True, border_mode=0),

        RandomBrightnessContrast2(brightness_limit=(-0.5, 0.2), contrast_limit=0.2, p=1),
        RandomGamma2(p=1),

        # alb.IAAAdditiveGaussianNoise(p=0.2),

        # alb.OneOf(
        #    [
        #        alb.CLAHE(p=1),
        #        alb.RandomBrightness(p=1),
        #        alb.RandomGamma(p=1),
        #    ],
        #    p=0.9,
        # ),

        # alb.OneOf(
        #    [
        #        alb.IAASharpen(p=1),
        #        alb.Blur(blur_limit=3, p=1),
        #        alb.MotionBlur(blur_limit=3, p=1),
        #    ],
        #    p=0.9,
        # ),
        HueSaturationValue2(hue_shift_limit=15, sat_shift_limit=(-20, 10), val_shift_limit=0, p=1.0),
        # alb.OneOf(
        #    [
        #        alb.RandomContrast(p=1),
        #        alb.HueSaturationValue(p=1),
        #    ],
        #    p=0.9,
        # ),
        # alb.Lambda(mask=round_clip_0_1),
        # Crop a random part of the input.
        alb.RandomCrop(height=conf.img_wh, width=conf.img_wh, always_apply=True),
    ]
    if is_stub:
        return alb.Compose([alb.PadIfNeeded(min_height=conf.img_wh, min_width=conf.img_wh,
                                            always_apply=True, border_mode=0),
                           alb.RandomCrop(height=conf.img_wh, width=conf.img_wh, always_apply=True)])
    return alb.Compose(train_transform)


def get_validation_augmentation(conf, is_stub=False):
    # Since batch-size in validation is 1, validation could be performed by whole crop-size.
    # To provide pos
    test_transform = [
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.RandomRotate90(always_apply=False, p=0.5),
        alb.PadIfNeeded(conf.img_wh_crop, conf.img_wh_crop, always_apply=True, border_mode=0),
        # alb.RandomCrop(height=conf.img_wh_crop, width=conf.img_wh_crop, always_apply=True),
    ]
    if is_stub:
        return alb.Compose([alb.PadIfNeeded(conf.img_wh_crop, conf.img_wh_crop, always_apply=True, border_mode=0)])
    return alb.Compose(test_transform)


def read_sample(img_path, himg_path, mask_path):
    # read data
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if himg_path is not None:
        himg = cv2.imread(himg_path)
        if len(himg.shape) > 2:
            himg = himg[..., 0][..., np.newaxis]

        if himg.shape[:2] != img.shape[:2]:
            print('WARNING: Height map has not matched image resolution. To match shape it was scaled.')
            himg = cv2.resize(himg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        img = np.concatenate((img, himg), axis=-1)

    mask = None
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).squeeze()

    return img, mask


def run(cfg):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Prepare data and split to train/test subsets
    data_dir, ids_train, ids_test = get_data(cfg, test_size=cfg.test_aspect)

    # Manage caching data access
    cache_folder = os.path.join(cfg.data_dir, '.cache')
    memory = Memory(cache_folder, verbose=0)
    memory.clear(warn=False)
    data_reader = memory.cache(read_sample) if memory is not None else read_sample

    see_sample_augmentations = False
    if see_sample_augmentations:
        matplotlib.use('TkAgg')  # Enable interactive mode

        # Lets look at augmented data we have
        dataset = Dataset(data_reader, data_dir, ids_train, cfg,
                          min_mask_ratio=0.01,
                          augmentation=get_training_augmentation(cfg)
                          )

        for i in range(150):
            image, mask = dataset[i]
            print('name: ', os.path.basename(dataset.images_fps[i % len(dataset.images_fps)]))
            print('img shape,dtype,min,max: ', image.shape, image.dtype, np.min(image), np.max(image))
            print('mask shape,dtype,min,max,info_ratio: ', mask.shape, mask.dtype, np.min(mask), np.max(mask),
                  np.count_nonzero(mask)/mask.size)

            image_rgb = (denormalize(image[..., :3]) * 255).astype(np.uint8)
            gt_cntrs_list = get_contours((mask * 255).astype(np.uint8))
            for class_index, class_ctrs in enumerate(gt_cntrs_list):
                cv2.drawContours(image_rgb, class_ctrs, -1, dataset.get_color(class_index), 3)
            visualize(
                title=dataset.get_fname(i),
                Image=image_rgb,
                Height=image[..., 3] if image.shape[-1] > 3 else None,
            )

        return

    # ****************************************************************************************************************
    # Create model
    # ****************************************************************************************************************
    model, weights_path, _ = create_model(conf=cfg, compile_model=True)

    # Dataset for train images
    train_dataset = Dataset(data_reader, data_dir, ids_train, cfg,
                            min_mask_ratio=cfg.min_mask_ratio,
                            augmentation=get_training_augmentation(cfg, cfg.minimize_train_aug))
    # Dataset for validation images
    valid_dataset = Dataset(data_reader, data_dir, ids_test, cfg,
                            min_mask_ratio=cfg.min_mask_ratio,
                            augmentation=get_validation_augmentation(cfg, cfg.minimize_train_aug))

    train_dataloader = Dataloder(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    train_batch = train_dataloader[0]
    # assert train_batch[0].shape == (cfg.batch_size, cfg.img_wh, cfg.img_wh, 3)
    # assert train_batch[1].shape == (cfg.batch_size, cfg.img_wh, cfg.img_wh, n_classes)
    print('X: ', train_batch[0].shape, train_batch[0].dtype, np.min(train_batch[0]), np.max(train_batch[0]))
    print('Y: ', train_batch[1].shape, train_batch[1].dtype, np.min(train_batch[1]), np.max(train_batch[1]))

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        # Save best result
        keras.callbacks.ModelCheckpoint(weights_path,
                                        monitor='val_f1-score',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode='max',
                                        verbose=1),
        # Save the latest result
        keras.callbacks.ModelCheckpoint('{}_last.h5'.format(os.path.join(os.path.dirname(weights_path),
                                                            os.path.splitext(os.path.basename(weights_path))[0])),
                                        monitor='val_f1-score',
                                        save_weights_only=True,
                                        save_best_only=False,
                                        mode='auto',
                                        verbose=0),

        # Adam optimizer SHOULD not control LR
        # keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, factor=0.2)
        #
        # keras.callbacks.EarlyStopping(monitor='val_mean_iou',
        #                              min_delta=0.01,
        #                              patience=40,
        #                              verbose=0, mode='max')
        PlotLosses(imgfile='{}.png'.format(os.path.join(os.path.dirname(weights_path),
                                                        os.path.splitext(os.path.basename(weights_path))[0])),
                   figsize=(8, 4))  # PNG-files processed in Windows & Ubuntu
    ]

    matplotlib.use('Agg')  # Disable TclTk because it sometime crash training!
    # train model
    model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=cfg.epochs,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )
