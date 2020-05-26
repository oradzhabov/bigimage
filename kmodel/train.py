import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as alb
from joblib import Memory
from .model import create_model
from .data import get_data, Dataset, Dataloder
from .config import cfg


TRIM_GPU = False
if TRIM_GPU:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 16))
    for i, (name, img) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(img)
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


def get_training_augmentation(conf):
    train_transform = [

        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        #
        # scale_limit ((float, float) or float) – scaling factor range. If scale_limit is a single float value,
        # the range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).
        # rotate_limit ((int, int) or int) – rotation range. If rotate_limit is a single int value,
        # the range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        # shift_limit ((float, float) or float) – shift factor range for both height and width. If shift_limit is
        # a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper
        # bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        # alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.5, p=1, border_mode=0),
        alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=90, p=1, border_mode=0),

        # Pad side of the image / max if side is less than desired number.
        alb.PadIfNeeded(min_height=conf.img_wh, min_width=conf.img_wh, always_apply=True, border_mode=0),

        alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        alb.RandomGamma(p=1),

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
    return alb.Compose(train_transform)


def get_validation_augmentation(conf):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        alb.PadIfNeeded(conf.img_wh, conf.img_wh),
        alb.RandomCrop(height=conf.img_wh, width=conf.img_wh, always_apply=True),
    ]
    return alb.Compose(test_transform)


def read_sample(img_path, himg_path, mask_path):
    # read data
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    add_height = True
    if add_height:
        himg = cv2.imread(himg_path)
        if len(himg.shape) > 2:
            himg = himg[..., 0][..., np.newaxis]
        img = np.concatenate((img, himg), axis=-1)

    mask = cv2.imread(mask_path, 0).astype('float32') / 255.0
    if len(mask.shape) == 2:
        mask = mask[..., np.newaxis]

    # add background if mask is not binary
    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)

    return img, mask


# if __name__ == '__main__':
def run():
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    data_dir, ids_train, ids_test = get_data(cfg, test_size=0.33)

    # Manage caching data access
    cache_folder = os.path.join(cfg.data_dir, '.cache')
    memory = Memory(cache_folder, verbose=0)
    memory.clear(warn=False)
    data_reader = memory.cache(read_sample) if memory is not None else read_sample

    """
    if False:
        dataset = Dataset(data_reader, data_dir, ids_train)

        image, mask = dataset[2]  # get some sample
        print('img shape,dtype,min,max: ', image.shape, image.dtype, np.min(image), np.max(image))
        print('mask shape,dtype,min,max: ', mask.shape, mask.dtype, np.min(mask), np.max(mask))
        visualize(
            image=image,
            muckpile_mask=mask[..., 0].squeeze()
        )
    """
    see_sample_augmentations = False
    if see_sample_augmentations:
        # Lets look at augmented data we have
        dataset = Dataset(data_reader,
                          data_dir,
                          ids_train,
                          min_mask_ratio=0.01,
                          augmentation=get_training_augmentation(cfg)
                          )

        for i in range(15):
            image, mask = dataset[i]
            print('img shape,dtype,min,max: ', image.shape, image.dtype, np.min(image), np.max(image))
            print('mask shape,dtype,min,max: ', mask.shape, mask.dtype, np.min(mask), np.max(mask))

            visualize(
                image=image,
                muckpile_mask=mask[..., 0].squeeze(),
            )

    # ****************************************************************************************************************
    # Create model
    # ****************************************************************************************************************
    model, weights_path, _ = create_model(conf=cfg, compile_model=True)

    # Dataset for train images
    train_dataset = Dataset(data_reader, data_dir, ids_train,
                            min_mask_ratio=0.01,
                            augmentation=get_training_augmentation(cfg),
                            backbone=cfg.backbone)
    # Dataset for validation images
    valid_dataset = Dataset(data_reader, data_dir, ids_test,
                            min_mask_ratio=0.01,
                            augmentation=get_validation_augmentation(cfg),
                            backbone=cfg.backbone)

    train_dataloader = Dataloder(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                 cpu_units_nb=Dataloder.get_cpu_units_nb())
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    train_batch = train_dataloader[0]
    # assert train_batch[0].shape == (cfg.batch_size, cfg.img_wh, cfg.img_wh, 3)
    # assert train_batch[1].shape == (cfg.batch_size, cfg.img_wh, cfg.img_wh, n_classes)
    print('X: ', train_batch[0].shape, train_batch[0].dtype, np.min(train_batch[0]), np.max(train_batch[0]))
    print('Y: ', train_batch[1].shape, train_batch[1].dtype, np.min(train_batch[1]), np.max(train_batch[1]))

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(weights_path,
                                        monitor='val_f1-score',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode='max',
                                        verbose=1),
        # Adam optimizer SHOULD not control LR
        # keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, factor=0.2)
        #
        # keras.callbacks.EarlyStopping(monitor='val_mean_iou',
        #                              min_delta=0.01,
        #                              patience=40,
        #                              verbose=0, mode='max')
    ]

    # train model
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=cfg.epochs,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(131)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation accuracy values
    plt.subplot(132)
    plt.plot(history.history['f1-score'])
    plt.plot(history.history['val_f1-score'])
    plt.title('Model F1-score')
    plt.ylabel('F1-score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(133)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
