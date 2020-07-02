import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib
from joblib import Memory
from .data import get_data, Dataloder
from .PlotLosses import PlotLosses
import sys
sys.path.append(sys.path[0] + "/..")
from solvers import ISolver
from data_provider import IDataProvider
from augmentation import IAug


TRIM_GPU = False
if TRIM_GPU:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)


def read_sample(data_paths, mask_path):
    data_paths = data_paths + [None] * (2 - len(data_paths))
    img_path, himg_path = data_paths

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


def run(cfg, solver: ISolver, dataprovider: IDataProvider, aug: IAug, review_augmented_sample=False):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Prepare data and split to train/test subsets
    data_dir, ids_train, ids_test = get_data(cfg, test_size=cfg.test_aspect)

    # Manage caching data access
    cache_folder = './cache'
    memory = Memory(cache_folder, verbose=0)
    memory.clear(warn=False)
    data_reader = memory.cache(read_sample) if memory is not None else read_sample

    if review_augmented_sample:
        matplotlib.use('TkAgg')  # Enable interactive mode

        # Lets look at augmented data we have
        dataset = dataprovider(data_reader, data_dir, ids_train, cfg,
                               min_mask_ratio=cfg.min_mask_ratio,
                               augmentation=aug.get_training_augmentation(cfg),
                               prep_getter=None  # don't use preparation to see actually augmentation the data
                               )
        print('Dataset length: ', len(dataset))

        for i in range(150):
            dataset.show(i)

        return

    # ****************************************************************************************************************
    # Create model
    # ****************************************************************************************************************
    # Dataset for train images
    train_dataset = dataprovider(data_reader, data_dir, ids_train, cfg,
                                 min_mask_ratio=cfg.min_mask_ratio,
                                 augmentation=aug.get_training_augmentation(cfg, cfg.minimize_train_aug),
                                 prep_getter=solver.get_prep_getter())
    # Dataset for validation images
    valid_dataset = dataprovider(data_reader, data_dir, ids_test, cfg,
                                 min_mask_ratio=cfg.min_mask_ratio,
                                 augmentation=aug.get_validation_augmentation(cfg, cfg.minimize_train_aug),
                                 prep_getter=solver.get_prep_getter())

    model, weights_path, metrics = solver.build(compile_model=True)

    train_dataloader = Dataloder(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # Inform general samples info
    train_batch = train_dataloader[0]
    print('X: ', train_batch[0].shape, train_batch[0].dtype, np.min(train_batch[0]), np.max(train_batch[0]))
    print('Y: ', train_batch[1].shape, train_batch[1].dtype, np.min(train_batch[1]), np.max(train_batch[1]))
    print('Train Samples Nb: ', len(train_dataset))
    print('Validate Samples Nb: ', len(valid_dataset))

    # Get monitoring metric
    monitoring_metric_name, monitoring_metric_mode = solver.monitoring_metric()

    # Define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        # Save best result
        keras.callbacks.ModelCheckpoint(weights_path,
                                        monitor=monitoring_metric_name,
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode=monitoring_metric_mode,
                                        verbose=1),
        # Save the latest result
        keras.callbacks.ModelCheckpoint('{}_last.h5'.format(os.path.join(os.path.dirname(weights_path),
                                                            os.path.splitext(os.path.basename(weights_path))[0])),
                                        monitor=monitoring_metric_name,
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
                   figsize=(12, 4*(1 + len(metrics))))  # PNG-files processed in Windows & Ubuntu
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
