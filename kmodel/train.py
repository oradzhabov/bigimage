import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import json
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
from kutils.read_sample import read_sample
from kutils.JSONEncoder import json_def_encoder


TRIM_GPU = False
if TRIM_GPU:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)


def run(cfg, solver: ISolver, dataprovider: IDataProvider, aug: IAug, review_augmented_sample=False, review_train=True):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        logging.error('There are no such data folder {}'.format(cfg.data_dir))
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

        # Specify params according to which subset to review
        ids, augm = (ids_train, aug.get_training_augmentation(cfg)) if review_train else \
            (ids_test, aug.get_validation_augmentation(cfg, cfg.minimize_train_aug))

        # Lets look at augmented data we have
        dataset = dataprovider(data_reader, data_dir, ids, ((0, 0), (None, None)), cfg,
                               min_mask_ratio=cfg.min_mask_ratio,
                               augmentation=augm,
                               prep_getter=None  # don't use preparation to see actually augmentation the data
                               )
        logging.info('Dataset length: {}'.format(len(dataset)))

        for i in range(150):
            dataset.show(i)

        return

    # ****************************************************************************************************************
    # Create model
    # ****************************************************************************************************************
    # Dataset for train images
    train_dataset = dataprovider(data_reader, data_dir, ids_train, ((0, 0), (None, None)), cfg,
                                 min_mask_ratio=cfg.min_mask_ratio,
                                 augmentation=aug.get_training_augmentation(cfg, cfg.minimize_train_aug),
                                 prep_getter=solver.get_prep_getter())
    # Dataset for validation images
    valid_dataset = dataprovider(data_reader, data_dir, ids_test, ((0, 0), (None, None)), cfg,
                                 min_mask_ratio=cfg.min_mask_ratio,
                                 augmentation=aug.get_validation_augmentation(cfg, cfg.minimize_train_aug),
                                 prep_getter=solver.get_prep_getter())

    train_dataloader = Dataloder(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # Inform general samples info
    train_batch = train_dataloader[0]
    logging.info('Train X: {},{},{},{}'.format(train_batch[0].shape, train_batch[0].dtype,
                                               np.min(train_batch[0]), np.max(train_batch[0])))
    logging.info('Train Y: {},{},{},{}'.format(train_batch[1].shape, train_batch[1].dtype,
                                               np.min(train_batch[1]), np.max(train_batch[1])))
    logging.info('Train Batch size multiplier: {}'.format(cfg.batch_size_multiplier))
    logging.info('Train Samples Nb: {}'.format(len(train_dataset)))
    class_weights = None
    if hasattr(train_dataset, 'mask_uniq_values_nb'):
        if train_dataset.mask_uniq_values_nb is not None and cfg.apply_class_weights:
            mask_min_nb = np.min(train_dataset.mask_uniq_values_nb)
            if mask_min_nb > 0:
                class_weights = (train_dataset.mask_uniq_values_nb / mask_min_nb) ** -1
    #
    val_batch = valid_dataloader[0]
    logging.info('Validate Samples Nb: {}'.format(len(valid_dataset)))
    logging.info('Val X: {},{},{},{}'.format(val_batch[0].shape, val_batch[0].dtype,
                                             np.min(val_batch[0]), np.max(val_batch[0])))
    logging.info('Val Y: {},{},{},{}'.format(val_batch[1].shape, val_batch[1].dtype,
                                             np.min(val_batch[1]), np.max(val_batch[1])))
    if train_batch[0].shape[1] != val_batch[0].shape[1] or train_batch[0].shape[2] != val_batch[0].shape[2]:
        logging.info('Pay attention, that sample HW in train subset is different to validation subset. '
                     'It may affect to metric cross comparison')

    model, weights_path, metrics = solver.build(compile_model=True, class_weights=class_weights)

    logging.info('Storing configuration...')
    with open(os.path.join(cfg.solution_dir, 'configuration.json'), 'w', newline=os.linesep) as f:
        json.dump(dict({'cfg': dict(cfg)}), f, default=json_def_encoder)

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

    if hasattr(cfg, 'callbacks'):
        callbacks = callbacks + cfg.callbacks

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
