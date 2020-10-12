import os
import logging
import numpy as np
from joblib import Memory
import matplotlib.pyplot as plt
from .data import get_data
from ..solvers import ISolver
from ..data_provider import IDataProvider
from ..augmentation import IAug
from ..kutils.read_sample import read_sample
from ..bin_keras import Dataloder, LRFinder


def run(cfg, solver: ISolver, dataprovider: IDataProvider, aug: IAug,
        start_lr=0.0001, end_lr=1, no_epochs=5, moving_average=20):

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

    # Dataset for train images
    train_dataset = dataprovider(data_reader, data_dir, ids_train, ((0, 0), (None, None)), cfg,
                                 min_mask_ratio=cfg.min_mask_ratio,
                                 augmentation=aug.get_training_augmentation(cfg, cfg.minimize_train_aug),
                                 prep_getter=solver.get_prep_getter())

    train_dataloader = Dataloder(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    logging.info('Train Batch size multiplier: {}'.format(cfg.batch_size_multiplier))
    logging.info('Train Samples Nb: {}'.format(len(train_dataset)))
    class_weights = None
    if hasattr(train_dataset, 'mask_uniq_values_nb'):
        if train_dataset.mask_uniq_values_nb is not None and cfg.apply_class_weights:
            mask_min_nb = np.min(train_dataset.mask_uniq_values_nb)
            if mask_min_nb > 0:
                class_weights = (train_dataset.mask_uniq_values_nb / mask_min_nb) ** -1

    model, weights_path, metrics = solver.build(compile_model=True, class_weights=class_weights)

    # Get monitoring metric
    monitoring_metric_name, monitoring_metric_mode = solver.monitoring_metric()

    # Instantiate the Learning Rate Range Test / LR Finder
    lr_finder = LRFinder(model)

    # Perform the Learning Rate Range Test
    outputs = lr_finder.find_generator(train_dataloader,
                                       start_lr=start_lr,
                                       end_lr=end_lr,
                                       epochs=no_epochs)

    # Get values
    learning_rates = lr_finder.lrs
    losses = lr_finder.losses
    loss_changes = []

    # Compute smoothed loss changes
    # Inspired by Keras LR Finder: https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
    for i in range(moving_average, len(learning_rates)):
        loss_changes.append((losses[i] - losses[i - moving_average]) / moving_average)

    # Generate plot for Loss Deltas
    plt.plot(learning_rates[moving_average:], loss_changes)
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.ylabel('loss delta')
    plt.xlabel('learning rate (log scale)')
    plt.title('Results for Learning Rate Range Test / Loss Deltas for Learning Rate')
    plt.savefig(os.path.join(cfg.solution_dir, 'Loss_Deltas_for_Learning_Rate.png'))
    plt.show()

    # Generate plot for Loss Values
    plt.plot(learning_rates, losses)
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('learning rate (log scale)')
    plt.title('Results for Learning Rate Range Test / Loss Values for Learning Rate')
    plt.savefig(os.path.join(cfg.solution_dir, 'Loss_Values_for_Learning_Rate.png'))
    plt.show()
