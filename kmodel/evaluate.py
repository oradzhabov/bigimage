import logging
import os
import numpy as np
import json
from .data import get_data, Dataloder
import sys
sys.path.append(sys.path[0] + "/..")
from solvers import ISolver
from data_provider import IDataProvider
from augmentation import IAug
from kutils.read_sample import read_sample


def run(cfg, solver: ISolver, dataprovider: IDataProvider, aug: IAug, show_random_items_nb=0):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        logging.error('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Prepare data and split to train/test subsets
    data_dir, ids_train, ids_test = get_data(cfg, test_size=cfg.test_aspect)

    data_reader = read_sample

    test_dataset = dataprovider(data_reader, data_dir, ids_test, ((0, 0), (None, None)), cfg,
                                min_mask_ratio=cfg.min_mask_ratio,
                                augmentation=aug.get_validation_augmentation(cfg),
                                prep_getter=solver.get_prep_getter())
    logging.info('Dataset length: {}'.format(len(test_dataset)))

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    model, _, metrics = solver.build(compile_model=True)

    if show_random_items_nb > 0:
        test_dataset.show_predicted(solver, show_random_items_nb)

    logging.info('Evaluate model...')
    scores = model.evaluate_generator(test_dataloader, verbose=1)

    # Provide evaluation report
    logging.info('*** EVALUATION REPORT ***')
    logging.info('Data dir: {}'.format(cfg.data_dir))
    logging.info('Data subset: {}'.format(cfg.data_subset))
    logging.info('Testing data aspect: {}'.format(cfg.test_aspect))
    logging.info('Solution dir: {}'.format(cfg.solution_dir))
    logging.info('Seed: {}'.format(cfg.seed))
    logging.info('Apply class weights: {}'.format(cfg.apply_class_weights))
    logging.info('Min data ratio: {}'.format(cfg.min_data_ratio))
    logging.info('Min mask ratio: {}'.format(cfg.min_mask_ratio))
    result_dict = dict({'cfg': dict(cfg)})
    logging.info("Loss: {:.5}".format(scores[0]))
    result_dict['loss'] = scores[0]
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        logging.info("mean {}: {:.5}".format(metric_name, value))
        result_dict[metric_name] = value
    #
    # todo: till config contains complex objects/classes it cannot be stored into json.
    #
    """
    dir_to_save = os.path.dirname(weights_path)
    with open(os.path.join(dir_to_save, 'evaluation.json'), 'w', newline=os.linesep) as f:
        json.dump(result_dict, f)
    """