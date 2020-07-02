import os
import json
from .data import get_data, Dataloder
from .train import read_sample
import sys
sys.path.append(sys.path[0] + "/..")
from solvers import ISolver
from data_provider import IDataProvider
from augmentation import IAug


def run(cfg, solver: ISolver, dataprovider: IDataProvider, aug: IAug, show_random_items_nb=0):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Prepare data and split to train/test subsets
    data_dir, ids_train, ids_test = get_data(cfg, test_size=cfg.test_aspect)

    data_reader = read_sample

    test_dataset = dataprovider(data_reader, data_dir, ids_test, cfg,
                                min_mask_ratio=cfg.min_mask_ratio,
                                augmentation=aug.get_validation_augmentation(cfg),
                                prep_getter=solver.get_prep_getter())
    print('Dataset length: {}'.format(len(test_dataset)))

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    model, weights_path, metrics = solver.build(compile_model=True)

    if show_random_items_nb > 0:
        test_dataset.show_predicted(solver, show_random_items_nb)

    print('Evaluate model...')
    scores = model.evaluate_generator(test_dataloader, verbose=1)

    result_dict = dict({'cfg': dict(cfg)})
    print("Loss: {:.5}".format(scores[0]))
    result_dict['loss'] = scores[0]
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        print("mean {}: {:.5}".format(metric_name, value))
        result_dict[metric_name] = value

    #
    # todo: till config contains complex objects/classes it cannot be stored into json.
    #
    """
    dir_to_save = os.path.dirname(weights_path)
    with open(os.path.join(dir_to_save, 'evaluation.json'), 'w', newline=os.linesep) as f:
        json.dump(result_dict, f)
    """