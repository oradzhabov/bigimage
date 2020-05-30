import os
import numpy as np
from .data import get_data, Dataset, Dataloder
from .model import create_model
from .config import cfg
from .train import read_sample, get_validation_augmentation, visualize, denormalize


def run():
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Get all data into test-set
    # data_dir, ids_test, _ = get_data(cfg, 0.0)
    data_dir, ids_train, ids_test = get_data(cfg, 0.33)

    data_reader = read_sample

    # ****************************************************************************************************************
    # Create model. Compile it to obtain metrics
    # ****************************************************************************************************************
    model, weights_path, metrics = create_model(conf=cfg, compile_model=True)

    test_dataset = Dataset(data_reader, data_dir, ids_test, cfg.data_subset,
                           min_mask_ratio=0.01,
                           augmentation=get_validation_augmentation(cfg),
                           backbone=cfg.backbone)
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    test_random_items_n = 15
    if test_random_items_n > 0:
        ids = np.random.choice(np.arange(len(test_dataset)), size=test_random_items_n)
        for i in ids:
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image).round()

            visualize(
                groundtruth_mask=(denormalize(image.squeeze()) + gt_mask) / 2,
                masked_image=(denormalize(image.squeeze()) + pr_mask[0]) / 2
            )

    print('Evaluate model...')
    scores = model.evaluate_generator(test_dataloader, verbose=1)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        print("mean {}: {:.5}".format(metric_name, value))
