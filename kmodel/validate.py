import os
import numpy as np
from data import get_data, Dataset, Dataloder
from model import create_model
from config import cfg
from train import read_sample, get_validation_augmentation, get_preprocessing, visualize, denormalize


if __name__ == '__main__':
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Get all data into test-set
    # data_dir, ids_test, _ = get_data(cfg, 0.0)
    data_dir, ids_train, ids_test = get_data(cfg, 0.33)

    data_reader = read_sample

    # ****************************************************************************************************************
    # Create model
    # ****************************************************************************************************************
    model, weights_path, preprocess_input, metrics = create_model(conf=cfg, compile_model=True)

    test_dataset = Dataset(data_reader, data_dir, ids_test,
                           min_mask_ratio=0.0,
                           # augmentation=get_validation_augmentation(cfg),  # do not crop
                           preprocessing=get_preprocessing(preprocess_input))
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    test_random_items_n = 15
    if test_random_items_n > 0:
        ids = np.random.choice(np.arange(len(test_dataset)), size=test_random_items_n)
        for i in ids:
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image).round()

            visualize(
                image=denormalize(image.squeeze()),
                gt_mask=gt_mask[..., 0].squeeze(),
                pr_mask=pr_mask[..., 0].squeeze(),
            )

    print('Evaluate model...')
    scores = model.evaluate_generator(test_dataloader, verbose=1)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        print("mean {}: {:.5}".format(metric_name, value))
