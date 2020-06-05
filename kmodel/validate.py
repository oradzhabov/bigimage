import os
import numpy as np
import cv2
from .data import get_data, Dataset, Dataloder
from .model import create_model
from .train import read_sample, get_validation_augmentation, visualize, denormalize
from .kutils import get_contours
from .production import create_model_production, get_preprocessing_production
import segmentation_models as sm


def prepare_model(cfg, test_production):
    # ****************************************************************************************************************
    # Create model. Compile it to obtain metrics
    # ****************************************************************************************************************
    if test_production:
        model, weights_path, metrics = create_model_production(conf=cfg, compile_model=True)
        prep_getter = get_preprocessing_production
        if not cfg.use_heightmap:
            print('ERROR: Production utilizes height map. Enable it before in config before running')
            model = None
        if cfg.mppx != 0.25:
            print('ERROR: Production utilizes 0.25 mppx. Setup it before in config before running')
            model = None
    else:
        model, weights_path, metrics = create_model(conf=cfg, compile_model=True)
        prep_getter = sm.get_preprocessing

    return model, weights_path, metrics, prep_getter


def run(cfg):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Prepare data and split to train/test subsets
    data_dir, ids_train, ids_test = get_data(cfg, test_size=cfg.test_aspect)

    data_reader = read_sample

    test_production = False
    model, _, metrics, prep_getter = prepare_model(cfg, test_production)

    test_dataset = Dataset(data_reader, data_dir, ids_test, cfg,
                           min_mask_ratio=0.01,
                           augmentation=get_validation_augmentation(cfg),
                           prep_getter=prep_getter)
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    test_random_items_n = 5
    if test_random_items_n > 0:
        ids = np.random.choice(np.arange(len(test_dataset)), size=test_random_items_n)
        result_list = list()
        for i in ids:
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image, verbose=0).round()  # todo: round() ?
            scores = model.evaluate(image, np.expand_dims(gt_mask, axis=0), batch_size=1, verbose=0)

            gt_cntrs = get_contours((gt_mask * 255).astype(np.uint8))
            pr_cntrs = get_contours((pr_mask[0] * 255).astype(np.uint8))
            img_metrics = dict()
            for metric, value in zip(metrics, scores[1:]):
                metric_name = metric if isinstance(metric, str) else metric.__name__
                img_metrics[metric_name] = value

            item = dict({'index': i, 'gt_cntrs': gt_cntrs, 'pr_cntrs': pr_cntrs, 'metrics': img_metrics})
            item['image'] = image.squeeze()
            result_list.append(item)
        # sort list to start from the worst result
        result_list = sorted(result_list, key=lambda it: it['metrics']['f1-score'])

        for item in result_list:
            image = item['image']
            img_fname = test_dataset.get_fname(item['index'])

            gt_cntrs = item['gt_cntrs']
            pr_cntrs = item['pr_cntrs']

            img_temp = (denormalize(image[..., :3]) * 255).astype(np.uint8)
            for class_index, class_ctrs in enumerate(gt_cntrs):
                cv2.drawContours(img_temp, class_ctrs, -1, test_dataset.get_color(class_index), 2)
            for class_index, class_ctrs in enumerate(pr_cntrs):
                color = test_dataset.get_color(class_index)
                cv2.drawContours(img_temp, class_ctrs, -1, color, 6)
                color = [c // 2 for c in color]
                cv2.drawContours(img_temp, class_ctrs, -1, color, 2)

            visualize(
                title='{}, F1:{:.4f}, IoU:{:.4f}'.format(img_fname,
                                                         item['metrics']['f1-score'],
                                                         item['metrics']['iou_score']),
                Result=img_temp,
                Height=image[..., 3] if image.shape[-1] > 3 else None,
            )

    print('Evaluate model...')
    scores = model.evaluate_generator(test_dataloader, verbose=1)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        print("mean {}: {:.5}".format(metric_name, value))
