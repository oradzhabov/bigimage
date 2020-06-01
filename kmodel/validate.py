import os
import numpy as np
import cv2
from .data import get_data, Dataset, Dataloder
from .model import create_model
from .config import cfg
from .train import read_sample, get_validation_augmentation, visualize, denormalize
from .kutils import get_contours


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
        result_list = list()
        for i in ids:
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image, verbose=0).round()
            scores = model.evaluate(image, np.expand_dims(gt_mask, axis=0), batch_size=1, verbose=0)

            gt_cntrs = get_contours((gt_mask.squeeze() * 255).astype(np.uint8))
            pr_cntrs = get_contours((pr_mask.squeeze() * 255).astype(np.uint8))
            img_metrics = dict()
            for metric, value in zip(metrics, scores[1:]):
                metric_name = metric if isinstance(metric, str) else metric.__name__
                img_metrics[metric_name] = value

            item = dict({'index': i, 'gt_cntrs': gt_cntrs, 'pr_cntrs': pr_cntrs, 'metrics': img_metrics})
            item['image'] = image
            result_list.append(item)
        # sort list to start from the worst result
        result_list = sorted(result_list, key=lambda item: item['metrics']['f1-score'])

        for item in result_list:
            image = item['image']
            img_fname = test_dataset.get_fname(item['index'])

            gt_cntrs = item['gt_cntrs']
            pr_cntrs = item['pr_cntrs']

            img_temp = (denormalize(image.squeeze()[..., :3]) * 255).astype(np.uint8)
            cv2.drawContours(img_temp, gt_cntrs, -1, (255, 0, 0), 3)
            cv2.drawContours(img_temp, pr_cntrs, -1, (0, 0, 255), 5)

            visualize(
                title='{}, F1:{:.4f}, IoU:{:.4f}'.format(img_fname,
                                                         item['metrics']['f1-score'],
                                                         item['metrics']['iou_score']),
                result=img_temp
            )

    print('Evaluate model...')
    scores = model.evaluate_generator(test_dataloader, verbose=1)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        print("mean {}: {:.5}".format(metric_name, value))
