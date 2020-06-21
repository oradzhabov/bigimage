import os
import numpy as np
import cv2
import json
from .data import get_data, Dataset, Dataloder
from .train import read_sample, get_validation_augmentation, visualize, denormalize
from .kutils import get_contours


def run(cfg, solver, show_random_items_nb=0):
    # Check folder path
    if not os.path.exists(cfg.data_dir):
        print('There are no such data folder {}'.format(cfg.data_dir))
        exit(-1)

    # Prepare data and split to train/test subsets
    data_dir, ids_train, ids_test = get_data(cfg, test_size=cfg.test_aspect)

    data_reader = read_sample

    test_dataset = Dataset(data_reader, data_dir, ids_test, cfg,
                           min_mask_ratio=cfg.min_mask_ratio,
                           augmentation=get_validation_augmentation(cfg),
                           prep_getter=solver.get_prep_getter())

    model, weights_path, metrics = solver.build(compile_model=True)

    print('Dataset length: {}'.format(len(test_dataset)))
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    if show_random_items_nb > 0:
        ids = np.random.choice(np.arange(len(test_dataset)), size=show_random_items_nb)
        result_list = list()
        for i in ids:
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            # pr_mask = model.predict(image, verbose=0).round()  # todo: round() ?
            pr_mask = model.predict(image, verbose=0)[0]
            pr_mask = np.where(pr_mask > 0.5, 1.0, 0.0)
            scores = model.evaluate(image, np.expand_dims(gt_mask, axis=0), batch_size=1, verbose=0)

            gt_cntrs = get_contours((gt_mask * 255).astype(np.uint8))
            pr_cntrs = get_contours((pr_mask * 255).astype(np.uint8))
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

    result_dict = dict({'cfg': dict(cfg)})
    print("Loss: {:.5}".format(scores[0]))
    result_dict['loss'] = scores[0]
    for metric, value in zip(metrics, scores[1:]):
        metric_name = metric if isinstance(metric, str) else metric.__name__
        print("mean {}: {:.5}".format(metric_name, value))
        result_dict[metric_name] = value

    dir_to_save = os.path.dirname(weights_path)
    with open(os.path.join(dir_to_save, 'evaluation.json'), 'w', newline=os.linesep) as f:
        json.dump(result_dict, f)
