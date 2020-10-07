import os
import logging
import numpy as np
from tqdm import tqdm
from . import SemanticSegmentationDataProvider
from ..kutils import utilites


class RegressionSegmentationDataProvider(SemanticSegmentationDataProvider):
    def _preprocess_mask(self, mask):
        mask = mask.astype(np.float32) / 255.0
        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        return mask

    def show(self, i):
        image, mask = self.__getitem__(i)

        # image_rgb = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)
        image_rgb = image[..., :3].copy()
        mask = (mask*255).astype(np.uint8)

        utilites.visualize(
            title=self.get_fname(i),
            img_fname=None,
            Image=image_rgb,
            Masked_Image=((image_rgb.astype(np.float32) +
                           np.dstack((mask, mask*0, mask*0)).astype(np.float32))//2).astype(np.uint8),
        )

    def show_predicted(self, solver, show_random_items_nb, save_imgs=False):
        ids = np.random.choice(np.arange(len(self)), size=show_random_items_nb, replace=False)
        result_list = list()
        for i in tqdm(ids):
            image, gt_mask = self.__getitem__(i)
            image = np.expand_dims(image, axis=0)
            pr_mask = solver.model.predict(image, verbose=0)[0]
            pr_mask = solver.post_predict(pr_mask)
            scores = solver.model.evaluate(image, np.expand_dims(gt_mask, axis=0), batch_size=1, verbose=0)

            # gt_cntrs = utilites.get_contours((gt_mask * 255).astype(np.uint8))
            # pr_cntrs = utilites.get_contours((pr_mask * 255).astype(np.uint8))
            img_metrics = dict()
            for metric, value in zip(solver.metrics, scores[1:]):
                metric_name = metric if isinstance(metric, str) else metric.__name__
                img_metrics[metric_name] = value

            item = dict({'index': i, 'metrics': img_metrics})
            item['gt_mask'] = gt_mask.squeeze()
            item['pr_mask'] = pr_mask.squeeze()
            item['image'] = image.squeeze()
            result_list.append(item)
        # sort list to start from the worst result
        result_list = sorted(result_list, key=lambda it: it['metrics']['mae'])[::-1]  # todo: why hardcoded mae ?

        img_storing_dir = os.path.join(self.conf.solution_dir, 'evaluate_imgs')
        if not os.path.isdir(img_storing_dir) and save_imgs:
            os.makedirs(img_storing_dir)
            logging.info('Folder {} has been created'.format(img_storing_dir))

        for item in result_list:
            image = item['image']
            img_fname = self.get_fname(item['index'])

            # gt_mask = item['gt_mask']
            pr_mask = item['pr_mask']

            img_temp = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)

            pr_mask = (pr_mask * 255).astype(np.uint8)

            utilites.visualize(
                title='{}, MAE:{:.4f}'.format(img_fname, item['metrics']['mae']),
                img_fname=os.path.join(img_storing_dir, img_fname) if save_imgs else None,
                Image=img_temp,
                Masked_Image=((img_temp.astype(np.float32) +
                               np.dstack((pr_mask, pr_mask*0, pr_mask*0)).astype(np.float32))//2).astype(np.uint8),
                Mask=pr_mask
            )


class RegressionSegmentationSingleDataProvider(RegressionSegmentationDataProvider):
    # todo: seems copy of other class SemanticSegmentationSingleDataProvider
    def __init__(self, data_reader, img_fname, himg_fname, bbox, configure, prep_getter):
        super().__init__(data_reader=data_reader,
                         data_dir='',
                         ids=list(),
                         bbox=bbox,
                         conf=configure,
                         min_mask_ratio=0.0,
                         augmentation=None,
                         prep_getter=prep_getter)

        self.src_folders = ['1', '2']  # In this case folder names are no mater. The order is matter as always.
        self.src_data = dict({k: list() for k in self.src_folders})

        self.src_data[self.src_folders[0]].append(img_fname)
        self.src_data[self.src_folders[1]].append(himg_fname)

        # Find the actual length of dataset
        keys = list(self.src_data)
        self._length = len(self.src_data[keys[0]]) if len(keys) > 0 else 0
