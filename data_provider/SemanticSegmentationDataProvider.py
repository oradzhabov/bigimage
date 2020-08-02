import logging
import sys
sys.path.append(sys.path[0] + "/..")
import os
import cv2
import numpy as np
from tqdm import tqdm
from . import IDataProvider
from kutils import utilites
import random
import albumentations as alb


class SemanticSegmentationDataProvider(IDataProvider):
    @staticmethod
    def _initializer(obj, data_reader, augmentation, bbox, configure, prep_getter):
        def random_color():
            levels = range(32, 256, 32)
            return [random.choice(levels) for _ in range(3)]
        colors_default = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
                          [255, 255, 0], [255, 0, 255], [0, 255, 255]]

        obj.images_fps = list()
        obj.himages_fps = list()
        obj.masks_fps = list()
        obj.augmentation = augmentation
        obj.bbox = bbox
        obj.conf = configure
        obj.data_reader = data_reader
        obj.prep_getter = prep_getter

        if obj.conf.cls_nb > len(colors_default):
            colors_default = colors_default + [random_color() for _ in range(obj.conf.cls_nb - len(colors_default))]
        obj.class_colors = colors_default[:obj.conf.cls_nb]

    def __init__(self,
                 data_reader,
                 data_dir,
                 ids,
                 bbox,
                 conf,
                 min_mask_ratio=0.0,
                 augmentation=None,
                 prep_getter=None):
        super(SemanticSegmentationDataProvider, self).__init__()
        SemanticSegmentationDataProvider._initializer(self, data_reader, augmentation, bbox, conf, prep_getter)

        # todo: it will be better if remove hardcoded subfolders outside the class
        if conf.use_heightmap:
            # Order is important because later it will be stacked into sample space
            self.src_folders = ['imgs', 'himgs']
        else:
            self.src_folders = ['imgs']
        src_mask_folder = 'masks.{}'.format(conf.data_subset)

        self.src_data = dict({k: list() for k in self.src_folders})
        self.src_mask = list()

        logging.info('Collect samples...')
        for fn in tqdm(ids):
            is_data_fully_exist = True
            subitem = dict({k: list() for k in self.src_folders})
            std_data_max = 0.0
            for data_folder, _ in self.src_data.items():
                file_fn = os.path.join(data_dir, data_folder, fn)
                if not os.path.isfile(file_fn):
                    is_data_fully_exist = False
                    break

                subitem[data_folder].append(file_fn)

                # Check is data not empty
                img = cv2.imread(file_fn)
                std_data = np.max(cv2.meanStdDev(img)[1])
                std_data_max = max(std_data_max, std_data)

            if is_data_fully_exist and std_data_max > 0.0:
                mask_fn = os.path.join(data_dir, src_mask_folder, fn)
                if os.path.isfile(mask_fn):
                    img = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)

                    mask_nonzero_nb = np.count_nonzero(img)
                    mask_nonzero_ratio = mask_nonzero_nb / img.size
                    if mask_nonzero_ratio >= min_mask_ratio:
                        self.src_mask.append(mask_fn)
                        for k, v in subitem.items():
                            self.src_data[k] += v

        # Find the actual length of dataset
        keys = list(self.src_data)
        self._length = len(self.src_data[keys[0]]) if len(keys) > 0 else 0

    def _preprocess_mask(self, mask):
        class_values = np.arange(len(self.conf.classes['class']) if self.conf.classes is not None else 1,
                                 dtype=np.uint8)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == 255 - v) for v in class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        return mask

    def __getitem__(self, i):
        i = i % len(self)

        data_paths = [self.src_data[k][i] if i < len(self.src_data[k]) else None for k in self.src_folders]
        mask_path = self.src_mask[i] if i < len(self.src_mask) else None

        image, mask = self.data_reader(data_paths, mask_path, self.bbox)

        if mask is not None:
            mask = self._preprocess_mask(mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply pre-processing
        if len(self.conf.backbone) > 0:
            # To support thread-safe and process-safe code we should obtain preprocessor on the fly
            # and do not prepare it before
            if self.prep_getter is not None:
                preprocessor = self.prep_getter(self.conf.backbone)
                if preprocessor:
                    if isinstance(preprocessor, alb.Compose):
                        sample = preprocessor(image=image)
                        # image, mask = sample['image'], sample['mask']
                        image = sample['image']
                    else:
                        image = preprocessor(image)  # todo: Take too much RAM
                        # Operate possible case when custom preprocessor modified data size
        if mask is not None:
            if image.shape[:2] != mask.shape[:2]:
                logging.info('Mask has not matched image resolution. To match shape it was scaled.')
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __len__(self):
        return self._length

    def get_fname(self, i):
        keys = self.src_folders
        if len(keys) == 0:
            return 0

        i = i % len(self)
        return os.path.basename(self.src_data[keys[0]][i])

    def get_color(self, class_ind):
        return self.class_colors[class_ind]

    def show(self, i):
        image, mask = self.__getitem__(i)
        logging.info('name: {}'.format(os.path.basename(self.get_fname(i))))
        logging.info('img shape {},dtype {},min {},max {}'.format(image.shape, image.dtype,
                                                                   np.min(image), np.max(image)))
        logging.info('mask shape {},dtype {},min {},max {}, info_ratio {}'.
                     format(mask.shape, mask.dtype, np.min(mask), np.max(mask), np.count_nonzero(mask) / mask.size))

        # image_rgb = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)
        image_rgb = image[..., :3].copy()
        gt_cntrs_list = utilites.get_contours(((mask * 255).astype(np.uint8)))
        for class_index, class_ctrs in enumerate(gt_cntrs_list):
            cv2.drawContours(image_rgb, class_ctrs, -1, self.get_color(class_index), int(3 * self.conf.img_wh / 512))
        if self.conf.classes is not None:
            class_names = self.conf.classes['class']
            for class_index, class_name in enumerate(class_names):
                fsc = self.conf.img_wh / 512
                x = int(6 * fsc)
                y = int(30 * (1 + class_index) * fsc)
                utilites.write_text(image_rgb, class_name, (x, y), self.get_color(class_index), fsc)

        utilites.visualize(
            title=self.get_fname(i),
            Image=image_rgb,
            Height=image[..., 3] if image.shape[-1] > 3 else None,
        )

    def show_predicted(self, solver, show_random_items_nb):
        ids = np.random.choice(np.arange(len(self)), size=show_random_items_nb)
        result_list = list()
        for i in ids:
            image, gt_mask = self.__getitem__(i)
            image = np.expand_dims(image, axis=0)
            pr_mask = solver.model.predict(image, verbose=0)[0]
            pr_mask = solver.post_predict(pr_mask)
            scores = solver.model.evaluate(image, np.expand_dims(gt_mask, axis=0), batch_size=1, verbose=0)

            gt_cntrs = utilites.get_contours((gt_mask * 255).astype(np.uint8))
            pr_cntrs = utilites.get_contours((pr_mask * 255).astype(np.uint8))
            img_metrics = dict()
            for metric, value in zip(solver.metrics, scores[1:]):
                metric_name = metric if isinstance(metric, str) else metric.__name__
                img_metrics[metric_name] = value

            item = dict({'index': i, 'gt_cntrs': gt_cntrs, 'pr_cntrs': pr_cntrs, 'metrics': img_metrics})
            item['image'] = image.squeeze()
            result_list.append(item)
        # sort list to start from the worst result
        result_list = sorted(result_list, key=lambda it: it['metrics']['f1-score'])

        for item in result_list:
            image = item['image']
            img_fname = self.get_fname(item['index'])

            gt_cntrs = item['gt_cntrs']
            pr_cntrs = item['pr_cntrs']

            img_temp = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)
            for class_index, class_ctrs in enumerate(gt_cntrs):
                cv2.drawContours(img_temp, class_ctrs, -1, self.get_color(class_index), 2)
            for class_index, class_ctrs in enumerate(pr_cntrs):
                color = self.get_color(class_index)
                cv2.drawContours(img_temp, class_ctrs, -1, color, 6)
                color = [c // 2 for c in color]
                cv2.drawContours(img_temp, class_ctrs, -1, color, 2)

            utilites.visualize(
                title='{}, F1:{:.4f}, IoU:{:.4f}'.format(img_fname,
                                                         item['metrics']['f1-score'],
                                                         item['metrics']['iou_score']),
                Result=img_temp,
                Height=image[..., 3] if image.shape[-1] > 3 else None,
            )


class SemanticSegmentationSingleDataProvider(SemanticSegmentationDataProvider):
    # todo: seems copy of other class RegressionSegmentationSingleDataProvider
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
