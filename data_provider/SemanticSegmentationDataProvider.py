import os
import cv2
import numpy as np
from tqdm import tqdm
from . import IDataProvider
import random
import albumentations as alb


class SemanticSegmentationDataProvider(IDataProvider):
    @staticmethod
    def _initializer(obj, data_reader, augmentation, configure, prep_getter):
        def random_color():
            levels = range(32, 256, 32)
            return [random.choice(levels) for _ in range(3)]
        colors_default = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
                          [255, 255, 0], [255, 0, 255], [0, 255, 255]]

        obj.images_fps = list()
        obj.himages_fps = list()
        obj.masks_fps = list()
        obj.augmentation = augmentation
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
                 conf,
                 min_mask_ratio=0.0,
                 augmentation=None,
                 prep_getter=None):
        super(SemanticSegmentationDataProvider, self).__init__()
        SemanticSegmentationDataProvider._initializer(self, data_reader, augmentation, conf, prep_getter)

        # todo: it will be better if remove hardcoded subfolders outside the class
        if conf.use_heightmap:
            # Order is important because later it will be stacked into sample space
            self.src_folders = ['imgs', 'himgs']
        else:
            self.src_folders = ['imgs']
        src_mask_folder = 'masks.{}'.format(conf.data_subset)

        self.src_data = dict({k: list() for k in self.src_folders})
        self.src_mask = list()

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

    def __getitem__(self, i):
        i = i % len(self)

        data_paths = [self.src_data[k][i] if i < len(self.src_data[k]) else None for k in self.src_folders]
        mask_path = self.src_mask[i] if i < len(self.src_mask) else None

        image, mask = self.data_reader(data_paths, mask_path)

        if mask is not None:
            class_values = np.arange(len(self.conf.classes['class']) if self.conf.classes is not None else 1,
                                     dtype=np.uint8)
            # extract certain classes from mask (e.g. cars)
            masks = [(mask == 255 - v) for v in class_values]
            mask = np.stack(masks, axis=-1).astype('float32')

            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply pre-processing
        if len(self.conf.backbone) > 0:
            # To support thread-safe and process-safe code we should obtain preprocessor on the fly
            # and do not prepare it before
            preprocessor = self.prep_getter(self.conf.backbone)
            if preprocessor:
                if isinstance(preprocessor, alb.Compose):
                    sample = preprocessor(image=image)
                    # image, mask = sample['image'], sample['mask']
                    image = sample['image']
                else:
                    image = preprocessor(image)
                    # Operate possible case when custom preprocessor modified data size
                    if mask is not None:
                        if image.shape[:2] != mask.shape[:2]:
                            print('WARNING: Mask has not matched image resolution. To match shape it was scaled.')
                            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __len__(self):
        return self._length

    def get_fname(self, i):
        keys = list(self.src_data)
        if len(keys) == 0:
            return 0

        i = i % len(self)
        return os.path.basename(self.src_data[keys[0]][i])

    def get_color(self, class_ind):
        return self.class_colors[class_ind]


class SemanticSegmentationSingleDataProvider(SemanticSegmentationDataProvider):
    def __init__(self, data_reader, img_fname, himg_fname, configure, prep_getter):
        super().__init__(data_reader=data_reader,
                         data_dir='',
                         ids=list(),
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
