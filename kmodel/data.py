import os
import cv2
import random
import numpy as np
import keras
from .kutils import get_tiled_bbox
from sklearn.model_selection import train_test_split
from osgeo import gdal
import multiprocessing as mp
import albumentations as alb
import segmentation_models as sm
from .parallel import SimpleProcessor


def get_ids(root_folder, subfolder_list):
    ids = [i for i in os.listdir(os.path.join(root_folder, 'imgs')) if os.path.splitext(i)[1] == '.png']
    ids_filtered = list()
    for fn in ids:
        existance = [os.path.isfile(os.path.join(root_folder, subfolder, fn)) for subfolder in subfolder_list]
        if all(existance):
            ids_filtered.append(fn)

    # Ensure reproducible result list because different OS iterate files with difference
    ids_filtered = sorted(ids_filtered)

    return ids_filtered


def read_image(geotiff_path):
    if not os.path.isfile(geotiff_path):
        print('ERROR: File {} does not exist'.format(geotiff_path))
        return None

    gtif = gdal.Open(geotiff_path)

    # Read data
    im = gtif.ReadAsArray()  # channel first
    del gtif

    if im.ndim > 2:
        im = np.rollaxis(im, 0, 3)  # channel last

    if im.dtype != np.uint8:
        im = im.astype(np.uint8)

    # RGB->BGR
    im[..., [0, 1, 2]] = im[..., [2, 1, 0]]

    return im


def crop(input_root, output_root, subfolder_list, file_names, output_shape):
    print('Cropping dataset by size: {}'.format(output_shape))

    for subfolder in subfolder_list:
        # Left output subfolders if it already exist.
        # It allows to extend dataset by ne mask-subfolders
        if os.path.isdir(os.path.join(output_root, subfolder)):
            continue
        for fname in file_names:
            fi_path = os.path.join(input_root, subfolder, fname)

            img = read_image(fi_path)
            if img is None:
                continue

            w0, w1, h0, h1 = get_tiled_bbox(img.shape, output_shape, output_shape)
            crop_img_idx = 0
            for i in range(len(w0)):
                cr_x, extr_x = (w0[i], 0) if w0[i] >= 0 else (0, -w0[i])
                cr_x2, extr_x2 = (w1[i], 0) if w1[i] < img.shape[1] else (img.shape[1], w1[i] - img.shape[1])

                cr_y, extr_y = (h0[i], 0) if h0[i] >= 0 else (0, -h0[i])
                cr_y2, extr_y2 = (h1[i], 0) if h1[i] < img.shape[0] else (img.shape[0], h1[i] - img.shape[0])

                patch = cv2.copyMakeBorder((img[cr_y:cr_y2, cr_x:cr_x2]), extr_y, extr_y2,
                                           extr_x, extr_x2, cv2.BORDER_CONSTANT, value=0)

                basename = os.path.basename(fi_path)
                fname_wo_ext = basename[:basename.index('.')]
                fname_ext = basename[basename.index('.'):]
                patch_path = os.path.join(output_root, subfolder, fname_wo_ext + '_{}'.format(crop_img_idx) + fname_ext)
                crop_img_idx = crop_img_idx + 1
                if not os.path.isdir(os.path.dirname(patch_path)):
                    os.makedirs(os.path.normpath(os.path.dirname(patch_path)))
                if not os.path.isfile(patch_path):
                    cv2.imwrite(patch_path, patch)


def get_cropped_ids(conf):
    # Actually here we could control the content of required sub-folders according to enabling height-map.
    # But it will conflict with mixing train/test data subsets when model partially trained without height-map and
    # then additionally train with height-map.
    subfolder_list = ('imgs', 'himgs', 'masks.{}'.format(conf.data_subset))
    output_folder = os.path.join(conf.data_dir, '.train.crop_wh{}'.format(conf.img_wh_crop))

    # Crop source data
    ids = get_ids(conf.data_dir, subfolder_list)

    crop(conf.data_dir,
         output_folder,
         subfolder_list,
         ids,
         [conf.img_wh_crop, conf.img_wh_crop])

    # Redirect data
    ids = get_ids(output_folder, subfolder_list)

    return output_folder, ids


def get_data(conf, test_size):
    def asd(y_list):
        class_frequencies = dict()
        stratify = list()
        for fname_ind, fname in enumerate(y_list):
            mask_fname = os.path.join(data_dir, 'masks.{}'.format(conf.data_subset), fname)
            y = cv2.imread(mask_fname, cv2.IMREAD_GRAYSCALE)

            classes, y_indices = np.unique(y, return_inverse=True)
            n_classes = classes.shape[0]
            class_counts = np.bincount(y_indices)
            strat_item = np.zeros(shape=(conf.cls_nb), dtype=np.int32)
            for cl_ind, cl in enumerate(classes):
                if cl not in class_frequencies:
                    class_frequencies[cl] = 0
                class_frequencies[cl] = class_frequencies[cl] + class_counts[cl_ind]
                strat_item[cl_ind] = class_counts[cl_ind]

            # strat_item_sum = sum(strat_item)
            # strat_item = [si / strat_item_sum for si in strat_item]
            stratify.append(strat_item)

        return class_frequencies, np.array(stratify)

    # Crop source data(if necessary)
    data_dir, ids = get_cropped_ids(conf)

    # stat, stratify = asd(ids)
    # print(stat)

    # Split Train/Test data
    ids_train, ids_test, _, _ = train_test_split(ids, ids, test_size=test_size, random_state=conf.seed, shuffle=True)

    return data_dir, ids_train, ids_test


class Dataset(object):
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

    def __init__(
            self,
            data_reader,
            data_dir,
            ids,
            conf,
            min_mask_ratio=0.0,
            augmentation=None,
            prep_getter=sm.get_preprocessing
    ):
        Dataset._initializer(self, data_reader, augmentation, conf, prep_getter)

        for fn in ids:
            image_fn = os.path.join(data_dir, 'imgs', fn)
            himage_fn = os.path.join(data_dir, 'himgs', fn) if self.conf.use_heightmap else None
            mask_fn = os.path.join(data_dir, 'masks.{}'.format(self.conf.data_subset), fn)
            if os.path.isfile(image_fn) and \
                    (himage_fn is None or os.path.isfile(himage_fn)) and \
                    os.path.isfile(mask_fn):

                # Check is data not empty
                img = cv2.imread(image_fn)
                std_data = np.max(cv2.meanStdDev(img)[1])
                if himage_fn:
                    img = cv2.imread(himage_fn)
                    std_himg = np.max(cv2.meanStdDev(img)[1])
                    std_data = max(std_data, std_himg)

                if std_data > 0.0:
                    img = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
                    mask_nonzero_nb = np.count_nonzero(img)
                    mask_nonzero_ratio = mask_nonzero_nb / img.size
                    if mask_nonzero_ratio >= min_mask_ratio:
                        self.images_fps.append(image_fn)
                        self.himages_fps.append(himage_fn)
                        self.masks_fps.append(mask_fn)

    def get_fname(self, i):
        i = i % len(self.images_fps)
        return os.path.basename(self.images_fps[i])

    def get_color(self, class_ind):
        return self.class_colors[class_ind]

    def __getitem__(self, i):
        i = i % len(self.images_fps)

        img_path = self.images_fps[i] if i < len(self.images_fps) else None
        himg_path = self.himages_fps[i] if i < len(self.himages_fps) else None
        mask_path = self.masks_fps[i] if i < len(self.masks_fps) else None

        image, mask = self.data_reader(img_path, himg_path, mask_path)

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
        return len(self.images_fps)


class DataSingle(Dataset):
    def __init__(self, data_reader, img_fname, himg_fname, configure, prep_getter=sm.get_preprocessing):
        super().__init__(data_reader=data_reader,
                         data_dir='',
                         ids=list(),
                         conf=configure,
                         min_mask_ratio=0.0,
                         augmentation=None,
                         prep_getter=prep_getter)

        self.images_fps.append(img_fname)
        self.himages_fps.append(himg_fname)


def dataloder_loader_per_process(ds, ind):
    return ds[ind]


class Dataloder(keras.utils.Sequence):
    @staticmethod
    def get_cpu_units_nb():
        return mp.cpu_count()

    def __init__(self, dataset, batch_size=1, shuffle=False, cpu_units_nb=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.pool = None
        if cpu_units_nb > 0:
            self.pool = mp.Pool(min(batch_size, cpu_units_nb))

            i = 0
            start = i * self.batch_size
            stop = (i + 1) * self.batch_size

            self.loader = SimpleProcessor(dataloder_loader_per_process,
                                          [(self.dataset, self.indexes[j % len(self.indexes)])
                                           for j in range(start, stop)],
                                          self.pool)
            self.loader.start(self.batch_size)

        self.on_epoch_end()

    def __del__(self):
        if self.pool:
            self.pool.close()

    def __getitem__(self, i):
        # collect batch data

        data = []
        if self.pool:
            """
            data = self.pool.starmap_async(dataloder_loader_per_process,
                                           [(self.dataset, self.indexes[j % len(self.indexes)])
                                            for j in range(start, stop)]).get()
            """
            data = [x for x in self.loader.get(timeout=3000)]
            #
            # Start preparing next batch in shadow processes
            start = (i + 1) * self.batch_size
            stop = (i + 2) * self.batch_size
            self.loader = SimpleProcessor(dataloder_loader_per_process,
                                          [(self.dataset, self.indexes[j % len(self.indexes)])
                                           for j in range(start, stop)],
                                          self.pool)
            self.loader.start(self.batch_size)
        else:
            start = i * self.batch_size
            stop = (i + 1) * self.batch_size
            for j in range(start, stop):
                data.append(self.dataset[self.indexes[j % len(self.indexes)]])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
