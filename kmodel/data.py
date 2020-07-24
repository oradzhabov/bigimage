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


def read_image(geotiff_path, bbox=((0, 0), (None, None))):
    if not os.path.isfile(geotiff_path):
        print('ERROR: File {} does not exist'.format(geotiff_path))
        return None

    gtif = gdal.Open(geotiff_path)

    # Read data
    xoff = bbox[0][0]
    yoff = bbox[0][1]
    xsize = bbox[1][0]
    ysize = bbox[1][1]
    im = gtif.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)  # channel first
    del gtif

    if im.ndim > 2:
        im = np.rollaxis(im, 0, 3)  # channel last

    if im.dtype != np.uint8:
        im = im.astype(np.uint8)

    # RGB->BGR
    im[..., [0, 1, 2]] = im[..., [2, 1, 0]]  # todo: take too much RAM (temporary)

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
