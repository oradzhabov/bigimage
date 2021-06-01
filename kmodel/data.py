import logging
import os
import cv2
import numpy as np
from .kutils import get_tiled_bbox
from sklearn.model_selection import train_test_split
from osgeo import gdal
import queue
import threading
from .. import get_submodules_from_kwargs


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
        logging.error('File {} does not exist'.format(geotiff_path))
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

    # todo: take too much RAM (temporary)
    # RGB(A)->BGR(A)
    if im.ndim == 3:
        if im.shape[2] > 2:  # It could be RGB->BGR or RGBA->BGRA
            im[..., [0, 1, 2]] = im[..., [2, 1, 0]]

    return im


def crop(input_root, output_root, subfolder_list, file_names, output_shape):
    logging.info('Cropping dataset by size: {}'.format(output_shape))

    for subfolder in subfolder_list:
        # Left output subfolders if it already exist.
        # It allows to extend dataset by new mask-subfolders
        if os.path.isdir(os.path.join(output_root, subfolder)):
            continue
        for fname in file_names:
            fi_path = os.path.join(input_root, subfolder, fname)

            img = read_image(fi_path)
            if img is None:
                continue

            crop_img_idx = 0
            bbox_list, extr_list = get_tiled_bbox(img.shape, output_shape, 0, return_extr=True)
            for i in range(len(bbox_list)):
                bbox = bbox_list[i]
                extr_x, extr_x2, extr_y, extr_y2 = extr_list[i]
                cr_x, cr_y = bbox[0]
                cr_w, cr_h = bbox[1]
                cr_x2 = cr_x + cr_w
                cr_y2 = cr_y + cr_h

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

            """
            w0, w1, h0, h1 = get_tiled_bbox(img.shape, output_shape, output_shape)
            crop_img_idx = 0
            src_img_shape = img.shape
            for i in range(len(w0)):
                cr_x, extr_x = (w0[i], 0) if w0[i] >= 0 else (0, -w0[i])
                cr_x2, extr_x2 = (w1[i], 0) if w1[i] < src_img_shape[1] else (src_img_shape[1], w1[i] - src_img_shape[1])

                cr_y, extr_y = (h0[i], 0) if h0[i] >= 0 else (0, -h0[i])
                cr_y2, extr_y2 = (h1[i], 0) if h1[i] < src_img_shape[0] else (src_img_shape[0], h1[i] - src_img_shape[0])

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
            """


def get_cropped_ids(conf):
    # Actually here we could control the content of required sub-folders according to enabling height-map.
    # But it will conflict with mixing train/test data subsets when model partially trained without height-map and
    # then additionally train with height-map.
    subfolder_list = ('imgs', 'himgs', 'masks.{}'.format(conf.data_subset))
    output_folder = os.path.join(conf.data_dir, 'train.crop_wh{}'.format(conf.img_wh_crop))

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
    """
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
    """

    # Crop source data(if necessary)
    data_dir, ids = get_cropped_ids(conf)

    # stat, stratify = asd(ids)
    # logging.info(stat)

    # Split Train/Test data
    ids_train, ids_test, _, _ = train_test_split(ids, ids, test_size=test_size, random_state=conf.seed, shuffle=True)

    # Left specified portion of training data. Since data have been shuffle before(in train_test_split) data will
    # remain in shuffle mode
    if 0.0 < conf.thin_out_train_ratio < 1.0:
        ids_train = ids_train[:int(len(ids_train) * conf.thin_out_train_ratio)]

    return data_dir, ids_train, ids_test


def dataloder_loader_per_process(ds, ind):
    return ds[ind]


def get_dataloader(**kwarguments):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    class Dataloder(_keras_utils.Sequence):
        def __init__(self, dataset, batch_size=1, shuffle=False, use_multithreading=True):
            logging.info('Dataloder instance created with batch_size:{}, shuffle:{}, use_multithreading:{}'.
                         format(batch_size, shuffle, use_multithreading))

            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(dataset))
            self.use_multithreading = use_multithreading

            if self.use_multithreading:
                self.locker = threading.Lock()  # indices content locker
            self.on_epoch_end()  # Permute data
            #
            # Prepare and start threads for speed up data preparation
            if self.use_multithreading:
                self.queue = queue.Queue()
                self.life_queue = queue.Queue()

                def do_life():
                    def data_access(index):
                        self.queue.put(self.dataset[index])

                    while True:
                        # Wait for the new batch's request
                        batch_index = self.life_queue.get()
                        #
                        start = batch_index * self.batch_size
                        stop = (batch_index + 1) * self.batch_size

                        self.locker.acquire()
                        try:
                            for j in range(start, stop):
                                t = threading.Thread(target=data_access, args=(self.indexes[j % len(self.indexes)],))
                                t.daemon = True
                                t.start()
                        finally:
                            self.locker.release()

                self.life = threading.Thread(target=do_life)

                self.life.daemon = True
                self.life.start()
                self.life_queue.put(0)  # start gathering data for batch_index = 0

        def __getitem__(self, i):
            # collect batch data
            data = []

            if self.use_multithreading:
                for j in range(self.batch_size):
                    data.append(self.queue.get())

                self.life_queue.put(i + 1)  # start gathering data for next access
            else:
                start = i * self.batch_size
                stop = (i + 1) * self.batch_size
                for j in range(start, stop):
                    data.append(self.dataset[self.indexes[j % len(self.indexes)]])

            # transpose list of lists
            batch = [np.stack(samples, axis=0) for samples in zip(*data)]

            return tuple(batch)

        def __len__(self):
            """Denotes the number of batches per epoch"""
            return len(self.indexes) // self.batch_size

        def on_epoch_end(self):
            """Callback function to shuffle indexes each epoch"""
            if self.shuffle:
                if self.use_multithreading:
                    self.locker.acquire()
                    try:
                        self.indexes = np.random.permutation(self.indexes)
                    finally:
                        self.locker.release()
                else:
                    self.indexes = np.random.permutation(self.indexes)

    return Dataloder
