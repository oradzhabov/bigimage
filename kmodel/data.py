import os
import cv2
import numpy as np
import keras
from .kutils import get_tiled_bbox
from sklearn.model_selection import train_test_split
from osgeo import gdal, osr


def get_ids(root_folder):
    ids = [i for i in os.listdir(os.path.join(root_folder, 'imgs')) if os.path.splitext(i)[1] == '.png']
    ids_filtered = list()
    for fn in ids:
        image_fn = os.path.join(root_folder, 'imgs', fn)
        himage_fn = os.path.join(root_folder, 'himgs', fn)
        mask_fn = os.path.join(root_folder, 'masks', os.path.splitext(fn)[0] + '.mask.png')
        if os.path.isfile(image_fn) and os.path.isfile(himage_fn) and os.path.isfile(mask_fn):
            ids_filtered.append(fn)

    # Ensure reproducible result list because different OS iterate files with difference
    ids_filtered = sorted(ids_filtered)

    return ids_filtered

"""
def get_scaled_img(geotiff_path):
    if not os.path.isfile(geotiff_path):
        print('ERROR: File {} does not exist'.format(geotiff_path))
        return None

    gtif = gdal.Open(geotiff_path)

    # Also load as a gdal image to get geotransform
    # (world file) info
    geoTrans = gtif.GetGeoTransform()

    scale = np.fabs((geoTrans[1], geoTrans[5], 0.0))
    tiepoint = np.array((geoTrans[0], geoTrans[3]))

    # Obtain length UNITS
    prj = gtif.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    unit = srs.GetAttrValue('unit')
    print('GeoTIFF length units: {}'.format(unit))
    scale_to_meter = 1.0
    if unit != 'metre':
        scale_to_meter = 0.3048

    # Rescale to be able read it to numpy
    tmp_down_file = ".tmp_downsample.tif"
    w, h, ch = gtif.RasterXSize, gtif.RasterYSize, gtif.RasterCount
    raster_size = w * h * ch
    # mem_threshold_bytes = 100000000
    mem_threshold_bytes = 10000
    rescale = math.sqrt(mem_threshold_bytes / raster_size)
    if rescale < 1.0:
        gtif = gdal.Translate(tmp_down_file, geotiff_path,
                              options="-outsize {} {} -ot Byte -r nearest -co NUM_THREADS=ALL_CPUS".
                              format(int(w * rescale), int(h * rescale)))
        geoTrans = gtif.GetGeoTransform()

        scale = np.fabs((geoTrans[1], geoTrans[5], 1.0))

    # Read data
    im = gtif.ReadAsArray()  # channel first
    del gtif
    if os.path.isfile(tmp_down_file):
        os.remove(tmp_down_file)

    if im.ndim > 2:
        im = np.rollaxis(im, 0, 3)  # channel last

    if im.dtype != np.uint8:
        im = im.astype(np.uint8)

    # RGB->BGR
    im[..., [0, 1, 2]] = im[..., [2, 1, 0]]

    # meters per pixel, tie point in meters, flag, UINT8 image, original units per meter
    return [im, scale * scale_to_meter]
"""


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
    if os.path.isdir(output_root):
        # shutil.rmtree(output_root)
        return

    print('Cropping dataset by size: {}'.format(output_shape))
    for fname in file_names:
        for subfolder in subfolder_list:
            if subfolder == 'masks':
                fi_path = os.path.join(input_root, subfolder, os.path.splitext(fname)[0] + '.mask.png')
            else:
                fi_path = os.path.join(input_root, subfolder, fname)

            # img = cv2.imread(fi_path)
            img = read_image(fi_path)
            if img is None:
                continue

            w0, w1, h0, h1 = get_tiled_bbox(img.shape, output_shape, output_shape)
            crop_img_idx = 0
            for i in range(len(w0)):
                cr_x, extr_x = (w0[i], 0) if w0[i] >= 0 else (0, -w0[i])
                cr_x2, extr_x2 = (w1[i], 0) if w1[i] < img.shape[1] else (img.shape[1]-1, w1[i] - img.shape[1])

                cr_y, extr_y = (h0[i], 0) if h0[i] >= 0 else (0, -h0[i])
                cr_y2, extr_y2 = (h1[i], 0) if h1[i] < img.shape[0] else (img.shape[0] - 1, h1[i] - img.shape[0])

                patch = cv2.copyMakeBorder((img[cr_y:cr_y2, cr_x:cr_x2]), extr_y, extr_y2,
                                           extr_x, extr_x2, cv2.BORDER_CONSTANT, value=0)
                """
                l_h = h1[i]-h0[i]
                l_w = w1[i]-w0[i]

                patch = cv2.copyMakeBorder((img[h0[i]:h1[i], w0[i]:w1[i]]), 0, output_shape[0] - l_h,
                                           0, output_shape[1] - l_w, cv2.BORDER_CONSTANT, value=0)
                """

                basename = os.path.basename(fi_path)
                fname_wo_ext = basename[:basename.index('.')]
                fname_ext = basename[basename.index('.'):]
                patch_path = os.path.join(output_root, subfolder, fname_wo_ext + '_{}'.format(crop_img_idx) + fname_ext)
                crop_img_idx = crop_img_idx + 1
                if not os.path.isdir(os.path.dirname(patch_path)):
                    os.makedirs(os.path.normpath(os.path.dirname(patch_path)))
                cv2.imwrite(patch_path, patch)


def get_cropped_ids(conf):
    output_folder = os.path.join(conf.data_dir, '.train.crop_wh{}'.format(conf.img_wh_crop))

    # Crop source data
    ids = get_ids(conf.data_dir)

    crop(conf.data_dir,
         output_folder,
         ('imgs', 'himgs', 'masks'),
         ids,
         [conf.img_wh_crop, conf.img_wh_crop])

    # Redirect data
    ids = get_ids(output_folder)

    return output_folder, ids


def get_data(conf, test_size):
    # Crop source data(if necessary)
    data_dir, ids = get_cropped_ids(conf)

    # Split Train/Test data
    ids_train, ids_test, _, _ = train_test_split(ids, ids, test_size=test_size, random_state=42, shuffle=True)

    return data_dir, ids_train, ids_test


class Dataset(object):
    def __init__(
            self,
            data_reader,
            data_dir,
            ids,
            min_mask_ratio=0.0,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = list()
        self.images_fps = list()
        self.himages_fps = list()
        self.masks_fps = list()

        for fn in ids:
            image_fn = os.path.join(data_dir, 'imgs', fn)
            himage_fn = os.path.join(data_dir, 'himgs', fn)
            mask_fn = os.path.join(data_dir, 'masks', os.path.splitext(fn)[0] + '.mask.png')
            if os.path.isfile(image_fn) and os.path.isfile(himage_fn) and os.path.isfile(mask_fn):
                img = cv2.imread(mask_fn)
                mean_mask, std_mask = cv2.meanStdDev(img)
                if np.max(mean_mask) / 255.0 >= min_mask_ratio:
                    self.ids.append(fn)
                    self.images_fps.append(image_fn)
                    self.himages_fps.append(himage_fn)
                    self.masks_fps.append(mask_fn)
                else:
                    pass
                    # print('Not acceptable images mask mean value: {}'.format(mean_mask))

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.data_reader = data_reader

    def __getitem__(self, i):
        i = i % len(self.images_fps)

        img_path, himg_path, mask_path = self.images_fps[i], self.himages_fps[i], self.masks_fps[i]

        image, mask = self.data_reader(img_path, himg_path, mask_path)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply pre-processing
        if self.preprocessing:
            # sample = self.preprocessing(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
            image = self.preprocessing(image)

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size

        data = []
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
