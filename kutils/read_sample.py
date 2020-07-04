import numpy as np
import cv2
import sys
sys.path.append(sys.path[0] + "/..")
from kmodel.data import read_image


def read_sample(data_paths, mask_path):
    data_paths = data_paths + [None] * (2 - len(data_paths))
    img_path, himg_path = data_paths

    # read data
    img = read_image(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if himg_path is not None:
        himg = read_image(himg_path)
        if len(himg.shape) > 2:
            himg = himg[..., 0][..., np.newaxis]
        if len(himg.shape) == 2:
            himg = himg[..., np.newaxis]

        if himg.shape[:2] != img.shape[:2]:
            print('WARNING: Height map has not matched image resolution. To match shape it was scaled.')
            himg = cv2.resize(himg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        img = np.concatenate((img, himg), axis=-1)

    mask = None
    if mask_path is not None:
        mask = read_image(mask_path).squeeze()

    return img, mask
