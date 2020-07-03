import numpy as np
import cv2


def read_sample(data_paths, mask_path):
    data_paths = data_paths + [None] * (2 - len(data_paths))
    img_path, himg_path = data_paths

    # read data
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if himg_path is not None:
        himg = cv2.imread(himg_path)
        if len(himg.shape) > 2:
            himg = himg[..., 0][..., np.newaxis]

        if himg.shape[:2] != img.shape[:2]:
            print('WARNING: Height map has not matched image resolution. To match shape it was scaled.')
            himg = cv2.resize(himg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        img = np.concatenate((img, himg), axis=-1)

    mask = None
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).squeeze()

    return img, mask
