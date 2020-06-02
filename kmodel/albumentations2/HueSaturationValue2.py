import numpy as np
import cv2
import albumentations as alb

from .functionality import clip


def _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    lut_hue = np.arange(0, 256, dtype=np.int16)
    lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)

    lut_sat = np.arange(0, 256, dtype=np.int16)
    lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)

    lut_val = np.arange(0, 256, dtype=np.int16)
    lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)

    hue = cv2.LUT(hue, lut_hue)
    sat = cv2.LUT(sat, lut_sat)
    val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    hue = cv2.add(hue, hue_shift)
    hue = np.where(hue < 0, hue + 180, hue)
    hue = np.where(hue > 180, hue - 180, hue)
    hue = hue.astype(dtype)
    sat = clip(cv2.add(sat, sat_shift), dtype, 255 if dtype == np.uint8 else 1.0)
    val = clip(cv2.add(val, val_shift), dtype, 255 if dtype == np.uint8 else 1.0)
    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    if img.dtype == np.uint8:
        return _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)

    return _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift)


class HueSaturationValue2(alb.HueSaturationValue):
    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        if image.shape[2] < 3:
            return image
        return np.concatenate((shift_hsv(image[..., :3], hue_shift, sat_shift, val_shift),
                               image[..., 3:]),
                              axis=-1)
