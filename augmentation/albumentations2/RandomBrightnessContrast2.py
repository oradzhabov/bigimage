import numpy as np
import cv2
import albumentations as alb
from functools import wraps
from .functionality import clip

MAX_VALUES_BY_DTYPE = {
    np.dtype('uint8'): 255,
    np.dtype('uint16'): 65535,
    np.dtype('uint32'): 4294967295,
    np.dtype('float32'): 1.0,
}


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def preserve_shape(func):
    """
    Preserve shape of the image

    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


@clipped
def _brightness_contrast_adjust_non_uint(img, alpha=1., beta=0., beta_by_max=False):
    dtype = img.dtype
    img = img.astype('float32')

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


@preserve_shape
def _brightness_contrast_adjust_uint(img, alpha=1., beta=0., beta_by_max=False):
    dtype = np.dtype('uint8')

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype('float32')

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def brightness_contrast_adjust(img, alpha=1., beta=0., beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)
    else:
        return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


class RandomBrightnessContrast2(alb.RandomBrightnessContrast):
    def get_params_dependent_on_targets(self, params):
        super().get_params_dependent_on_targets(params)

    def apply(self, image, alpha=1., beta=0., **params):
        if image.shape[2] < 3:
            return image
        return np.concatenate((brightness_contrast_adjust(image[..., :3], alpha, beta, self.brightness_by_max),
                               image[..., 3:]),
                              axis=-1)
