import numpy as np
import cv2
import albumentations as alb
from functools import wraps


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


@preserve_shape
def gamma_transform(img, gamma, eps=1e-7):
    if img.dtype == np.uint8:
        inv_gamma = 1.0 / (gamma + eps)
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** inv_gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)

    return img


class RandomGamma2(alb.RandomGamma):
    def apply(self, image, gamma=1, **params):
        if image.shape[2] < 3:
            return image
        return np.concatenate((gamma_transform(image[..., :3], gamma=gamma, eps=self.eps),
                               image[..., 3:]),
                              axis=-1)

