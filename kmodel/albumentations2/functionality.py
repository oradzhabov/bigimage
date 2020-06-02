import numpy as np


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)
