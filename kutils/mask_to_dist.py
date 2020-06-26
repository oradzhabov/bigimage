import cv2
import numpy as np
from tqdm import tqdm


def mask_to_dist(img_u8c1):

    find_mode = cv2.RETR_CCOMP  # Contour retrieval mode
    find_alg = cv2.CHAIN_APPROX_SIMPLE  # Contour approximation method
    if cv2.__version__.startswith("3"):
        im, contours, hierarchy = cv2.findContours(img_u8c1, find_mode, find_alg)
    else:
        contours, hierarchy = cv2.findContours(img_u8c1, find_mode, find_alg)

    mask_all = np.zeros_like(img_u8c1, dtype=np.int32)
    for i, cntr in enumerate(contours):
        cv2.fillPoly(mask_all, [cntr], (1 + i))

    dist_all = cv2.distanceTransform(img_u8c1, cv2.DIST_L2, 3)
    for i, cntr in enumerate(tqdm(contours)):
        c = np.where(mask_all == i + 1)
        dist_all[c] /= np.max(dist_all[c])

    dist_all = (dist_all * 255).astype(np.uint8)

    return dist_all
