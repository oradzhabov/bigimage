import gc
import sys
import numpy as np
import cv2
from . import RegrSolver
sys.path.append(sys.path[0] + "/..")
from kutils import utilites
from kmodel.data import read_image


def find_rock_masks(prob_field, prob_glue_th=0.25, debug=False):
    prob_field = prob_field.squeeze()

    # ANN do not trends to smooth result
    # Blur probability to obtain smoothed field.
    prob_field = cv2.blur(prob_field, (3, 3))
    gc.collect()

    # Find gradient vector field
    g, dx, dy = utilites.grad_magn(prob_field, ddepth=cv2.CV_16S)
    g_less_01 = g < 0.1 * 255  # 0.015??  # 0.1
    g_less_05 = g < 0.01 * 255
    del g
    gc.collect()
    prob_field_greate_025 = prob_field > prob_glue_th * 255  # 0.25
    prob_field_greate_005 = prob_field > 0.02 * 255  # 0.05
    if not debug:
        del prob_field
        gc.collect()

    # Find divergence of gradient vector field
    _, divx, divy = utilites.grad_magn(None, dx, dy, ddepth=cv2.CV_16S)
    del dx
    del dy
    gc.collect()
    pdiv = cv2.addWeighted(divx, 0.5, divy, 0.5, 0)
    del divx
    del divy
    gc.collect()

    mask1 = pdiv <= -0.001 * 255  # good for little rocks(with noise as well) but not enough to cover big rocks
    del pdiv
    gc.collect()

    mask2 = np.logical_and(g_less_01, prob_field_greate_025)  # glue for big rocks
    mask3 = np.logical_and(mask1, np.logical_and(prob_field_greate_005, g_less_05))  # filter out noise

    if debug:
        bgr = np.dstack((prob_field, prob_field, prob_field))
        bgr[mask1] = [255, 0, 0]
        bgr[prob_field_greate_025] = [255, 255, 0]
        bgr[mask2] = [0, 255, 0]
        bgr[mask3] = [0, 0, 255]
        # bgr[g2_less_01] = [0, 255, 255]
        cv2.imwrite('bgr.png', bgr)
        del bgr
    #
    return mask3, mask2


def instance_segmentation_old(prob_field, debug=False):
    """
    prob_field: shape:[h,w], dtype: float32, range[0..1]
    """
    """
    print('Instance segmentation...')
    if prob_field is None or prob_field.size == 0:
        print('ERROR: Source array is empty')
        return None
    """
    # Less than 13 predict not all little rocks, but till logic cannot separate them from big blob,
    # we cannot use less value
    k = 13  # 3 good to detect small, but to not "crash" bit 13 is better
    plateaus_delta = 0.009  # 0.009
    sure_fg = utilites.nms(prob_field, k, plateaus_delta)

    # Increase size of local maximum
    k = 3
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_DILATE, np.ones((k, k)))
    if debug:
        cv2.imwrite('sure_fg.png', sure_fg)

    sure_bg = np.ones(shape=prob_field.shape, dtype=np.uint8) * 255
    unknown = sure_bg - sure_fg
    unknown[prob_field < 0.1] = 0  # 0.2(0.15)
    if debug:
        cv2.imwrite('unknown.png', unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    img = unknown + sure_fg
    # cv2.watershed() requires 3-channel data
    markers = cv2.watershed(np.dstack((img, img, img)), markers)

    # img[markers == -1] = [255, 0, 0]

    return markers, img


def collect_statistics(contours):
    geometry_px = []

    for cnt in contours:
        #
        # Collect geometry
        #
        # Rotated Rectangle
        rect = cv2.minAreaRect(cnt)  # output is: ((x, y), (w, h), angle)
        # Pay attention that even 0-size means 1-pixel, hence min available diameters eq 1
        min_diameter = max(np.min(rect[1]), 1)
        max_diameter = max(np.max(rect[1]), 1)

        # CenterX, CenterY, MinD, MaxD
        item = [rect[0][0], rect[0][1], min_diameter, max_diameter, cnt.tolist()]
        geometry_px.append(item)

    return geometry_px


def postprocess_prob_list(prob_list, debug=False):
    prob_th = 0.05 * 255  # 0.05

    # Process the first scale
    img_descriptor = list(filter(lambda x: x['scale'] == 1.0, prob_list))[0]['img']
    pr_field_sc1 = read_image(img_descriptor) if isinstance(img_descriptor, str) else img_descriptor
    pr_field_sc1 = pr_field_sc1.squeeze()
    # Set proper type to guaranty that pipeline will be processed
    if pr_field_sc1.dtype != np.uint8:
        pr_field_sc1 = (pr_field_sc1 * 255).astype(np.uint8)

    # Store shape
    prob_field_shape = pr_field_sc1.shape

    # Update by the first scale result
    prob_field_greate_prob_th = pr_field_sc1 > prob_th

    little_rocks_mask, big_rocks_mask_sc1 = find_rock_masks(pr_field_sc1, prob_glue_th=0.25, debug=debug)
    del pr_field_sc1
    gc.collect()

    # Process the next scale
    img_descriptor = list(filter(lambda x: x['scale'] != 1.0, prob_list))[0]['img']
    pr_field_sc025 = read_image(img_descriptor) if isinstance(img_descriptor, str) else img_descriptor
    pr_field_sc025 = pr_field_sc025.squeeze()
    # Set proper type to guaranty that pipeline will be processed
    if pr_field_sc025.dtype != np.uint8:
        pr_field_sc025 = (pr_field_sc025 * 255).astype(np.uint8)

    # Update by next scale result
    prob_field_greate_prob_th = np.bitwise_or(prob_field_greate_prob_th, pr_field_sc025 > prob_th)

    # Suppress little rocks on the bound of big rocks
    little_rocks_mask[pr_field_sc025 > 0.02 * 255] = 0

    little_rocks_mask_sc025, big_rocks_mask_sc025 = find_rock_masks(pr_field_sc025, prob_glue_th=0.25, debug=debug)
    del little_rocks_mask_sc025
    del pr_field_sc025
    del prob_list
    gc.collect()

    # Merge glue areas of big rocks
    big_rocks_mask_sc1 = np.bitwise_or(big_rocks_mask_sc1, big_rocks_mask_sc025)
    del big_rocks_mask_sc025
    gc.collect()

    img = np.zeros(shape=prob_field_shape[:2], dtype=np.uint8)
    img[little_rocks_mask] = 255  # mark little rocks
    img[big_rocks_mask_sc1] = 255  # glue little rocks together into big rocks
    del little_rocks_mask
    del big_rocks_mask_sc1
    gc.collect()
    #
    # Extend existing rock's boundaries in reasonable(by threshold) area
    #
    # Label each found rock's core
    ret, markers = cv2.connectedComponents(img)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero.
    # So following indices will be in use:
    # 0 - Unknown area(will be filled later)
    # 1 - Background. Absolutely well-known that there are no rocks
    # 2 - Index of first rock
    markers[(img == 0) & prob_field_greate_prob_th] = 0

    # Format the results
    img = prob_field_greate_prob_th.astype(np.uint8) * 255

    if debug:
        cv2.imwrite('prob_field_th.png', img)

    instances = cv2.watershed(np.dstack((img, img, img)), markers)
    del img
    gc.collect()

    mask = instances > 1
    del instances
    gc.collect()
    # Adapt result type
    # Result range maps to proper range[0..1] automatically
    mask = mask.astype(np.uint8)

    if debug:
        cv2.imwrite('instance_mask.png', mask * 255)

    return mask


class RegrRocksSolver(RegrSolver):
    def __init__(self, conf):
        super(RegrRocksSolver, self).__init__(conf)

    def post_predict(self, pr_result):
        return np.clip(pr_result, 0, 1)

    def get_contours(self, pr_mask_list):
        debug = False
        pr_mask = postprocess_prob_list(pr_mask_list, debug)
        return utilites.get_contours((pr_mask * 255).astype(np.uint8), find_alg=cv2.CHAIN_APPROX_SIMPLE,
                                     find_mode=cv2.RETR_TREE, inverse_mask=True)
