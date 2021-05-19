import gc
import os
import numpy as np
import cv2
import logging
from ..kutils import utilites
from ..kmodel.data import read_image
from .. import get_submodules_from_kwargs
from kmodel import kutils


def find_rock_masks(prob_field, prob_glue_th=0.25, debug=False):
    prob_field = prob_field.squeeze()

    # ANN do not trends to smooth result
    # Blur probability to obtain smoothed field.
    prob_field = cv2.blur(prob_field, (3, 3))
    gc.collect()

    # Find gradient vector field
    g, dx, dy = utilites.grad_magn(prob_field, ddepth=cv2.CV_16S)
    g_less_10 = g < 0.1 * 255
    g_less_01 = g < 0.02 * 255
    del g
    gc.collect()
    prob_field_greate_th = prob_field > prob_glue_th * 255  # 0.25
    prob_field_greate_002 = prob_field > 0.01 * 255  # 0.05
    if not debug:
        del prob_field
        gc.collect()

    # Find divergence of gradient vector field
    _, divx, divy = utilites.grad_magn(None, dx, dy, ddepth=cv2.CV_16S)
    del dx
    del dy
    gc.collect()
    # Function cv2.max(), instead of cv2.addWeighted() will guaranty than further condition (pdiv < 0) will satisfy
    # only when both components (divx and divy) have negative values.
    pdiv = cv2.max(divx, divy)  # cv2.addWeighted(divx, 0.5, divy, 0.5, 0)
    del divx
    del divy
    gc.collect()

    # If the second derivative is less than zero, we have a "convex in the increasing probability direction "
    # projection section. This is a necessary (but not sufficient) condition for determining the found object.
    # Good for little rocks (with noise as well) but not enough to cover big rocks
    mask1 = pdiv < 0
    del pdiv
    gc.collect()

    mask2 = np.logical_and(g_less_10, prob_field_greate_th)  # glue for big rocks
    mask3 = np.logical_and(mask1, np.logical_and(prob_field_greate_002, g_less_01))  # filter out noise

    if debug:
        bgr = np.dstack((prob_field, prob_field, prob_field))
        bgr[mask1] = [255, 0, 0]
        bgr[prob_field_greate_th] = [255, 255, 0]
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
    logging.info('Instance segmentation...')
    if prob_field is None or prob_field.size == 0:
        logging.error('Source array is empty')
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
        item = [int(rect[0][0]), int(rect[0][1]), min_diameter, max_diameter, cnt.tolist()]
        geometry_px.append(item)

    return geometry_px


"""
def postprocess_prob_u8_list(prob_list, debug=False):
    storage_dir = os.path.dirname(prob_list[0]['img'])
    skip_prediction = prob_list[0]['skip_prediction'] if 'skip_prediction' in prob_list[0] else False
    bbox_predictions_img_fname = os.path.join(storage_dir, 'postprocess_prob_list.png')
    if not os.path.isfile(bbox_predictions_img_fname) or not skip_prediction:
        src_img_shape = prob_list[0]['shape']
        crop_size_px = (16384, 16384)

        bbox_list = list()
        overlap = 50
        offset_size_px = (crop_size_px[0] - overlap, crop_size_px[1] - overlap)
        w0, w1, h0, h1 = kutils.get_tiled_bbox(src_img_shape, crop_size_px, offset_size_px)
        for i in range(len(w0)):
            cr_x, extr_x = (w0[i], 0) if w0[i] >= 0 else (0, -w0[i])
            cr_x2, extr_x2 = (w1[i], 0) if w1[i] < src_img_shape[1] else (src_img_shape[1], w1[i] - src_img_shape[1])

            cr_y, extr_y = (h0[i], 0) if h0[i] >= 0 else (0, -h0[i])
            cr_y2, extr_y2 = (h1[i], 0) if h1[i] < src_img_shape[0] else (src_img_shape[0], h1[i] - src_img_shape[0])

            bbox = ((cr_x, cr_y), (cr_x2 - cr_x, cr_y2 - cr_y))
            bbox_list.append(bbox)

        bbox_predictions_fname = os.path.join(storage_dir, 'postprocess_prob_list.tmp')
        bbox_predictions_result = np.memmap(bbox_predictions_fname, dtype=np.uint8, mode='w+', shape=src_img_shape)
        bbox_predictions_result.fill(0)
        del bbox_predictions_result
        one_row_size_bytes = src_img_shape[1] * src_img_shape[2] * np.dtype(np.uint8).itemsize
        gc.collect()

        for bbox in bbox_list:
            patch = _postprocess_prob_list(prob_list, bbox, debug=debug) * 255
            if len(patch.shape) == 2:
                patch = patch[..., np.newaxis]

            xy, wh = bbox
            shape_new = (wh[1], src_img_shape[1], src_img_shape[2])
            offset = xy[1] * one_row_size_bytes
            bbox_predictions_result = np.memmap(bbox_predictions_fname, dtype=np.uint8,
                                                mode='r+', shape=shape_new, offset=offset)
            bbox_predictions_result[0:patch.shape[0], xy[0]:xy[0] + patch.shape[1]] = \
                np.max([bbox_predictions_result[0:patch.shape[0], xy[0]:xy[0] + patch.shape[1]], patch], axis=0)
            del bbox_predictions_result

        bbox_predictions_result = np.memmap(bbox_predictions_fname, dtype=np.uint8, mode='r', shape=src_img_shape)
        cv2.imwrite(bbox_predictions_img_fname, bbox_predictions_result)
        del bbox_predictions_result  # close memmap
        logging.info('Probability postprocessing stored into {}'.format(bbox_predictions_img_fname))
    else:
        logging.info('Probability postprocessing skipped. Result restored from file {}'.format(bbox_predictions_img_fname))
    mask = read_image(bbox_predictions_img_fname)

    return mask
"""


def _postprocess_prob_list(prob_list, bbox=((0, 0), (None, None)), debug=False):
    # 0.01..0.05. 0.01 fills all gaps between rocks.
    # Since source data was pre-processed by CLAHE and median filter, gaps between rocks increased. To suppress this
    # issue and return fulfilled results, decrease background threshold from 5% to 2%.
    # In fact, 5% remain as better precised rock's boundaries.
    prob_th = 0.05 * 255

    # Process the first scale
    img_descriptor = list(filter(lambda x: x['scale'] == 1.0, prob_list))[0]['img']
    pr_field_sc1 = read_image(img_descriptor, bbox) if isinstance(img_descriptor, str) else img_descriptor
    pr_field_sc1 = pr_field_sc1.squeeze()
    # Set proper type to guaranty that pipeline will be processed
    if pr_field_sc1.dtype != np.uint8:
        pr_field_sc1 = (pr_field_sc1 * 255).astype(np.uint8)

    # Store shape
    prob_field_shape = pr_field_sc1.shape

    little_rocks_mask, big_rocks_mask_sc1 = find_rock_masks(pr_field_sc1, prob_glue_th=0.25, debug=debug)
    # Update by the first scale result
    prob_field_greate_prob_th = pr_field_sc1 > prob_th
    del pr_field_sc1
    gc.collect()

    # Process the next scale
    img_descriptor = list(filter(lambda x: x['scale'] != 1.0, prob_list))[0]['img']
    pr_field_sc025 = read_image(img_descriptor, bbox) if isinstance(img_descriptor, str) else img_descriptor
    pr_field_sc025 = pr_field_sc025.squeeze()
    # Set proper type to guaranty that pipeline will be processed
    if pr_field_sc025.dtype != np.uint8:
        pr_field_sc025 = (pr_field_sc025 * 255).astype(np.uint8)

    # Update by next scale result
    prob_field_greate_prob_th = np.bitwise_or(prob_field_greate_prob_th, pr_field_sc025 > prob_th)

    # Suppress little rocks on the bound of big rocks
    little_rocks_mask[pr_field_sc025 > 0.02 * 255] = 0

    # suppress thin area around little rocks to not grow rocks around them
    little_rocks_mask_u8 = little_rocks_mask.astype(np.uint8) * 255
    little_rocks_mask_u8 = cv2.dilate(little_rocks_mask_u8, np.ones((3, 3), np.uint8)) - little_rocks_mask_u8
    prob_field_greate_prob_th[little_rocks_mask_u8 > 0] = 0

    # Up-scaled rocks should have bigger threshold to not merge little rocks: prob_glue_th=0.40
    # But 0.4 (with all other original conditions) do not cover whole area of big rocks and make them smaller.
    # So to resolve this situation, we left 0.25, but apply cv2.medianBlur(gray, 3) for sc_factor < 1.0, and use
    # d4-group for such scale to predict better big rocks and their boundaries to resolve main problem - not merging
    # little rocks.
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
        cv2.imwrite('markers.png', (markers > 1) * 255)

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


def get_solver(**kwarguments):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    from .RegrSolver import get_solver as get_regr_solver
    regr_solver_class = get_regr_solver(**kwarguments)

    class RegrRocksSolver(regr_solver_class):
        def __init__(self, conf):
            super(RegrRocksSolver, self).__init__(conf)

        def post_predict(self, pr_result):
            return np.clip(pr_result, 0, 1)

        def get_contours(self, pr_mask_list):
            class _Array2DDelegate(object):
                def __init__(self, prob_list, debug=False):
                    self.prob_list = prob_list
                    self.debug = debug

                @property
                def shape(self):
                    src_img_shape = self.prob_list[0]['shape']
                    return src_img_shape

                def __getitem__(self, key):
                    if isinstance(key, tuple):
                        # Get the start, stop, and step from the slices
                        key_h = key[0].start, key[0].stop, key[0].step
                        key_w = key[1].start, key[1].stop, key[1].step

                        bbox = ((key_w[0], key_h[0]), (key_w[1] - key_w[0], key_h[1] - key_h[0]))

                        patch = _postprocess_prob_list(self.prob_list, bbox, debug=self.debug) * 255
                        if len(patch.shape) == 2:
                            patch = patch[..., np.newaxis]
                        return patch
                    else:
                        raise NotImplementedError()

            debug = False
            pr_mask_u8 = _Array2DDelegate(pr_mask_list, debug)

            return utilites.get_contours(pr_mask_u8, find_alg=cv2.CHAIN_APPROX_SIMPLE,
                                         find_mode=cv2.RETR_TREE, inverse_mask=True,
                                         crop_size_px=(16384, 16384))

    return RegrRocksSolver
