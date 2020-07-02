import gc
import sys
import numpy as np
import cv2
from . import RegrSolver
sys.path.append(sys.path[0] + "/..")
from kutils import utilites


def instance_segmentation(prob_field, debug=False):
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


def collect_statistics(instances, debug=False):
    print('Collect statistics...')
    geometry_px = []

    # Since instance range is [-1 ... +RocksNb] we need to convert it to type which can represent negative values
    # cv2.threshold() supports f32 input
    instances_f32 = instances.astype(np.float32)
    ret, mask = cv2.threshold(instances_f32, 1, 255, 0)
    del instances_f32
    gc.collect()
    # Threshold return type as input type. Simplify it
    mask = mask.astype(np.uint8)

    if debug:
        cv2.imwrite('instance_mask.png', mask)

    rocks = utilites.get_contours(mask, find_alg=cv2.CHAIN_APPROX_SIMPLE, find_mode=cv2.RETR_TREE, inverse_mask=True)
    rocks = rocks[0]  # we operate with single class

    if debug:
        img = np.dstack((mask, mask, mask))
        cv2.drawContours(img, rocks, -1, (0, 255, 0), 1)
        cv2.imwrite('mask_rocks.png', img)

    for cnt in rocks:
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

        """
        if img is not None:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
        """

    if debug:
        print('Number of rocks: {}'.format(len(rocks)))

    return geometry_px, mask


def postprocess(prob_field):
    debug = False

    prob_field = prob_field.squeeze()

    inst, _ = instance_segmentation(prob_field, debug=debug)

    geometry_px, mask = collect_statistics(inst, debug=debug)

    # since later it will be used as prediction result, scale [0..255] to [0..1]
    mask[mask > 0] = 1

    return mask


class RegrRocksSolver(RegrSolver):
    def __init__(self, conf):
        super(RegrRocksSolver, self).__init__(conf)

    def post_predict(self, pr_result):
        return np.clip(pr_result, 0, 1)

    def get_contours(self, pr_mask):
        pr_mask = postprocess(pr_mask)
        return utilites.get_contours((pr_mask * 255).astype(np.uint8), find_alg=cv2.CHAIN_APPROX_SIMPLE,
                                     find_mode=cv2.RETR_TREE, inverse_mask=True)
