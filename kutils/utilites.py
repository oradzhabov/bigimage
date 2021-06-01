import gc
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from kmodel import kutils
from tqdm import tqdm
from joblib import Parallel, delayed


def nms(image, k=13, remove_plateaus_delta=-1.0):
    # https://stackoverflow.com/a/21023493/5630599
    #
    kernel = np.ones((k, k))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    mask = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
    mask = cv2.compare(image, mask, cv2.CMP_GE)
    if remove_plateaus_delta >= 0.0:
        kernel = np.ones((k, k))
        non_plateau_mask = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)

        # non_plateau_mask = cv2.compare(image, non_plateau_mask, cv2.CMP_GT)
        cond = (image - non_plateau_mask) > remove_plateaus_delta
        non_plateau_mask = cond.astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, non_plateau_mask)

    return mask


def zero_crossing(image):
    # Detect zero-crossing
    # https://stackoverflow.com/a/48440931/

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel = np.ones((3, 3))
    l_o_g = cv2.Laplacian(image, cv2.CV_32F)
    min_l_o_g = cv2.morphologyEx(l_o_g, cv2.MORPH_ERODE, kernel)
    max_l_o_g = cv2.morphologyEx(l_o_g, cv2.MORPH_DILATE, kernel)
    zero_cross = np.logical_or(np.logical_and(min_l_o_g < 0,  l_o_g > 0), np.logical_and(max_l_o_g > 0, l_o_g < 0))

    return zero_cross


def grad_magn(gray, fx=None, fy=None, ddepth=cv2.CV_32F):
    scale = 1
    delta = 0
    if True:
        # Here said(see Notes) that cv2.Scharr better than cv2.Sobel
        # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#formulation
        if fx is None:
            grad_x = cv2.Scharr(gray, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        else:
            grad_x = cv2.Scharr(fx, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        if fy is None:
            grad_y = cv2.Scharr(gray, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        else:
            grad_y = cv2.Scharr(fy, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        dtype = grad_x.dtype
        np.floor_divide(grad_x, 8 * grad_x.itemsize, grad_x)
        np.floor_divide(grad_y, 8 * grad_y.itemsize, grad_y)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2).astype(dtype) if gray is not None else None
        gc.collect()
        return grad, grad_x, grad_y

    raise NotImplementedError

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # Because Sobel uses kernel with sum of weights 2*4.
    # So result should be divided by 8
    grad_x /= 8.0
    grad_y /= 8.0
    grad = np.sqrt(grad_x**2 + grad_y**2)
    return grad, grad_x, grad_y


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def visualize(title, img_fname, **images):
    """PLot images in one row."""
    img_filtered = {key: value for (key, value) in images.items() if value is not None}
    n = len(img_filtered)
    fig = plt.figure(figsize=(16, 16))
    for i, (name, img) in enumerate(img_filtered.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(img)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    if img_fname is not None:
        plt.savefig(img_fname)
    else:
        plt.show()
    plt.close(fig)


def get_contours(mask_u8cn, find_alg=cv2.CHAIN_APPROX_SIMPLE, find_mode=cv2.RETR_EXTERNAL, inverse_mask=False,
                 crop_size_px=None):
    src_img_shape = mask_u8cn.shape
    overlap_px = 0

    if crop_size_px:
        overlap_px = min(500, min(src_img_shape[0], src_img_shape[1]) // 2)
        # Collect bounding boxes
        bbox_list = kutils.get_tiled_bbox(src_img_shape, crop_size_px, overlap_px)
    else:
        bbox_list = list([((0, 0), (src_img_shape[1], src_img_shape[0]))])

    def _get_contours_job(bbox):
        xy, wh = bbox

        # Separator functions make division by middle of overlap-gap to maximize the probability that matched content
        # have similarity, and assumed as the same in the overlapped-area but could have differences on the
        # boundaries.
        def filter_left_contours(contour):
            maxx = np.max(contour[:, 0, 0])
            return maxx + 1 >= overlap_px // 2

        def filter_right_contours(contour):
            maxx = np.max(contour[:, 0, 0])
            return maxx + 1 < wh[0] - overlap_px // 2

        def filter_top_contours(contour):
            maxy = np.max(contour[:, 0, 1])
            return maxy + 1 >= overlap_px // 2

        def filter_btm_contours(contour):
            maxy = np.max(contour[:, 0, 1])
            return maxy + 1 < wh[1] - overlap_px // 2

        patch = mask_u8cn[xy[1]:xy[1]+wh[1], xy[0]:xy[0]+wh[0]]

        patch_cntrs = _get_contours(patch, find_alg=find_alg, find_mode=find_mode, inverse_mask=inverse_mask)

        # Filter out contours repeated in overlapped areas
        if overlap_px > 0:
            if xy[0] > 0:
                patch_cntrs = [list(filter(filter_left_contours, class_contours)) for class_contours in patch_cntrs]
            if xy[0] + wh[0] < src_img_shape[1]:
                patch_cntrs = [list(filter(filter_right_contours, class_contours)) for class_contours in patch_cntrs]
            if xy[1] > 0:
                patch_cntrs = [list(filter(filter_top_contours, class_contours)) for class_contours in patch_cntrs]
            if xy[1] + wh[1] < src_img_shape[0]:
                patch_cntrs = [list(filter(filter_btm_contours, class_contours)) for class_contours in patch_cntrs]

        if False:
            if len(patch_cntrs) > 0:
                if len(patch_cntrs[0]) > 0:
                    cv2.imwrite('patch.png', patch)
                    image = np.zeros([patch.shape[0], patch.shape[1], 1], dtype=np.uint8)
                    for class_ind, class_ctrs in enumerate(patch_cntrs):
                        cv2.drawContours(image, class_ctrs, -1, (255), 0)
                    cv2.imwrite('patch_cntrs.png', image)

        # Map contours to base CS
        patch_cntrs = [[cntr + bbox[0] for cntr in cntrs] for cntrs in patch_cntrs]

        return patch_cntrs

    use_parallel = True
    if use_parallel:
        with Parallel(n_jobs=min(os.cpu_count(), len(bbox_list)), backend='threading') as executor:
            tasks = (delayed(_get_contours_job)(bbox) for bbox in bbox_list)
            bbox_contours_arr = executor(tasks)
    else:
        bbox_contours_arr = list()
        for bbox in bbox_list:
            bbox_contours_arr.append(_get_contours_job(bbox))

    pr_cntrs_list_px = bbox_contours_arr[0]
    for bbox_contours in bbox_contours_arr[1:]:
        # For each particular class
        for class_ind in range(len(pr_cntrs_list_px)):
            pr_cntrs_list_px[class_ind] = pr_cntrs_list_px[class_ind] + bbox_contours[class_ind]

    return pr_cntrs_list_px


def _get_contours(mask_u8cn, find_alg=cv2.CHAIN_APPROX_SIMPLE, find_mode=cv2.RETR_EXTERNAL, inverse_mask=False):
    if len(mask_u8cn.shape) < 3:
        mask_u8cn = mask_u8cn[..., np.newaxis]
    if inverse_mask:
        mask_u8cn = 255 - mask_u8cn

    class_nb = max(1, mask_u8cn.shape[2] - 1)
    contours_list = list()

    # Collect contours except background
    for i in range(class_nb):
        ret, thresh = cv2.threshold(mask_u8cn[..., i], 127, 255, cv2.THRESH_BINARY)

        # todo: seems OpenCV implements method cv2.findContours() in 1-core environment. Very slow.
        if cv2.__version__.startswith("3"):
            im, contours, hierarchy = cv2.findContours(thresh, find_mode, find_alg)
        else:
            contours, hierarchy = cv2.findContours(thresh, find_mode, find_alg)

        if find_mode == cv2.RETR_TREE:
            #
            # To describe relation in hierarchy with type cv2.RETR_TREE
            # hierarchy[0][i] = [next sibling, prev sibling, child, parent]
            #
            """
            grand = [contours[i] for i in range(len(contours)) if
                     hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] < 0] # NO parents HAVE children
            logging.info('len(grand): {}'.format(len(grand)))
            holes = [contours[i] for i in range(len(contours)) if
                     hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0] # HAVE parents NO children
            logging.info('len(holes): {}'.format(len(holes)))
            ones = [contours[i] for i in range(len(contours)) if
                    hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0] # NO parents NO children
            logging.info('len(ones): {}'.format(len(ones)))
            siblings = [contours[i] for i in range(len(contours)) if
                        hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0 and
                        (hierarchy[0][i][0] >= 0 or hierarchy[0][i][1] >= 0) ] # HAVE parents NO childs HAVE SIBLINGS
            logging.info('len(siblings): {}'.format(len(siblings)))
            """
            use_optimize = True
            if use_optimize:
                i = np.arange(len(contours))
                cci = np.where((hierarchy[0, i, 2] < 0) & (hierarchy[0, i, 3] >= 0))
                c1 = list(map(contours.__getitem__, cci[0]))
                cci = np.where((hierarchy[0, i, 2] >= 0) & (hierarchy[0, i, 3] >= 0))
                c2 = list(map(contours.__getitem__, cci[0]))
            else:
                # Pay attention - if objects are black which put on white background -
                # each objects will be a child, and main parent - image rectangle
                c1 = [contours[i] for i in range(len(contours)) if
                      hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0]  # HAVE parents NO children
                # Collect objects which are not main parent(image rect) i.e. HAVE parents and HAVE children -
                # it could be big objects, on which the objects are smaller
                c2 = [contours[i] for i in range(len(contours)) if
                      hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] >= 0]  # HAVE parents HAVE children
            contours = c1 + c2

        # Filter out non-manifold contours
        contours = list(filter(lambda x: len(x) > 2, contours))

        contours_list.append(contours)

    return contours_list


def filter_thin_bands(contours, img_shape, debug=False):
    # Since filtering could take too much RAM, work in scaled space
    scaled_size = 2048
    scale = min(1, scaled_size / min(img_shape[0], img_shape[1]))

    # Map data to scaled space
    img_shape = (int(img_shape[0]*scale), int(img_shape[1]*scale))
    contours_sc = [(cntr*scale).astype(np.int32) for cntr in contours]
    #
    contours_sc_flt = list()
    for contour in contours_sc:
        if len(contour) < 3:
            continue
        # Get boundaries WITH extra 1-pixel to obtain proper distance for all cases
        minx = np.min(contour[:, 0, 0]) - 1
        miny = np.min(contour[:, 0, 1]) - 1
        maxx = np.max(contour[:, 0, 0]) + 2
        maxy = np.max(contour[:, 0, 1]) + 2
        imgTemp = np.zeros(shape=(maxy-miny, maxx-minx, 1), dtype=np.uint8)
        contour = contour - [minx, miny]

        cv2.fillPoly(imgTemp, pts=[contour], color=255)
        if debug:
            cv2.imwrite('filter_thin_bands_contours.png', imgTemp)

        dist = cv2.distanceTransform(imgTemp, cv2.DIST_L2, 3)
        if debug:
            cv2.imwrite('filter_thin_bands_dist1.png', dist / np.max(dist) * 255)
        max_band_radius_px = np.mean(dist[dist > 0]) + np.std(dist[dist > 0]) * 1

        dist = cv2.distanceTransform((dist < max_band_radius_px).astype(np.uint8), cv2.DIST_L2, 3)
        dist[imgTemp[..., 0] == 0] = 0
        if debug:
            cv2.imwrite('filter_thin_bands_dist2.png', dist / np.max(dist) * 255)

        if debug:
            imgTemp2 = imgTemp.copy()
            imgTemp2[(imgTemp2[..., 0] > 0) & (dist < max_band_radius_px)] = 127
            cv2.imwrite('filter_thin_bands_result_masked.png', imgTemp2)

        imgTemp[dist > max_band_radius_px] = 0
        if debug:
            cv2.imwrite('filter_thin_bands_result.png', imgTemp)
        if cv2.__version__.startswith("3"):
            im, contours, hierarchy = cv2.findContours(imgTemp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        else:
            contours, hierarchy = cv2.findContours(imgTemp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        if len(contours) > 0:
            contours = [cntr + [minx, miny] for cntr in contours]
            contours_sc_flt = contours_sc_flt + contours

    # Map contours back to original scale
    contours = [(cntr / scale).astype(np.int32) for cntr in contours_sc_flt]

    return contours


def filter_small_contours(contours, min_area_px, max_tol_dist_px, img_shape, debug=False):
    # Since filtering could take too much RAM, work in scaled space
    scaled_size = 2048
    scale = min(1, scaled_size / min(img_shape[0], img_shape[1]))

    # Map data to scaled space
    img_shape = (int(img_shape[0]*scale), int(img_shape[1]*scale))
    contours = [(cntr*scale).astype(np.int32) for cntr in contours]
    min_area_px = min_area_px * (scale**2)
    max_tol_dist_px = max_tol_dist_px * scale

    # Filter and approximate contours
    contours_filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area_px:
            # Parameter specifying the approximation accuracy.
            # This is the maximum distance between the original curve and its approximation.
            epsilon = max_tol_dist_px
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if cv2.contourArea(approx) > min_area_px:
                contours_filtered.append(approx)

    # To remove artifacts after approximation(contour can intersect themselves) reconstruct it
    imgTemp = np.zeros(shape=(img_shape[0], img_shape[1], 1), dtype=np.uint8)
    cv2.fillPoly(imgTemp, pts=contours_filtered, color=255)
    if cv2.__version__.startswith("3"):
        im, contours, hierarchy = cv2.findContours(imgTemp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    else:
        contours, hierarchy = cv2.findContours(imgTemp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    gc.collect()
    contours_filtered.clear()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area_px:
            # Parameter specifying the approximation accuracy.
            # This is the maximum distance between the original curve and its approximation.
            epsilon = max_tol_dist_px
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if cv2.contourArea(approx) > min_area_px:
                contours_filtered.append(approx)

    # Sort contours by decreasing area
    if len(contours_filtered) > 1:
        contours_area = [cv2.contourArea(c) for c in contours_filtered]
        # Do not forget "key"-parameter to properly operate with contours which have equal area
        contours_area, contours_filtered = zip(*sorted(zip(contours_area, contours_filtered),
                                                       key=lambda x: x[0], reverse=True))

    # Map contours back to original scale
    contours = [(cntr / scale).astype(np.int32) for cntr in contours_filtered]

    return contours


def filter_contours(contours, min_area_px, max_tol_dist_px, img_shape, debug=False):
    contours = filter_thin_bands(contours, img_shape, debug=debug)

    contours_filtered = filter_small_contours(contours, min_area_px, max_tol_dist_px, img_shape, debug=debug)
    return contours_filtered


def write_text(img_rgb, text, bottom_left_corner_of_text, font_color, font_scale=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = 2

    cv2.putText(img_rgb, text,
                bottom_left_corner_of_text,
                font,
                font_scale,
                (0, 0, 0),
                thickness=4,
                lineType=line_type)

    cv2.putText(img_rgb, text,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                thickness=1,
                lineType=line_type)
