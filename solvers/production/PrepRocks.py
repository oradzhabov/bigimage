import gc
import numpy as np
import cv2
import segmentation_models as sm
import keras
from .model import create_model_rocks


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
    prob_field = (prob_field * 255).astype(np.uint8).squeeze()

    image = prob_field.copy()

    # Detect zero-crossing
    # https://stackoverflow.com/a/48440931/5630599
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel = np.ones((3, 3))
    LoG = cv2.Laplacian(image, cv2.CV_16S)
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, kernel)
    maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, kernel)
    zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))
    # zeroCross = np.logical_and(minLoG < 0, LoG > 0)  # left only one condition
    if debug:
        cv2.imwrite('zeroCross.png', zeroCross.astype(np.uint8)*255)
    del LoG
    del minLoG
    del maxLoG
    gc.collect()

    # Find first derive
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    # Here said(see Notes) that cv2.Scharr better than cv2.Sobel
    # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#formulation
    grad_x = cv2.Scharr(image, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    del grad_x
    grad_y = cv2.Scharr(image, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    del grad_y
    grad_magn = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    del abs_grad_x
    del abs_grad_y
    gc.collect()

    # Create mask where zeros are crossed EXCEPT flat area (top rocks regions)
    mask = np.logical_and(zeroCross, grad_magn > 127)  # 32 or 127
    del grad_magn
    del zeroCross
    gc.collect()

    # Filter zero-crossed areas except flat area (top rocks regions)
    image[mask] = 0
    del mask

    # Filter low-probability
    (_, prob_field_th) = cv2.threshold(prob_field, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image[prob_field_th == 0] = 0
    del prob_field_th

    # Filter tiny/noise points
    # image = cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((3,3)))
    # image = cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((3,3)))
    # Binarize image
    image[image > 0] = 255
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # image = filter_pix_noise(image)
    # image = cv2.bitwise_not(filter_pix_noise(cv2.bitwise_not(image)))
    if debug:
        cv2.imwrite('image.png', image)

    # Marker labelling
    ret, markers = cv2.connectedComponents(image)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # To quickly mark contours by negative instance index, find zero-cross by binarized image
    LoG = cv2.Laplacian(image, cv2.CV_16S)
    minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3, 3)))
    zeroCross = np.logical_and(minLoG < 0,  LoG > 0)
    markers[zeroCross] = 0
    del LoG
    del minLoG
    del zeroCross
    gc.collect()

    t = np.asarray(np.dstack((prob_field, prob_field, prob_field)), dtype=np.uint8)  # img
    instances = cv2.watershed(t, markers)
    if debug:
        t[instances == -1] = [255, 0, 0]
        cv2.imwrite('t.png', t)
    del markers
    del t
    gc.collect()

    (t, prob_field_th) = cv2.threshold(prob_field, 255*0.3, 255, cv2.THRESH_BINARY)
    if debug:
        cv2.imwrite('prob_field_th.png', prob_field_th)

    # shape: [w,h], dtype: int32
    return instances, prob_field_th


def collect_statistics(instances, img=None, debug=False, orthophoto_filename=''):
    print('Collect statistics...')
    geometry_px = []
    instMax = np.max(instances) + 1

    # Since instance range is [-1 ... +RocksNb] we need to convert it to type which can represent negative values
    instances_f32 = instances.astype(np.float32)
    # Threshold allow f32 input
    ret, mask = cv2.threshold(instances_f32, 1, 255, 0)
    del instances_f32
    gc.collect()
    # Threshold return type as input type. Simplify it
    mask = mask.astype(np.uint8)
    # Magic trick to speed up contour detection - make rocks are black
    mask = 255 - mask
    if debug:
        cv2.imwrite('instance_mask.png', mask)

    findMode = cv2.RETR_TREE  # Contour retrieval mode
    findAlg = cv2.CHAIN_APPROX_SIMPLE  # Contour approximation method
    if cv2.__version__.startswith("3"):
        im, contours, hierarchy = cv2.findContours(mask, findMode, findAlg)
    else:
        contours, hierarchy = cv2.findContours(mask, findMode, findAlg)
    del mask
    """
    #
    # To describe relation in hierarhy with type cv2.RETR_TREE
    # hierarchy[0][i] = [next sibling, prev sibling, child, parent]
    #
    grand = [contours[i] for i in range(len(contours)) if hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] < 0] # NO parents HAVE childs
    cv2.drawContours(img, grand, -1, (0,0,255), 3)
    print('len(grand): ', len(grand))
    holes = [contours[i] for i in range(len(contours)) if hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0] # HAVE parents NO childs
    cv2.drawContours(img, holes, -1, (0,255,0), 1)
    print('len(holes): ', len(holes))
    ones = [contours[i] for i in range(len(contours)) if hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0] # NO parents NO childs
    cv2.drawContours(img, ones, -1, (0,255,255), 1)
    print('len(ones): ', len(ones))
    siblings = [contours[i] for i in range(len(contours)) if hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0 and (hierarchy[0][i][0] >= 0 or hierarchy[0][i][1] >= 0) ] # HAVE parents NO childs HAVE SIBLINGS
    cv2.drawContours(img, siblings, -1, (255,255,0), 1)
    print('len(siblings): ', len(siblings))
    """
    # Pay attention - if rocks are black which put on white background -
    # each rock will be a child, and main parent - image rectangle
    rocks = [contours[i] for i in range(len(contours)) if
             hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0]  # HAVE parents NO childs
    # Collect rocks which are not main parent(image rect) i.e. HAVE parents and HAVE childs -
    # it could be big rocks, on which the rocks are smaller
    rocks2 = [contours[i] for i in range(len(contours)) if
              hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] >= 0]  # HAVE parents HAVE childs
    rocks = rocks + rocks2

    for cnt in rocks:
        #
        # Collect geometry
        #
        # Rotated Rectangle
        rect = cv2.minAreaRect(cnt)  # output is: ((x, y), (w, h), angle)
        # Pay attention that even 0-size means 1-pixel, hence min available diameters eq 1
        minDiameter = max(np.min(rect[1]), 1)
        maxDiameter = max(np.max(rect[1]), 1)

        # CenterX, CenterY, MinD, MaxD
        item = [rect[0][0], rect[0][1], minDiameter, maxDiameter, cnt.tolist()]
        geometry_px.append(item)

        if img is not None:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 1)

    if debug:
        print('Number of rocks: {}'.format(len(rocks)))

    if img is not None:
        cv2.drawContours(img, rocks, -1, (0, 255, 0), 1)

    return geometry_px


def postprocessing_prod_rock(prob_field):
    inst, _ = instance_segmentation(prob_field, debug=False)
    # ptob = ptob.astype(np.float32) / 255.0
    geometry_px = collect_statistics(inst)
    inst = np.zeros_like(prob_field)
    for g in geometry_px:
        cntr = np.array(g[4], dtype=np.int32)
        cv2.fillPoly(inst, pts=[cntr], color=1.0)

    return inst


def preprocess(ximg):
    # map to BGRA, since production uses BGRA channels order instead of this pipeline(RGBx)
    ximg = np.concatenate((ximg[..., :3][..., ::-1], ximg[..., 3:]), axis=-1)

    bgr = ximg[..., :3]
    bgr[bgr[:, :, 2] == 0] = [127, 127, 127]  # todo: ???

    ximg_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(ximg_hsv)  # Saturation range is [0,255] and Value range is [0,255]

    # Switch statistic to the best one
    mean, stddev = cv2.meanStdDev(val)
    val_f32 = val.astype(np.float32)
    val_f32 = (val_f32 - mean[0][0]) / stddev[0][0]
    val_f32 = (val_f32 * 33.0) + 136.0
    val_f32[val_f32 > 255.0] = 255
    val_f32[val_f32 < 0] = 0
    val = val_f32.astype(np.uint8)

    xhimg = ximg[..., 3]
    ximg = cv2.merge([val, val, xhimg])

    return ximg


def get_preprocessing_production_rock(_):
    return preprocess


def create_model_production_rock(conf, compile_model=True):
    n_classes = 1
    model = create_model_rocks(img_hw=None, input_channels=3, dropout_value=0.0)

    metrics = None
    if compile_model:
        # define optimizer
        optimizer = keras.optimizers.Adam(conf.lr)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        # dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        # total_loss = dice_loss + (3 * focal_loss)
        total_loss = focal_loss
        # total_loss = 'binary_crossentropy'

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile model with defined optimizer, loss and metrics
        model.compile(optimizer, total_loss, metrics)

    return model, metrics
