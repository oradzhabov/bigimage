import numpy as np
import math
import cv2


def get_tiled_bbox(img_shape, tile_size, offset):
    aw0 = []
    aw1 = []
    ah0 = []
    ah1 = []
    adjust_extra = True
    for i in range(int(math.ceil(1 + (img_shape[0] - tile_size[1])/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(1 + (img_shape[1] - tile_size[0])/(offset[0] * 1.0)))):
            if adjust_extra:
                h1 = offset[1] * i + tile_size[1]
                h0 = h1 - tile_size[1]
                w1 = offset[0] * j + tile_size[0]
                w0 = w1 - tile_size[0]
            else:
                h1 = min(offset[1] * i + tile_size[1], img_shape[0])
                h0 = max(0, h1 - tile_size[1])
                w1 = min(offset[0] * j + tile_size[0], img_shape[1])
                w0 = max(0, w1 - tile_size[0])
            aw0.append(w0)
            aw1.append(w1)
            ah0.append(h0)
            ah1.append(h1)
    if adjust_extra:
        w_extra_half = (max(aw1) - img_shape[1]) // 2
        aw0 = [i - w_extra_half for i in aw0]
        aw1 = [i - w_extra_half for i in aw1]
        h_extra_half = (max(ah1) - img_shape[0]) // 2
        ah0 = [i - h_extra_half for i in ah0]
        ah1 = [i - h_extra_half for i in ah1]

    return aw0, aw1, ah0, ah1


def get_contours(mask_u8cn):
    if len(mask_u8cn.shape) < 3:
        mask_u8cn = mask_u8cn[..., np.newaxis]

    class_nb = mask_u8cn.shape[2] - 1 if mask_u8cn.shape[2] > 1 else 1
    contours_list = list()

    # Collect contours except background
    for i in range(class_nb):
        ret, thresh = cv2.threshold(mask_u8cn[..., i], 127, 255, cv2.THRESH_BINARY)

        if cv2.__version__.startswith("3"):
            im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        else:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        contours_list.append(contours)

    return contours_list


def write_text(img_rgb, text, bottom_left_corner_of_text, fontColor, font_scale=1):
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
                fontColor,
                thickness=1,
                lineType=line_type)
