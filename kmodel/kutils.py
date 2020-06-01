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


def get_contours(mask_u8c1):
    # Find contours
    ret, thresh = cv2.threshold(mask_u8c1, 127, 255, cv2.THRESH_BINARY)

    if cv2.__version__.startswith("3"):
        im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    else:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    return contours
