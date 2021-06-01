import math


def is_bbox_intersected(bbox1, bbox2):
    xy1, wh1 = bbox1
    xy2, wh2 = bbox2
    return not (xy1[0] + wh1[0] < xy2[0] or xy1[0] > xy2[0] + wh2[0] or
                xy1[1] + wh1[1] < xy2[1] or xy1[1] > xy2[1] + wh2[1])


def _get_tiled_bbox(img_shape, tile_size, offset):
    aw0 = []
    aw1 = []
    ah0 = []
    ah1 = []
    adjust_extra = True
    for i in range(int(math.ceil(1 + (img_shape[0] - tile_size[0])/(offset[0] * 1.0)))):
        for j in range(int(math.ceil(1 + (img_shape[1] - tile_size[1])/(offset[1] * 1.0)))):
            if adjust_extra:
                h1 = offset[0] * i + tile_size[0]
                h0 = h1 - tile_size[0]
                w1 = offset[1] * j + tile_size[1]
                w0 = w1 - tile_size[1]
            else:
                h1 = min(offset[0] * i + tile_size[0], img_shape[0])
                h0 = max(0, h1 - tile_size[0])
                w1 = min(offset[1] * j + tile_size[1], img_shape[1])
                w0 = max(0, w1 - tile_size[1])
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


def get_tiled_bbox(src_img_shape, crop_size_px, overlap_px, roi_bbox_array=None, return_extr=False):
    offset_size_px = (max(crop_size_px[0] - overlap_px, crop_size_px[0] // 2),
                      max(crop_size_px[1] - overlap_px, crop_size_px[1] // 2))

    w0, w1, h0, h1 = _get_tiled_bbox(src_img_shape, crop_size_px, offset_size_px)

    bbox_list = list()
    extr_list = list()
    for i in range(len(w0)):
        cr_x, extr_x = (w0[i], 0) if w0[i] >= 0 else (0, -w0[i])
        cr_x2, extr_x2 = (w1[i], 0) if w1[i] < src_img_shape[1] else (src_img_shape[1], w1[i] - src_img_shape[1])

        cr_y, extr_y = (h0[i], 0) if h0[i] >= 0 else (0, -h0[i])
        cr_y2, extr_y2 = (h1[i], 0) if h1[i] < src_img_shape[0] else (src_img_shape[0], h1[i] - src_img_shape[0])

        bbox = ((cr_x, cr_y), (cr_x2 - cr_x, cr_y2 - cr_y))

        # Check ROIs if they exist
        if roi_bbox_array is not None:
            intersect = [is_bbox_intersected(roi_bbox, bbox) for roi_bbox in roi_bbox_array]
            # Skip candidate if there is no intersections
            if not (True in intersect):
                continue

        bbox_list.append(bbox)

        if return_extr:
            extr_list.append((extr_x, extr_x2, extr_y, extr_y2))

    if return_extr:
        return bbox_list, extr_list

    return bbox_list
