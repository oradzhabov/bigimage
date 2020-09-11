import logging
import os
import json
import numpy as np
import cv2
from .PrepareData import get_raster_info


def contains(list1, list2):
    return set(list2).issubset(list1)


def create_json_item(img_fname, cntrs_list, region_attr_mapper):
    item_key = '{}-1'.format(img_fname)
    item_value = dict()
    item_value['filename'] = img_fname
    item_value['size'] = int(-1)

    regions = list()
    for class_ind, class_ctrs in enumerate(cntrs_list):
        for contour in class_ctrs:
            # contour = np.multiply(contour, scale_factor).astype(int)

            epsilon = 3
            contour = cv2.approxPolyDP(contour, epsilon, True)

            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            if len(contour) > 2:
                region = dict()
                shape_attributes = dict()
                region_attributes = dict()
                #
                shape_attributes['name'] = 'polygon'
                shape_attributes['all_points_x'] = [int(pnt[0]) for pnt in contour]
                shape_attributes['all_points_y'] = [int(pnt[1]) for pnt in contour]

                if region_attr_mapper is not None:
                    for k in region_attr_mapper.keys():
                        region_attributes[k] = region_attr_mapper[k][class_ind]

                region['shape_attributes'] = shape_attributes
                region['region_attributes'] = region_attributes
                regions.append(region)
    item_value['regions'] = regions

    return {item_key: item_value}


def get_imgs(json_filename):
    result = list()
    with open(json_filename, 'r') as f:
        filecontent = f.read()
        content = json.loads(filecontent)
        for parameters in content.values():
            # img_filename = os.path.join(os.path.dirname(json_filename), parameters['filename'].rstrip())
            img_filename = parameters['filename'].rstrip()
            result.append(os.path.basename(img_filename))
    return result


def convert_to_images(json_filename, json_mppx, region_attr_mapper, mask_postprocess=None, preview=False):
    with open(json_filename, 'r') as f:
        filecontent = f.read()
        content = json.loads(filecontent)
        for parameters in content.values():
            img_filename = os.path.join(os.path.dirname(json_filename), parameters['filename'].rstrip())
            img_filename = os.path.normpath(img_filename)

            if not os.path.exists(img_filename):
                logging.info('Image {} not found. Try to find it in sibling of parent'.format(img_filename))

                img_filename = os.path.join(os.path.dirname(img_filename), '../imgs', os.path.basename(img_filename))
                img_filename = os.path.abspath(img_filename)
                if not os.path.exists(img_filename):
                    logging.error('Image {} not found'.format(img_filename))
                    continue

            out_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.png'
            out_filename = os.path.join(os.path.dirname(json_filename), out_filename)
            contours_map = dict()
            for region in parameters['regions']:
                class_ind = 0

                if region_attr_mapper is not None:
                    region_attrs = region['region_attributes']
                    region_attr_int = {k: (region_attr_mapper[k].index(v) if v in region_attr_mapper[k] else -1)
                                       if k in region_attr_mapper else -1 for k, v in region_attrs.items()}
                    if 'class' in region_attr_int:
                        class_ind = region_attr_int['class']

                if class_ind >= 0:
                    if class_ind not in contours_map:
                        contours_map[class_ind] = list()

                    attrs = region['shape_attributes']
                    if contains(attrs, ('all_points_x', 'all_points_y')):
                        np_arr = zip(*(attrs['all_points_x'], attrs['all_points_y']))
                        contours_map[class_ind].append(np.array(list(np_arr)))
                    elif contains(attrs, ('x', 'y', 'width', 'height')):
                        x, y, w, h = attrs['x'], attrs['y'], attrs['width'], attrs['height']
                        np_arr = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                        contours_map[class_ind].append(np.array(np_arr))

            img_shape, img_mppx = get_raster_info(img_filename)
            if img_shape is None:
                logging.error('File {} does not have raster info'.format(img_filename))

            contours_map = {k: [np.multiply(c, json_mppx / img_mppx).astype(np.int32) for c in v]
                            for k, v in contours_map.items()}

            img = np.zeros(img_shape, dtype=np.uint8)
            cntrs_nb = 0
            for k, v in contours_map.items():
                color = 255 - k
                cv2.fillPoly(img, pts=v, color=color)
                cntrs_nb = cntrs_nb + len(v)
            #
            if mask_postprocess is not None:
                logging.info('Extra processing mask file {}'.format(out_filename))
                img = mask_postprocess(img)

            cv2.imwrite(out_filename, img)
            logging.info("Image \"{}\" created successfully. {} classes, {} contours".format(out_filename,
                                                                                             len(contours_map.keys()),
                                                                                             cntrs_nb))

            if preview:
                cv2.imshow(out_filename, img)
                cv2.waitKey()
