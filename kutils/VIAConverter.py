import os
import json
import numpy as np
import cv2
from .PrepareData import get_raster_info


def contains(list1, list2):
    return set(list2).issubset(list1)


def convert(json_filename, json_mppx, region_attr_mapper, preview=False):
    with open(json_filename, 'r') as f:
        filecontent = f.read()
        content = json.loads(filecontent)
        for parameters in content.values():
            img_filename = os.path.join(os.path.dirname(json_filename), parameters['filename'])
            img_filename = os.path.normpath(img_filename)

            if not os.path.exists(img_filename):
                print('ERROR: {} not found'.format(img_filename))
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
            contours_map = {k: [np.multiply(c, json_mppx / img_mppx).astype(np.int32) for c in v]
                            for k, v in contours_map.items()}

            img = np.zeros(img_shape, dtype=np.uint8)
            cntrs_nb = 0
            for k, v in contours_map.items():
                color = 255 - k
                cv2.fillPoly(img, pts=v, color=color)
                cntrs_nb = cntrs_nb + len(v)
            cv2.imwrite(out_filename, img)
            print("Image \"{}\" created successfully. {} classes, {} contours".format(out_filename,
                                                                                      len(contours_map.keys()),
                                                                                      cntrs_nb))

            if preview:
                cv2.imshow(out_filename, img)
                cv2.waitKey()
