import os
import sys
import json
import numpy as np
import cv2
from .PrepareData import maskExt, get_raster_info

EXTENSION = maskExt
BACK_COLOR = (0,)
FRONT_COLOR = (255,)


def getInfo():
    with open(sys.argv[0], 'r') as f:
        while True:
            read = f.readline().strip()
            if read == "":
                break
            yield read


def contains(list1, list2):
    return set(list2).issubset(list1)


def convert(json_filename, json_mppx, preview=False):
    with open(json_filename, 'r') as f:
        filecontent = f.read()
        content = json.loads(filecontent)
        for parameters in content.values():
            img_filename = os.path.join(os.path.dirname(json_filename), parameters['filename'])
            img_filename = os.path.normpath(img_filename)

            if not os.path.exists(img_filename):
                print('ERROR: {} not found'.format(img_filename))
                continue

            out_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.' + EXTENSION
            out_filename = os.path.join(os.path.dirname(json_filename), out_filename)
            contours = []
            for region in parameters['regions']:
                attrs = region['shape_attributes']
                if contains(attrs, ('all_points_x', 'all_points_y')):
                    np_arr = zip(*(attrs['all_points_x'], attrs['all_points_y']))
                    contours.append(np.array(list(np_arr)))
                elif contains(attrs, ('x', 'y', 'width', 'height')):
                    np_arr = [(attrs['x'], attrs['y']), \
                              (attrs['x'] + attrs['width'], attrs['y']), \
                              (attrs['x'] + attrs['width'], attrs['y'] + attrs['height']), \
                              (attrs['x'], attrs['y'] + attrs['height'])]
                    contours.append(np.array(np_arr))

            # img = cv2.imread(img_filename, 0)
            img_shape, img_mppx = get_raster_info(img_filename)
            contours = [np.multiply(c, json_mppx / img_mppx).astype(np.int32) for c in contours]

            img = np.ones(img_shape, dtype=np.uint8) * BACK_COLOR
            cv2.fillPoly(img, pts=contours, color=FRONT_COLOR)
            cv2.imwrite(out_filename, img)
            print("Image \"{0}\" created successfully. {1} contours".format(out_filename, len(contours)))

            if preview:
                cv2.imshow(out_filename, img)
                cv2.waitKey()

"""
if __name__ == "__main__":
    args = sys.argv[1:]
    preview = '-p' in args
    if preview:
        args.pop(args.index('-p'))

    if len(args) == 0 or len(args) > 1:
        print("\n".join(getInfo()))
        raise Exception("Wrong arguments.")
    elif len(args) == 1:
        json_filename = args[0]

    convert(json_filename, preview=preview)
"""