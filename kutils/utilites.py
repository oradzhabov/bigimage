import numpy as np
import cv2
import matplotlib.pyplot as plt


def nms(image, k=13, remove_plateaus_delta=-1.0):
    # https://stackoverflow.com/a/21023493/5630599
    #
    kernel = np.ones((k, k))
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


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def visualize(title, **images):
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
    plt.show()


def get_contours(mask_u8cn, find_alg=cv2.CHAIN_APPROX_SIMPLE, find_mode=cv2.RETR_EXTERNAL, inverse_mask=False):
    if len(mask_u8cn.shape) < 3:
        mask_u8cn = mask_u8cn[..., np.newaxis]
    if inverse_mask:
        mask_u8cn = 255 - mask_u8cn

    class_nb = mask_u8cn.shape[2] - 1 if mask_u8cn.shape[2] > 1 else 1
    contours_list = list()

    # Collect contours except background
    for i in range(class_nb):
        ret, thresh = cv2.threshold(mask_u8cn[..., i], 127, 255, cv2.THRESH_BINARY)

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
            print('len(grand): ', len(grand))
            holes = [contours[i] for i in range(len(contours)) if
                     hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0] # HAVE parents NO children
            print('len(holes): ', len(holes))
            ones = [contours[i] for i in range(len(contours)) if
                    hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0] # NO parents NO children
            print('len(ones): ', len(ones))
            siblings = [contours[i] for i in range(len(contours)) if
                        hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0 and
                        (hierarchy[0][i][0] >= 0 or hierarchy[0][i][1] >= 0) ] # HAVE parents NO childs HAVE SIBLINGS
            print('len(siblings): ', len(siblings))
            """
            # Pay attention - if objects are black which put on white background -
            # each objects will be a child, and main parent - image rectangle
            c1 = [contours[i] for i in range(len(contours)) if
                  hierarchy[0][i][2] < 0 and hierarchy[0][i][3] >= 0]  # HAVE parents NO children
            # Collect objects which are not main parent(image rect) i.e. HAVE parents and HAVE children -
            # it could be big objects, on which the objects are smaller
            c2 = [contours[i] for i in range(len(contours)) if
                  hierarchy[0][i][2] >= 0 and hierarchy[0][i][3] >= 0]  # HAVE parents HAVE children
            contours = c1 + c2

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
