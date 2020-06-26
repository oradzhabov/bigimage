import numpy as np
import cv2
import matplotlib.pyplot as plt


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
