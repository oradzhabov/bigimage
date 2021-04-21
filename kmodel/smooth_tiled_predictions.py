# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE


"""Do smooth predictions on an image from tiled prediction patches."""

# ros: input shape fixed


import logging
import os
import numpy as np
import scipy.signal
from tqdm import tqdm

import gc

import cv2
from ..kutils.utilites import denormalize


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False


def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 2)
        wind = wind * wind.transpose(1, 0, 2)
        if PLOT_PROGRESS:
            # For demo purpose, let's look once at the window:
            plt.imshow(wind[:, :, 0], cmap="viridis")
            plt.title("2D Windowing Function for a Smooth Blending of "
                      "Overlapping Patches")
            plt.show()
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    """
    Add borders to img for a "valid" border pattern according to "window_size" and
    "subdivisions".
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    # gc.collect()
    logging.info('Padding aug size: {}. Img size changed from {} to {}'.format(aug, img.shape[:2], ret.shape[:2]))

    if PLOT_PROGRESS:
        # For demo purpose, let's look once at the window:
        plt.imshow(ret)
        plt.title("Padded Image for Using Tiled Prediction Patches\n"
                  "(notice the reflection effect on the padded borders)")
        plt.show()
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    """
    Undo what's done in the `_pad_img` function.
    Image is an np array of shape (x, y, nb_channels).
    """
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    # gc.collect()
    return ret


def _create_ovelap05_mask(x_nb, y_nb):
    """
    To be able analyse the boundary of overlapped areas (which should be merged in special way), this method creates
    the list of masks for detecting boundaries. Result masks will have codes: 1 for corner, 2 for edge, 4-inside
    :param x_nb:
    :param y_nb:
    :return:
    """
    # Fill overlapping mask. Assumes that overlap is 0.5.
    # Each patch represented by square 2x2, so mask with params x_nb=3, y_nb=2 will have next content
    # 1,2,2,1
    # 2,4,4,2
    # 1,2,2,1
    mask = np.zeros(shape=(y_nb + 1, x_nb + 1), dtype=np.uint8)
    for y in range(y_nb):
        for x in range(x_nb):
            mask[y:y+2, x:x+2] += 1

    # Stack the doubled rows. Since result will use entire loop instead of loop in loop(by x_nb and y_nb),
    # make flat array for each patch. So mask created with params x_nb=3, y_nb=2 will have next content
    # 1,2, 2,2, 2,1, 2,4, 4,4, 4,2
    # 2,4, 4,4, 4,2, 1,2, 2,2, 2,1
    result = np.zeros(shape=(2, x_nb*y_nb*2), dtype=np.uint8)
    for y in range(y_nb):
        for x in range(x_nb):
            ind = (y * x_nb + x) * 2
            result[:, ind:ind+2] = mask[y:y+2, x:x+2]

    return result


def _rotate_mirror_do(im):
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func, memmap_batch_size, temp_dir):
    """
    @param memmap_batch_size: Note unet(efficientb3) could process 6 x 1024_wh by 4GB GPU
    If subdivisions is 2, boundary will be processed by special logic and will not be suppressed by smoothing mask
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    pady_len = padded_img.shape[0]
    padx_len = padded_img.shape[1]

    # Crop subdivs into file
    subdivs_fname = os.path.join(temp_dir, 'subdivs.tmp')
    if os.path.isfile(subdivs_fname):
        os.remove(subdivs_fname)
    dtype = padded_img.dtype
    y_range = np.arange(0, pady_len-window_size+1, step)
    x_range = np.arange(0, padx_len-window_size+1, step)
    shape = (len(y_range), len(x_range), window_size, window_size, padded_img.shape[2])
    subdivs0 = np.memmap(subdivs_fname, dtype=dtype, mode='w+', shape=shape)
    overlap05_mask = _create_ovelap05_mask(shape[1], shape[0]) if subdivisions == 2 else None

    # Store subdivisions to file
    one_row_size_bytes = subdivs0.shape[1] * subdivs0.shape[2] * subdivs0.shape[3] * subdivs0.shape[4] * np.dtype(dtype).itemsize
    flush_step = max(1, int(1.e+9 / one_row_size_bytes))
    start_i = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            subdivs0[i - start_i][j] = padded_img[y_range[i]:y_range[i]+window_size, x_range[j]:x_range[j]+window_size, :]
        if (1 < i + 1 < shape[0]) and i % flush_step == 0:
            del subdivs0  # close file
            gc.collect()
            start_i = i + 1
            offset = start_i * one_row_size_bytes
            shape_new = list(shape)
            shape_new[0] -= start_i
            subdivs0 = np.memmap(subdivs_fname, dtype=dtype, mode='r+', shape=tuple(shape_new), offset=offset)
    del subdivs0  # close file
    gc.collect()

    a, b, c, d, e = shape
    logging.info('Smoothing prediction samples: {}'.format(a * b))

    if memmap_batch_size > 0:
        logging.info('Smoothing prediction memmap_batch_size: {}'.format(memmap_batch_size))
        subdivs_r_fname = os.path.join(temp_dir, 'subdivs_r.tmp')
        if os.path.isfile(subdivs_r_fname):
            os.remove(subdivs_r_fname)

        subdivs_r = None
        for i in range(0, a * b, memmap_batch_size):
            # Open required part of subdivisions
            memmap_batch_size_it = memmap_batch_size
            if i + memmap_batch_size > a * b:
                memmap_batch_size_it = a * b - i
            offset = i * c * d * e * np.dtype(dtype).itemsize
            subdivs = np.memmap(subdivs_fname, dtype=dtype, mode='r',
                                shape=(memmap_batch_size_it, c, d, e), offset=offset)

            # for aa in range(memmap_batch_size_it):
            #     sc_png = 'input_' + str(aa) + '.png'
            #     img_temp = (denormalize(subdivs[aa, ...]) * 255).astype(np.uint8)
            #     cv2.imwrite(os.path.join('./', sc_png), img_temp.astype(np.uint8))

            # Predict results of opened subdivisions
            pred_1 = pred_func(subdivs[0:memmap_batch_size_it])

            # for aa in range(memmap_batch_size_it):
            #     sc_png = 'output_' + str(aa) + '.png'
            #     pred_1[aa] = np.clip(pred_1[aa], 0, 1)
            #     img_temp = (pred_1[aa] * 255).astype(np.uint8)
            #     cv2.imwrite(os.path.join('./', sc_png), img_temp.astype(np.uint8))

            # Result will have the same type as source data
            pred_1 = pred_1.astype(dtype)

            # Release memory
            del subdivs  # close file
            gc.collect()

            # Store results
            if subdivs_r is None:
                subdivs_r = np.memmap(subdivs_r_fname, dtype=pred_1.dtype, mode='w+', shape=(a * b, *pred_1.shape[1:]))
            subdivs_r[i:i+memmap_batch_size_it] = pred_1[0:memmap_batch_size_it]

        subdivs = subdivs_r
    else:
        subdivs = np.memmap(subdivs_fname, dtype=dtype, mode='r', shape=(a * b, c, d, e))
        prediction = pred_func(subdivs)
        del subdivs  # close file
        subdivs = prediction

    WINDOW_SPLINE_2D = WINDOW_SPLINE_2D.astype(dtype)
    # Update results on the disk. Thats why here is loop instead of direct operation(which lost ref to memmap)
    for i in range(len(subdivs)):
        w = WINDOW_SPLINE_2D.copy()

        if overlap05_mask is not None:
            m = overlap05_mask[:, i*2: (i+1)*2]

            if np.max(m[0, :]) < 4:
                w = np.flip(w, axis=0)
                w = np.vstack([w[:w.shape[0] // 2, :],
                               np.repeat(w[w.shape[0] // 2, :][np.newaxis, :],
                                         w.shape[0] - w.shape[0] // 2, axis=0)])
                w = np.flip(w, axis=0)
            if np.max(m[1, :]) < 4:
                w = np.vstack([w[:w.shape[0] // 2, :],
                               np.repeat(w[w.shape[0] // 2, :][np.newaxis, :],
                                         w.shape[0] - w.shape[0] // 2, axis=0)])
            if np.max(m[:, 1]) < 4:
                w = np.hstack([w[:, :w.shape[1] // 2],
                               np.repeat(w[:, w.shape[1] // 2][:, np.newaxis],
                                         w.shape[1] - w.shape[1] // 2, axis=1)])
            if np.max(m[:, 0]) < 4:
                w = np.flip(w, axis=1)
                w = np.hstack([w[:, :w.shape[1] // 2],
                               np.repeat(w[:, w.shape[1] // 2][:, np.newaxis],
                                         w.shape[1] - w.shape[1] // 2, axis=1)])
                w = np.flip(w, axis=1)
        subdivs[i] = subdivs[i] * w

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)

    # Since it could be object of class np.memmap, do not forget to delete it after utilizing
    return subdivs


def _recreate_from_subdivs(y, subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    pady_len = padded_out_shape[0]
    padx_len = padded_out_shape[1]

    y = y.reshape(padded_out_shape)
    y *= 0

    a = 0
    for i in range(0, pady_len-window_size+1, step):
        b = 0
        for j in range(0, padx_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    y /= (subdivisions ** 2)
    return y


def pads_generator_1(im):
    yield im


def pads_generator_undo_1():
    x = yield; yield x


def pads_generator(im):
    yield im
    yield np.rot90(im, axes=(0, 1), k=1)
    yield np.rot90(im, axes=(0, 1), k=2)
    yield np.rot90(im, axes=(0, 1), k=3)

    yield im[:, ::-1]
    yield np.rot90(im[:, ::-1], axes=(0, 1), k=1)
    yield np.rot90(im[:, ::-1], axes=(0, 1), k=2)
    yield np.rot90(im[:, ::-1], axes=(0, 1), k=3)


def pads_generator_undo():
    x = yield; yield x
    x = yield; yield np.rot90(x, axes=(0, 1), k=3)
    x = yield; yield np.rot90(x, axes=(0, 1), k=2)
    x = yield; yield np.rot90(x, axes=(0, 1), k=1)
    x = yield; yield x[:, ::-1]
    x = yield; yield np.rot90(x, axes=(0, 1), k=3)[:, ::-1]
    x = yield; yield np.rot90(x, axes=(0, 1), k=2)[:, ::-1]
    x = yield; yield np.rot90(x, axes=(0, 1), k=1)[:, ::-1]


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func,
                                      memmap_batch_size=0, temp_dir='./', use_group_d4=True):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    If subdivisions is 2, boundary will be processed by special logic and will not be suppressed by smoothing mask
    """
    input_img_shape = input_img.shape
    pad = _pad_img(input_img, window_size, subdivisions)
    del input_img
    gc.collect()

    # Note that the implementation could be more memory-efficient by merging
    # the behavior of `_windowed_subdivs` and `_recreate_from_subdivs` into
    # one loop doing in-place assignments to the new image matrix, rather than
    # using a temporary 5D array.

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero dommain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    pads_num = 8 if use_group_d4 else 1
    gen = pads_generator if use_group_d4 else pads_generator_1
    undo_gen = pads_generator_undo() if use_group_d4 else pads_generator_undo_1()
    padded_results = None
    y_fname = os.path.join(temp_dir, 'y.tmp')
    if os.path.isfile(y_fname):
        os.remove(y_fname)
    y = np.memmap(y_fname, dtype=pad.dtype, mode='w+', shape=tuple(list(pad.shape[:-1])+[nb_classes]))
    for pad in tqdm(gen(pad), total=pads_num):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func, memmap_batch_size, temp_dir)
        one_padded_result = _recreate_from_subdivs(
            y, sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        # Since this object could be np.memmap, close file by deleting them
        del sd

        gc.collect()

        next(undo_gen)
        one_padded_result_reconstructed = undo_gen.send(one_padded_result)
        if padded_results is None:
            padded_results_fname = os.path.join(temp_dir, 'padded_results.tmp')
            if os.path.isfile(padded_results_fname):
                os.remove(padded_results_fname)

            padded_results = np.memmap(padded_results_fname,
                                       dtype=one_padded_result_reconstructed.dtype,
                                       mode='w+',
                                       shape=one_padded_result_reconstructed.shape)
            padded_results *= 0
        padded_results += one_padded_result_reconstructed

    del y  # close file

    # Merge after rotations:
    padded_results /= pads_num

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img_shape[0], :input_img_shape[1], :]

    if isinstance(prd, np.memmap):
        prd_ram = np.array(prd)
        del prd  # close file
        del padded_results
        prd = prd_ram

    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prd


def cheap_tiling_prediction(img, window_size, nb_classes, pred_func):
    """
    Does predictions on an image without tiling.
    """
    original_shape = img.shape
    full_border = img.shape[0] + (window_size - (img.shape[0] % window_size))
    prd = np.zeros((full_border, full_border, nb_classes))
    tmp = np.zeros((full_border, full_border, original_shape[-1]))
    tmp[:original_shape[0], :original_shape[1], :] = img
    img = tmp
    logging.info('{}, {}, {}'.format(img.shape, tmp.shape, prd.shape))
    for i in tqdm(range(0, prd.shape[0], window_size)):
        for j in range(0, prd.shape[0], window_size):
            im = img[i:i+window_size, j:j+window_size]
            prd[i:i+window_size, j:j+window_size] = pred_func([im])
    prd = prd[:original_shape[0], :original_shape[1]]
    if PLOT_PROGRESS:
        plt.imshow(prd)
        plt.title("Cheaply Merged Patches")
        plt.show()
    return prd


def get_dummy_img(xy_size=128, nb_channels=3):
    """
    Create a random image with different luminosity in the corners.
    Returns an array of shape (xy_size, xy_size, nb_channels).
    """
    x = np.random.random((xy_size, xy_size, nb_channels))
    x = x + np.ones((xy_size, xy_size, 1))
    lin = np.expand_dims(
        np.expand_dims(
            np.linspace(0, 1, xy_size),
            nb_channels),
        nb_channels)
    x = x * lin
    x = x * lin.transpose(1, 0, 2)
    x = x + x[::-1, ::-1, :]
    x = x - np.min(x)
    x = x / np.max(x) / 2
    gc.collect()
    if PLOT_PROGRESS:
        plt.imshow(x)
        plt.title("Random image for a test")
        plt.show()
    return x


def round_predictions(prd, nb_channels_out, thresholds):
    """
    From a threshold list `thresholds` containing one threshold per output
    channel for comparison, the predictions are converted to a binary mask.
    """
    assert (nb_channels_out == len(thresholds))
    prd = np.array(prd)
    for i in range(nb_channels_out):
        # Per-pixel and per-channel comparison on a threshold to
        # binarize prediction masks:
        prd[:, :, i] = prd[:, :, i] > thresholds[i]
    return prd


if __name__ == '__main__':
    ###
    # Image:
    ###

    img_resolution = 600
    # 3 such as RGB, but there could be more in other cases:
    nb_channels_in = 3

    # Get an image
    input_img = get_dummy_img(img_resolution, nb_channels_in)
    # Normally, preprocess the image for input in the neural net:
    # input_img = to_neural_input(input_img)

    ###
    # Neural Net predictions params:
    ###

    # Number of output channels. E.g. a U-Net may output 10 classes, per pixel:
    nb_channels_out = 3
    # U-Net's receptive field border size, it does not absolutely
    # need to be a divisor of "img_resolution":
    window_size = 128

    # This here would be the neural network's predict function, to used below:
    def predict_for_patches(small_img_patches):
        """
        Apply prediction on images arranged in a 4D array as a batch.
        Here, we use a random color filter for each patch so as to see how it
        will blend.
        Note that the np array shape of "small_img_patches" is:
            (nb_images, x, y, nb_channels_in)
        The returned arra should be of the same shape, except for the last
        dimension which will go from nb_channels_in to nb_channels_out
        """
        small_img_patches = np.array(small_img_patches)
        rand_channel_color = np.random.random(size=(
            small_img_patches.shape[0],
            1,
            1,
            small_img_patches.shape[-1])
        )
        return small_img_patches * rand_channel_color * 2

    ###
    # Doing cheap tiled prediction:
    ###

    # Predictions, blending the patches:
    cheaply_predicted_img = cheap_tiling_prediction(
        input_img, window_size, nb_channels_out, pred_func=predict_for_patches
    )

    ###
    # Doing smooth tiled prediction:
    ###

    # The amount of overlap (extra tiling) between windows. A power of 2, and is >= 2:
    subdivisions = 2

    # Predictions, blending the patches:
    smoothly_predicted_img = predict_img_with_smooth_windowing(
        input_img, window_size, subdivisions,
        nb_classes=nb_channels_out, pred_func=predict_for_patches
    )

    ###
    # Demonstrating that the reconstruction is correct:
    ###

    # No more plots from now on
    PLOT_PROGRESS = False

    # useful stats to get a feel on how high will be the error relatively
    print(
        "Image's min and max pixels' color values:",
        np.min(input_img),
        np.max(input_img))

    # First, defining a prediction function that just returns the patch without
    # any modification:
    def predict_same(small_img_patches):
        """
        Apply NO prediction on images arranged in a 4D array as a batch.
        This implies that nb_channels_in == nb_channels_out: dimensions
        and contained values are unchanged.
        """
        return small_img_patches

    same_image_reconstructed = predict_img_with_smooth_windowing(
        input_img, window_size, subdivisions,
        nb_classes=nb_channels_out, pred_func=predict_same
    )

    diff = np.mean(np.abs(same_image_reconstructed - input_img))
    print(
        "Mean absolute reconstruction difference on pixels' color values:",
        diff)
    print(
        "Relative absolute mean error on pixels' color values:",
        100*diff/(np.max(input_img)) - np.min(input_img),
        "%")
    print(
        "A low error (e.g.: 0.28 %) confirms that the image is still "
        "the same before and after reconstruction if no changes are "
        "made by the passed prediction function.")