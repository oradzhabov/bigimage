import os
import sys
import numpy as np
import cv2
from keras import backend as K
import gc
sys.path.append(sys.path[0] + "/..")
from kutils import PrepareData
from kutils.read_sample import read_sample
from kmodel.smooth_tiled_predictions import predict_img_with_smooth_windowing
from kmodel.data import read_image
from kutils.utilites import denormalize


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def predict_contours(cfg, src_proj_dir, skip_prediction=False, memmap_batch_size=0, predict_img_with_group_d4=True):
    """
    :param cfg:
    :param src_proj_dir:
    :param skip_prediction: If True following flag it will avoid long prediction and will try to read already
    created result. Useful for debugging.
    :param memmap_batch_size: If > 0, stitching will process with np.memmap. Value 6 is good for 4 GB GPU as for
    efficientb5(512_wh) as for efficcientb3(1024_wh). So if GPU will be 16 GB GPU, could be increased to 6**2 = 36
    :param predict_img_with_group_d4: If False, it will take 8 times faster and 2-times less CPU RAM, but will not use
    D4-group augmentation for prediction smoothing.
    :return:
    """
    solver = cfg.solver(cfg)
    provider = cfg.provider_single

    dest_img_fname = os.path.join(src_proj_dir,
                                  'tmp_mppx{:.2f}.png'.format(cfg.mppx))
    dest_himg_fname = os.path.join(src_proj_dir,
                                   'htmp_mppx{:.2f}.png'.format(cfg.mppx)) if cfg.use_heightmap else None

    is_success = PrepareData.build_from_project(src_proj_dir, cfg.mppx, dest_img_fname, dest_himg_fname)
    if not is_success:
        print('ERROR: Cannot prepare data')
        return -1, dict({})

    dataset = provider(read_sample,
                       dest_img_fname,
                       dest_himg_fname,
                       cfg,
                       prep_getter=solver.get_prep_getter())

    model = None

    image, _ = dataset[0]
    # map input space to float16
    image = image.astype(np.float16)
    gc.collect()

    store_predicted_result_to_ram = False

    pr_mask_list = list()

    # Save original image params
    src_image_shape = image.shape
    src_image_dtype = image.dtype
    for sc_factor in cfg.pred_scale_factors:
        predict_sc_png = 'probability_' + solver.signature() + '_' + str(sc_factor) + '.png'
        fpath = os.path.join(src_proj_dir, predict_sc_png)

        pr_item = dict({'scale': sc_factor})

        pr_result_descriptor = fpath

        if skip_prediction and os.path.isfile(fpath):
            # Use pre-saved results if it allowed and file exist

            # If result obtained from pre-saved file, its type will be the same as source file
            pr_item['img_dtype'] = src_image_dtype

            print('Prediction skipped. Trying to read already prepared result from {}'.format(predict_sc_png))

            if store_predicted_result_to_ram:
                # Read and map result to the same type as source data. It completely simulate prediction results
                pr_mask = read_image(fpath).astype(src_image_dtype) / 255.0

                # Just to simulate shape of real generated data
                if len(pr_mask.shape) == 2:
                    pr_mask = pr_mask[..., np.newaxis]

                pr_result_descriptor = pr_mask
        else:
            # Prepare model if it necessary
            if model is None:
                print('Build model...')
                model, _, _ = solver.build(compile_model=False)
                if model is None:
                    print('ERROR: Cannot create model')
                    return -1, dict({})

            # Scale image if it necessary
            if sc_factor != 1.0:
                # For downscale the best interpolation INTER_AREA
                image = cv2.resize(image.astype(np.float32), (0, 0),
                                   fx=sc_factor, fy=sc_factor, interpolation=cv2.INTER_AREA).astype(src_image_dtype)

            # Store result with unique(scaled) name
            # sc_png = 'image_scaled_' + solver.signature() + '_' + str(sc_factor) + '.png'
            # img_temp = (denormalize(image[..., :3]) * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(src_proj_dir, sc_png), img_temp.astype(np.uint8))

            # IMPORTANT:
            # * Do not use size bigger than actual image size because blending(with generated border)
            # will suppress actual prediction result.
            window_size = int(find_nearest([64, 128, 256, 512, 1024], min(image.shape[0], image.shape[1])))
            print('Window size in smoothing predicting: {}'.format(window_size))

            pr_mask = predict_img_with_smooth_windowing(
                image,
                window_size=window_size,
                subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                nb_classes=cfg.cls_nb,
                pred_func=(
                    lambda img_batch_subdiv: model.predict(img_batch_subdiv)
                ),
                memmap_batch_size=memmap_batch_size,
                temp_dir=src_proj_dir,
                use_group_d4=predict_img_with_group_d4
            )
            gc.collect()

            # Upscale result if it necessary
            if pr_mask.shape[0] != src_image_shape[0] or pr_mask.shape[1] != src_image_shape[1]:
                pr_mask_dtype = pr_mask.dtype
                pr_mask_shape = pr_mask.shape
                # For upscale the best interpolation is CUBIC
                pr_mask = cv2.resize(pr_mask.astype(np.float32),
                                     (src_image_shape[1], src_image_shape[0]),
                                     interpolation=cv2.INTER_CUBIC).astype(pr_mask_dtype)
                # Adjust shape after resizing
                if len(pr_mask.shape) > len(pr_mask_shape):
                    pr_mask = pr_mask.squeeze()
                if len(pr_mask.shape) < len(pr_mask_shape):
                    pr_mask = pr_mask[..., np.newaxis]

            pr_mask = solver.post_predict(pr_mask)

            pr_item['img_dtype'] = pr_mask.dtype

            # Store result with unique(per scale) name
            print('Store predicted result to file {}'.format(predict_sc_png))
            cv2.imwrite(fpath, (pr_mask * 255).astype(np.uint8))

            if store_predicted_result_to_ram:
                pr_result_descriptor = pr_mask
            else:
                # Release memory from predicted result if it will not use
                del pr_mask
                gc.collect()

        pr_item['img'] = pr_result_descriptor
        pr_mask_list.append(pr_item)

    # =================================================================================================================
    # Release memory
    K.clear_session()
    del image
    gc.collect()

    pr_cntrs_list_px = solver.get_contours(pr_mask_list)

    return 0, dict({'contours_px': pr_cntrs_list_px, 'dataset': dataset, 'solver': solver})
