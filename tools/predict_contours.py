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
    if not skip_prediction:
        model, _, _ = solver.build(compile_model=False)
        if model is None:
            print('ERROR: Cannot create model')
            return -1, dict({})

    image, _ = dataset[0]
    # map input space to float16
    image = image.astype(np.float16)
    gc.collect()

    predict_png = 'probability_' + solver.signature() + '.png'
    if model is not None:

        # IMPORTANT:
        # * Do not use size bigger than actual image size because blending(with generated border) will suppress actual
        # prediction result.
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
        K.clear_session()
        gc.collect()

        pr_mask = solver.post_predict(pr_mask)

        cv2.imwrite(os.path.join(src_proj_dir, predict_png), (pr_mask * 255).astype(np.uint8))
    else:
        fpath = os.path.join(src_proj_dir, predict_png)
        print('Prediction skipped. Trying to read already predicted result from {}'.format(fpath))

        # Read and map result to the same type as source data. It completely simulate prediction results
        pr_mask = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype(image.dtype) / 255.0

        # Just to simulate shape of real generated data
        if len(pr_mask.shape) == 2:
            pr_mask = pr_mask[..., np.newaxis]

    # Release memory
    del image
    gc.collect()

    pr_cntrs_list_px = solver.get_contours(pr_mask)

    return 0, dict({'contours_px': pr_cntrs_list_px, 'dataset': dataset, 'solver': solver})
