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


def predict_contours(cfg, src_proj_dir, skip_prediction=False, use_batch_1=True):
    """
    :param cfg:
    :param src_proj_dir:
    :param skip_prediction: If True following flag it will avoid long prediction and will try to read already
    created result. Useful for debugging.
    :param use_batch_1: If True, stitching will process patch by patch and guaranty that GPU RAM will be enough for
    process any number of patches(any size of image). From other side it increases the processing time.
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
        model, weights_path, _ = solver.build(compile_model=False)
        if model is None:
            print('ERROR: Cannot create model')
            return -1, dict({})

    image, _ = dataset[0]
    predict_png = 'probability_' + os.path.splitext(os.path.basename(solver.weights_path))[0] + '.png'
    # IMPORTANT:
    # * Do not use size bigger than actual image size because blending(with generated border) will suppress actual
    # prediction result.
    window_size = int(find_nearest([64, 128, 256, 512, 1024], min(image.shape[0], image.shape[1])))
    print('Window size in smoothing predicting: {}'.format(window_size))
    if model is not None:
        pr_mask = predict_img_with_smooth_windowing(
            image,
            window_size=window_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=cfg.cls_nb,
            pred_func=(
                lambda img_batch_subdiv: model.predict(img_batch_subdiv)
            ),
            use_batch_1=use_batch_1
        )
        K.clear_session()
        gc.collect()

        pr_mask = solver.post_predict(pr_mask)

        cv2.imwrite(os.path.join(src_proj_dir, predict_png), (pr_mask * 255).astype(np.uint8))
    else:
        fpath = os.path.join(src_proj_dir, predict_png)
        print('Prediction skipped. Trying to read already predicted result from {}'.format(fpath))
        pr_mask = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        # Just to simulate shape of real generated data
        if len(pr_mask.shape) == 2:
            pr_mask = pr_mask[..., np.newaxis]

    pr_cntrs_list_px = solver.get_contours(pr_mask)

    return 0, dict({'contours_px': pr_cntrs_list_px, 'dataset': dataset, 'solver': solver})
