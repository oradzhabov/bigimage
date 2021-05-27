import pickle
import os
import gc
import logging
from ..solvers.ISolver import ISolver
from ..bin_keras import predict_field
from .read_sample import read_sample


def predict_contours(cfg, src_proj_dir, skip_prediction=False, memmap_batch_size=0, predict_img_with_group_d4=True,
                     crop_size_px=(10000, 10000), overlap_px=0, merging_fname_head='merged_predictions',
                     debug=False):
    """
    :param cfg:
    :param src_proj_dir:
    :param skip_prediction: If True following flag it will avoid long prediction and will try to read already
    created result. Useful for debugging.
    :param memmap_batch_size: If > 0, stitching will process with np.memmap. Value 6 is good for 4 GB GPU as for
    efficientb5(512_wh) as for efficcientb3(1024_wh). So if GPU will be 16 GB GPU, could be increased to 6**2 = 36
    :param predict_img_with_group_d4: If False, it will take 8 times faster and 2-times less CPU RAM, but will not
    use D4-group augmentation for prediction smoothing.
    :param crop_size_px:
    :param overlap_px: > 0
    :param merging_fname_head: head of filename used for storing temporary files
    :param debug: Default: False
    :return:
    """

    # Check is solver has not been instanced before
    if not isinstance(cfg.solver, ISolver):
        cfg.solver = cfg.solver(cfg)
    solver = cfg.solver
    provider = cfg.provider_single

    result_contours_fpath = os.path.join(src_proj_dir, 'result_cntrs_' + solver.signature() + '.pkl')

    dest_img_fname = os.path.join(src_proj_dir,
                                  'tmp_mppx{:.2f}.png'.format(cfg.mppx))
    dest_himg_fname = os.path.join(src_proj_dir,
                                   'htmp_mppx{:.2f}.png'.format(cfg.mppx)) if cfg.use_heightmap else None

    # Prepare final dataset without pre-processing to minimize RAM using during image reading
    dataset = provider(read_sample,
                       dest_img_fname,
                       dest_himg_fname,
                       ((0, 0), (None, None)),  # Whole source image
                       cfg,
                       prep_getter=None)

    pr_cntrs_list_px = None
    if os.path.isfile(result_contours_fpath) and skip_prediction:
        with open(result_contours_fpath, 'rb') as inp:
            pr_cntrs_list_px = pickle.load(inp)
            logging.info('Prediction skipped. Result has been restored from file {}'.format(result_contours_fpath))

    if pr_cntrs_list_px is None:
        merged_pr_mask_list = predict_field(dataset, src_proj_dir, skip_prediction, memmap_batch_size,
                                            predict_img_with_group_d4, crop_size_px, overlap_px,
                                            merging_fname_head, debug)

        if merged_pr_mask_list is None:
            return -1, dict()

        logging.info('Getting contours...')
        pr_cntrs_list_px = solver.get_contours(merged_pr_mask_list)
        gc.collect()

        # Store results for debug purposes
        with open(result_contours_fpath, 'wb') as output:
            pickle.dump(pr_cntrs_list_px, output, pickle.HIGHEST_PROTOCOL)

    return 0, dict({'contours_px': pr_cntrs_list_px, 'dataset': dataset, 'solver': solver})
