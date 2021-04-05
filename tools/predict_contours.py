import pickle
import logging
import os
import numpy as np
import cv2
import gc
from ..solvers.ISolver import ISolver
from ..kutils import PrepareData
from ..kutils.read_sample import read_sample
from ..kutils.PrepareData import get_raster_info
from ..kmodel.smooth_tiled_predictions import predict_img_with_smooth_windowing
from ..kmodel.data import read_image
from ..kmodel import kutils
from .. import get_submodules_from_kwargs
# from ..kutils.utilites import denormalize


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def store_predicted_result(pr_mask, fpath):
    if len(pr_mask.shape) > 2 and pr_mask.shape[2] == 2:
        # Operate with 2-channels output
        # Add extra channel and save
        cv2.imwrite(fpath, ((np.pad(pr_mask, ((0, 0),) * 2 + ((0, 1),),
                                    'constant', constant_values=0)) * 255).astype(np.uint8))
    else:
        cv2.imwrite(fpath, (pr_mask * 255).astype(np.uint8))


def predict_on_bbox(cfg, solver, dataset, src_proj_dir,
                    skip_prediction=False, memmap_batch_size=0, predict_img_with_group_d4=True,
                    bbox=((0, 0), (None, None)), debug=False):

    bbox_str = 'bb{}-{}-{}-{}'.format(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])

    working_dtype = np.float16

    # If don't store results to RAM, predicted results will be reused from Disk. So it will save the RAM during
    # postprocessing if read each result from Disk by demand.
    store_predicted_result_to_ram = False

    pr_mask_list = list()

    for sc_factor in cfg.pred_scale_factors:
        predict_sc_png = 'probability_' + solver.signature() + '_' + bbox_str + '_' + str(sc_factor) + '.png'
        predict_raw_png = 'probability_raw_' + solver.signature() + '_' + bbox_str + '_' + str(sc_factor) + '.png'
        fpath = os.path.join(src_proj_dir, predict_sc_png)
        fpath_raw = os.path.join(src_proj_dir, predict_raw_png)

        # Define predicted result structure
        pr_item = dict({'scale': sc_factor})

        # Default result descriptor - path to result file. It could be overwrite by np.ndarray later
        # if condition satisfied
        pr_result_descriptor = fpath

        # Use pre-saved results if it allowed and file exist
        if skip_prediction and os.path.isfile(fpath):

            logging.info('Prediction skipped. Trying to read already prepared result from {}'.format(predict_sc_png))

            # If result obtained from pre-saved file, its type will be the same as source file
            pr_item['img_dtype'] = working_dtype

            if store_predicted_result_to_ram:
                # Read and map result to the same type as source data. It completely simulate prediction results
                pr_mask = read_image(fpath).astype(working_dtype) / 255.0

                # Just to simulate shape of real generated data
                if len(pr_mask.shape) == 2:
                    pr_mask = pr_mask[..., np.newaxis]

                # Sometime(e.g. 2-channels output) data stored with bigger channels num. Trunc used channels.
                if len(pr_mask.shape) > 2:
                    pr_mask = pr_mask[..., :cfg.cls_nb]

                pr_result_descriptor = pr_mask
        else:
            image, _ = dataset[0]
            # map input space to float16
            image = image.astype(working_dtype)
            gc.collect()
            # Save original image params
            src_image_shape = image.shape

            # Prepare the model if it necessary
            if solver.model is None:
                logging.info('Build the model...')
                model, _, _ = solver.build(compile_model=False)
                if model is None:
                    logging.error('Cannot create the model')
                    return -1, dict({})

            # Scale image if it necessary
            if sc_factor != 1.0:
                # For downscale the best interpolation INTER_AREA
                image = cv2.resize(image.astype(np.float32), (0, 0),
                                   fx=sc_factor, fy=sc_factor, interpolation=cv2.INTER_AREA).astype(working_dtype)
                gc.collect()

            # Store result with unique(scaled) name
            # sc_png = 'image_scaled_' + solver.signature() + '_' + bbox_str + '_' + str(sc_factor) + '.png'
            # img_temp = (denormalize(image[..., :3]) * 255).astype(np.uint8)
            # cv2.imwrite(os.path.join(src_proj_dir, sc_png), img_temp.astype(np.uint8))

            # IMPORTANT:
            # * Do not use size bigger than actual image size because blending(with generated reflected border)
            # could suppress actual prediction result. Accuracy will be degraded.
            # * In theory, as bigger window size as bigger data content will be processed by one iteration, and expected
            # accuracy will be better. todo: maybe max value should be controled? What if GPU supports window size 2048?
            # * Bigger window size speed up the processing time.
            window_size = int(find_nearest(np.power(2, np.arange(6, 1 + int(cfg.window_size_2pow_max))),
                                           min(image.shape[0], image.shape[1])))
            logging.info('Window size in smoothing predicting: {}'.format(window_size))

            pr_mask = predict_img_with_smooth_windowing(
                image,
                window_size=window_size,
                # Minimal amount of overlap for windowing.
                # Must be an even number.
                # Algorithm supports good boundary for half overlap. So preferable value is 2.
                subdivisions=2,
                nb_classes=cfg.cls_nb,
                pred_func=(
                    lambda img_batch_subdiv: solver.model.predict(img_batch_subdiv)
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
                # Adjust shape after resizing because cv2.resize() could modify the shape length
                if len(pr_mask.shape) > len(pr_mask_shape):
                    pr_mask = pr_mask.squeeze()
                if len(pr_mask.shape) < len(pr_mask_shape):
                    pr_mask = pr_mask[..., np.newaxis]

            logging.info('Store predicted results...')

            # Store raw result with unique(per scale) name before post-predict
            if debug:
                store_predicted_result(pr_mask, fpath_raw)

            pr_mask = solver.post_predict(pr_mask)
            pr_item['img_dtype'] = pr_mask.dtype

            # Store post-processed result with unique(per scale) name
            store_predicted_result(pr_mask, fpath)

            if store_predicted_result_to_ram:
                pr_result_descriptor = pr_mask
            else:
                # Release memory from predicted result if it will not use
                del pr_mask
                gc.collect()

            # Release memory
            del image

        # Release memory between scale-iterations
        gc.collect()

        pr_item['img'] = pr_result_descriptor
        pr_mask_list.append(pr_item)

    return 0, pr_mask_list


def predict_contours(**kwarguments):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    def predict_contours_(cfg, src_proj_dir, skip_prediction=False, memmap_batch_size=0, predict_img_with_group_d4=True,
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

        dest_img_fname = os.path.join(src_proj_dir,
                                      'tmp_mppx{:.2f}.png'.format(cfg.mppx))
        dest_himg_fname = os.path.join(src_proj_dir,
                                       'htmp_mppx{:.2f}.png'.format(cfg.mppx)) if cfg.use_heightmap else None
        result_contours_fpath = os.path.join(src_proj_dir, 'result_cntrs_' + solver.signature() + '.pkl')

        pr_cntrs_list_px = None
        if os.path.isfile(result_contours_fpath) and skip_prediction:
            with open(result_contours_fpath, 'rb') as inp:
                pr_cntrs_list_px = pickle.load(inp)
                logging.info('Prediction skipped. Result has been restored from file {}'.format(result_contours_fpath))

        if pr_cntrs_list_px is None:
            is_success = PrepareData.build_from_project(src_proj_dir, cfg.mppx, dest_img_fname, dest_himg_fname)
            if not is_success:
                logging.error('Cannot prepare data')
                return -1, dict({})

            # Obtain source image shape
            src_img_shape, _ = get_raster_info(dest_img_fname)

            # Collect bounding boxes
            bbox_list = list()
            offset_size_px = (max(crop_size_px[0] - overlap_px, crop_size_px[0] // 2),
                              max(crop_size_px[1] - overlap_px, crop_size_px[1] // 2))
            w0, w1, h0, h1 = kutils.get_tiled_bbox(src_img_shape, crop_size_px, offset_size_px)
            for i in range(len(w0)):
                cr_x, extr_x = (w0[i], 0) if w0[i] >= 0 else (0, -w0[i])
                cr_x2, extr_x2 = (w1[i], 0) if w1[i] < src_img_shape[1] else (src_img_shape[1],
                                                                              w1[i] - src_img_shape[1])

                cr_y, extr_y = (h0[i], 0) if h0[i] >= 0 else (0, -h0[i])
                cr_y2, extr_y2 = (h1[i], 0) if h1[i] < src_img_shape[0] else (src_img_shape[0],
                                                                              h1[i] - src_img_shape[0])

                bbox = ((cr_x, cr_y), (cr_x2-cr_x, cr_y2-cr_y))
                bbox_list.append(bbox)
            logging.info('Source image cropped to {} patches with cropping size {}'.format(len(bbox_list),
                                                                                           crop_size_px))

            # Process each bound box
            bbox_predictions = list()
            for bbox_ind, bbox in enumerate(bbox_list):
                logging.info('Patch #{} (from {}) in processing...'.format(bbox_ind+1, len(bbox_list)))
                dataset = provider(read_sample,
                                   dest_img_fname,
                                   dest_himg_fname,
                                   bbox,
                                   cfg,
                                   prep_getter=solver.get_prep_getter())

                err, pr_mask_list = predict_on_bbox(cfg, solver, dataset, src_proj_dir,
                                                    skip_prediction=skip_prediction,
                                                    memmap_batch_size=memmap_batch_size,
                                                    predict_img_with_group_d4=predict_img_with_group_d4,
                                                    bbox=bbox,
                                                    debug=debug)
                bbox_predictions.append(dict({'bbox': bbox, 'pr_mask_list': pr_mask_list}))

            # Release memory
            if solver.model is not None:
                del solver.model
                solver.model = None
            _backend.clear_session()
            gc.collect()

            # Merge data
            merged_pr_mask_list = list()
            used_scales = len(bbox_predictions[0]['pr_mask_list'])
            tmp_img = cv2.imread(bbox_predictions[0]['pr_mask_list'][0]['img'],
                                 cv2.IMREAD_UNCHANGED)  # todo: is there is no other way to get channels nb?
            used_dtype = np.uint8  # bbox_predictions[0]['pr_mask_list'][0]['img_dtype']
            used_shape = (src_img_shape[0], src_img_shape[1], tmp_img.shape[2] if len(tmp_img.shape) > 2 else 1)
            del tmp_img
            gc.collect()
            one_row_size_bytes = used_shape[1] * used_shape[2] * np.dtype(used_dtype).itemsize
            dstep = 1.0 / (overlap_px - 1)
            mask_template = np.arange(0.0, 1.0 + dstep / 2, dstep)[np.newaxis, :]
            for scale_ind in range(used_scales):
                merged_pr_mask_list_item = dict()
                bbox_predictions_fname_woext = '{}-sc{}'.format(merging_fname_head, scale_ind)
                merged_pr_mask_list_item['img'] = os.path.join(src_proj_dir,
                                                               '{}.png'.format(bbox_predictions_fname_woext))
                merged_pr_mask_list_item['img_dtype'] = bbox_predictions[0]['pr_mask_list'][scale_ind]['img_dtype']
                merged_pr_mask_list_item['scale'] = bbox_predictions[0]['pr_mask_list'][scale_ind]['scale']
                merged_pr_mask_list_item['shape'] = used_shape
                merged_pr_mask_list_item['skip_prediction'] = skip_prediction
                if os.path.isfile(merged_pr_mask_list_item['img']) and skip_prediction:
                    merged_pr_mask_list.append(merged_pr_mask_list_item)
                    logging.info('Merging skipped. Result has been restored from file {}'.format(
                        merged_pr_mask_list_item['img']))
                    continue
                bbox_predictions_fname = os.path.join(src_proj_dir, '{}.tmp'.format(bbox_predictions_fname_woext))
                bbox_predictions_result = np.memmap(bbox_predictions_fname, dtype=used_dtype, mode='w+',
                                                    shape=used_shape)
                bbox_predictions_result.fill(0)
                del bbox_predictions_result  # close file
                for bbox_predictions_item in bbox_predictions:
                    xy, wh = bbox_predictions_item['bbox']
                    pr_mask_list = bbox_predictions_item['pr_mask_list']
                    pr_mask = pr_mask_list[scale_ind]
                    #
                    # merged_pr_mask_list_item['img_dtype'] = pr_mask['img_dtype']
                    # merged_pr_mask_list_item['scale'] = pr_mask['scale']
                    #
                    patch = cv2.imread(pr_mask['img'], cv2.IMREAD_UNCHANGED)
                    if len(patch.shape) == 2:
                        patch = patch[:, :, np.newaxis]

                    shape_new = (wh[1], used_shape[1], used_shape[2])
                    offset = xy[1] * one_row_size_bytes
                    bbox_predictions_result = np.memmap(bbox_predictions_fname, dtype=used_dtype,
                                                        mode='r+', shape=shape_new, offset=offset)
                    if xy[0] > 0:  # blend left side
                        mask = np.repeat(mask_template, patch.shape[0], 0)[:, :, np.newaxis]
                        patch[:, 0:overlap_px] = np.multiply(patch[:, 0:overlap_px], mask).astype(patch.dtype)
                    if xy[1] > 0:  # blend top side
                        mask = np.repeat(mask_template.T, patch.shape[1], 1)[:, :, np.newaxis]
                        patch[0:overlap_px, :] = np.multiply(patch[0:overlap_px, :], mask).astype(patch.dtype)
                    if xy[0] + wh[0] < used_shape[1]:  # blend right side
                        mask = np.repeat(mask_template, patch.shape[0], 0)[:, :, np.newaxis]
                        patch = np.flip(patch, 1)
                        patch[:, 0:overlap_px] = np.multiply(patch[:, 0:overlap_px], mask).astype(patch.dtype)
                        patch = np.flip(patch, 1)
                    if xy[1] + wh[1] < used_shape[0]:  # blend bottom side
                        mask = np.repeat(mask_template.T, patch.shape[1], 1)[:, :, np.newaxis]
                        patch = np.flip(patch, 0)
                        patch[0:overlap_px, :] = np.multiply(patch[0:overlap_px, :], mask).astype(patch.dtype)
                        patch = np.flip(patch, 0)

                    bbox_predictions_result[0:patch.shape[0], xy[0]:xy[0] + patch.shape[1]] += patch

                    del bbox_predictions_result

                # Save result merged image
                result = np.memmap(bbox_predictions_fname, dtype=used_dtype, mode='r', shape=used_shape)
                cv2.imwrite(merged_pr_mask_list_item['img'], result)
                del result  # close memmap
                logging.info('Merged results stored into {}'.format(merged_pr_mask_list_item['img']))
                #
                merged_pr_mask_list.append(merged_pr_mask_list_item)

            pr_cntrs_list_px = solver.get_contours(merged_pr_mask_list)
            gc.collect()

            # Store results for debug purposes
            with open(result_contours_fpath, 'wb') as output:
                pickle.dump(pr_cntrs_list_px, output, pickle.HIGHEST_PROTOCOL)

        # Prepare final dataset without pre-processing to minimize RAM using during image reading
        dataset = provider(read_sample,
                           dest_img_fname,
                           dest_himg_fname,
                           ((0, 0), (None, None)),  # Whole source image
                           cfg,
                           prep_getter=None)

        return 0, dict({'contours_px': pr_cntrs_list_px, 'dataset': dataset, 'solver': solver})

    return predict_contours_
