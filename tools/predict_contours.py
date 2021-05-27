import logging
import os
import numpy as np
import cv2
import gc
from ..kutils import PrepareData
from ..kutils.read_sample import read_sample
from ..kutils.PrepareData import get_raster_info
from ..kmodel.smooth_tiled_predictions import predict_img_with_smooth_windowing
from ..kmodel.data import read_image
from ..kmodel import kutils
from .. import get_submodules_from_kwargs
from ..data_provider import IDataProvider
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


def predict_on_bbox(cfg, solver, dataset, src_proj_dir, pred_scale_factors,
                    skip_prediction=False, memmap_batch_size=0, predict_img_with_group_d4=True,
                    bbox=((0, 0), (None, None)), working_dtype=np.float16, debug=False):

    bbox_str = 'bb{}-{}-{}-{}'.format(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])

    # If don't store results to RAM, predicted results will be reused from Disk. So it will save the RAM during
    # postprocessing if read each result from Disk by demand.
    store_predicted_result_to_ram = False

    pr_mask_list = list()

    for sc_factor in pred_scale_factors:
        predict_sc_png = 'probability_' + solver.signature() + '_' + bbox_str + '_' + str(sc_factor) + '.png'
        predict_raw_png = 'probability_raw_' + solver.signature() + '_' + bbox_str + '_' + str(sc_factor) + '.png'
        fpath = os.path.join(src_proj_dir, predict_sc_png)
        fpath_raw = os.path.join(src_proj_dir, predict_raw_png)

        # Define predicted result structure
        pr_item = dict({'scale': sc_factor})
        # If result obtained from pre-saved file, its type will be the same as source file
        pr_item['img_dtype'] = working_dtype

        # Default result descriptor - path to result file. It could be overwrite by np.ndarray later
        # if condition satisfied
        pr_result_descriptor = fpath

        # Use pre-saved results if it allowed and file exist
        if skip_prediction and os.path.isfile(fpath):

            logging.info('Prediction skipped. Trying to read already prepared result from {}'.format(predict_sc_png))

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
            if dataset is not None:
                src_image_shape, image = dataset.get_scaled_image(0, sc_factor)
                # map input space to float16
                image = image.astype(working_dtype)
                gc.collect()

                # Prepare the model if it necessary
                if solver.model is None:
                    logging.info('Build the model...')
                    model, _, _ = solver.build(compile_model=False)
                    if model is None:
                        logging.error('Cannot create the model')
                        return -1, dict({})

                # Store result with unique(scaled) name
                # sc_png = 'image_scaled_' + solver.signature() + '_' + bbox_str + '_' + str(sc_factor) + '.png'
                # img_temp = (denormalize(image[..., :3]) * 255).astype(np.uint8)
                # cv2.imwrite(os.path.join(src_proj_dir, sc_png), img_temp.astype(np.uint8))

                # IMPORTANT:
                # * Do not use size bigger than actual image size because blending(with generated reflected border)
                # could suppress actual prediction result. Accuracy will be degraded.
                # * In theory, as bigger window size as bigger data content will be processed by one iteration, and
                # expected accuracy will be better. todo: maybe max value should be controlled? What if GPU supports
                #  window size 2048?
                # * Bigger window size speed up the processing time.
                window_size = int(find_nearest(np.power(2, np.arange(6, 1 + int(cfg.window_size_2pow_max))),
                                               min(image.shape[0], image.shape[1])))
                logging.info('Window size in smoothing predicting: {}'.format(window_size))

                pr_mask = predict_img_with_smooth_windowing(
                    image,
                    window_size=window_size,
                    # Minimal amount of overlap for windowing.
                    subdivisions=1/0.875,  # 1/0.5
                    nb_classes=cfg.cls_nb,
                    pred_func=(
                        lambda img_batch_subdiv: solver.model.predict(img_batch_subdiv)
                    ),
                    memmap_batch_size=memmap_batch_size,
                    temp_dir=src_proj_dir,
                    # todo: it should be controlled from config
                    use_group_d4=(True if sc_factor < 1.0 else predict_img_with_group_d4)
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


def predict_field(**kwarguments):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    def predict_field_(dataset: IDataProvider, src_proj_dir, skip_prediction, memmap_batch_size,
                       predict_img_with_group_d4, crop_size_px, overlap_px, merging_fname_head, debug):

        # Collect source data
        src_data = dataset.get_src_data()
        cfg = dataset.conf
        used_scales = len(cfg.pred_scale_factors)
        dest_img_fname = src_data[0][0]
        dest_himg_fname = src_data[1][0]
        solver = cfg.solver
        provider = cfg.provider_single

        is_success = PrepareData.build_from_project(src_proj_dir, cfg.mppx, dest_img_fname, dest_himg_fname)
        if not is_success:
            logging.error('Cannot prepare data')
            return None

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

            bbox = ((cr_x, cr_y), (cr_x2 - cr_x, cr_y2 - cr_y))
            bbox_list.append(bbox)
        logging.info('Source data represented by {} patches(in scale-space dimension {}) with cropping size {}'.
                     format(len(bbox_list)*used_scales, used_scales, crop_size_px))

        # Collect patches without processing them by NN.
        bbox_predictions = list()
        working_dtype = np.float16
        for bbox in bbox_list:
            err, pr_mask_list = predict_on_bbox(cfg, solver, None, src_proj_dir, cfg.pred_scale_factors,
                                                skip_prediction=False,
                                                memmap_batch_size=memmap_batch_size,
                                                predict_img_with_group_d4=predict_img_with_group_d4,
                                                bbox=bbox,
                                                working_dtype=working_dtype,
                                                debug=debug)
            bbox_predictions.append(dict({'bbox': bbox, 'pr_mask_list': pr_mask_list}))

        # Merge data
        merged_pr_mask_list = list()
        used_dtype = np.uint8
        # Possible numbers of channels
        channels_list = [1, 3, 4]
        # Here use int(cfg.cls_nb) to proper match int-list with float-cfg.cls_nb if it occurred
        channels_nb = channels_list[np.searchsorted(channels_list, int(cfg.cls_nb), side='left')]
        used_shape = (src_img_shape[0], src_img_shape[1], channels_nb)
        one_row_size_bytes = used_shape[1] * used_shape[2] * np.dtype(used_dtype).itemsize
        dstep = 1.0 / (overlap_px - 1)
        mask_template = np.arange(0.0, 1.0 + dstep / 2, dstep)[np.newaxis, :]
        for scale_ind in range(used_scales):
            merged_pr_mask_list_item = dict()
            bbox_predictions_fname_woext = '{}-sc{}'.format(merging_fname_head, scale_ind)
            merged_pr_mask_list_item['img'] = os.path.join(src_proj_dir,
                                                           '{}.png'.format(bbox_predictions_fname_woext))
            # merged_pr_mask_list_item['img_dtype'] = bbox_predictions[0]['pr_mask_list'][scale_ind]['img_dtype']
            merged_pr_mask_list_item['img_dtype'] = working_dtype
            merged_pr_mask_list_item['scale'] = bbox_predictions[0]['pr_mask_list'][scale_ind].get('scale')
            merged_pr_mask_list_item['shape'] = used_shape
            merged_pr_mask_list_item['skip_prediction'] = skip_prediction
            if os.path.isfile(merged_pr_mask_list_item['img']) and skip_prediction:
                merged_pr_mask_list.append(merged_pr_mask_list_item)
                logging.info('Merging skipped. Result will be reused from file {}'.format(
                    merged_pr_mask_list_item['img']))
                continue
            bbox_predictions_fname = os.path.join(src_proj_dir, '{}.tmp'.format(bbox_predictions_fname_woext))
            bbox_predictions_result = np.memmap(bbox_predictions_fname, dtype=used_dtype, mode='w+',
                                                shape=used_shape)
            bbox_predictions_result.fill(0)
            del bbox_predictions_result  # close file
            for bbox_ind, bbox_predictions_item in enumerate(bbox_predictions):
                xy, wh = bbox_predictions_item['bbox']
                pr_mask_list = bbox_predictions_item['pr_mask_list']
                pr_mask = pr_mask_list[scale_ind]
                #
                # merged_pr_mask_list_item['img_dtype'] = pr_mask['img_dtype']
                # merged_pr_mask_list_item['scale'] = pr_mask['scale']
                #
                num_patches = len(bbox_predictions) * used_scales
                patch_ind = len(bbox_predictions) * scale_ind + bbox_ind
                logging.info('Patch #{} (from {}) in processing...'.format(patch_ind + 1, num_patches))
                if not os.path.isfile(pr_mask.get('img')):
                    dataset = provider(read_sample,
                                       dest_img_fname,
                                       dest_himg_fname,
                                       bbox_predictions_item['bbox'],
                                       cfg,
                                       prep_getter=solver.get_prep_getter())

                    err, pr_mask_list = predict_on_bbox(cfg, solver, dataset, src_proj_dir, [pr_mask.get('scale')],
                                                        skip_prediction=skip_prediction,
                                                        memmap_batch_size=memmap_batch_size,
                                                        predict_img_with_group_d4=predict_img_with_group_d4,
                                                        bbox=bbox_predictions_item['bbox'],
                                                        working_dtype=working_dtype,
                                                        debug=debug)
                    if err != 0:
                        return None

                patch = cv2.imread(pr_mask.get('img'), cv2.IMREAD_UNCHANGED)
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

        # Release memory
        if solver.model is not None:
            del solver.model
            solver.model = None
        _backend.clear_session()
        gc.collect()

        return merged_pr_mask_list

    return predict_field_
