import logging
import gc
import os
import numpy as np
import cv2
from kutils import utilites
from config_stockpile import cfg
from kutils.VIAConverter import *
from tools.predict_contours import predict_contours


def test_prediction(src_proj_dir):
    # If enable following flag it will avoid long prediction and will try to read already created result.
    # Useful for debugging
    skip_prediction = False
    memmap_batch_size = 1  # 4 for config_rocks, 1 for config_stockpile
    predict_img_with_group_d4 = False  # REALLY HELPS, BUT 8+ TIMES SLOWER

    err_code, result_dict = predict_contours(cfg, src_proj_dir, skip_prediction, memmap_batch_size,
                                             predict_img_with_group_d4)
    if err_code != 0:
        return -1

    pr_cntrs_list, dataset, solver = result_dict['contours_px'], result_dict['dataset'], result_dict['solver']
    image, _ = dataset[0]
    image_fname = dataset.get_fname(0)

    # Make sure that we will drop out Alpha channel
    image = image[..., :3].copy()

    for class_ind, class_ctrs in enumerate(pr_cntrs_list):
        cv2.drawContours(image, class_ctrs, -1, dataset.get_color(class_ind), 0)

    result_png = 'classes_' + solver.signature() + '.png'
    cv2.imwrite(os.path.join(src_proj_dir, result_png), image[..., ::-1])

    utilites.visualize(
        title='{}'.format(src_proj_dir),
        result=image
    )
    del image
    gc.collect()

    logging.info('Creating VIA-json...')
    via_item = create_json_item(image_fname, pr_cntrs_list, cfg.classes)
    output_filename = 'via_' + os.path.splitext(os.path.basename(solver.weights_path))[0] + '.json'
    with open(os.path.join(src_proj_dir, output_filename), 'w', newline=os.linesep) as f:
        json.dump(via_item, f)

    return 0


if __name__ == "__main__":

    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dyno/1334'  # Especially good for rocks
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dyno/1341'  # small size
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12105'  # unseen during training BIG
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12120'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12363'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12266'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/10762'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12945'  # unseen during training(quite big)
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3554'  # big size
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3637'  # huge size(4GB-GPU impossible)
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12976'  # BAD PRODUCTION RESULT. SMALL
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7966'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12977'  # BAD PRODUCTION RESULT. SMALL
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7965'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12989'  # BAD PRODUCTION RESULT
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7964'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12189'
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7969'
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/dev-oktai/7128'  # big rocks, small ortho
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/dyno/2192'  # mppx 0.05 big rocks, big ortho ortho
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/dev-oktai/7145'  # orig:d2192 but 0.01 mppx. HUGE ROCKS
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/17042'  # ROCKS DETECTION/COLORING DEATH
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/17115'  # DEATH IN ROCKS POSTPROC EVEN ON SERVER
    ## proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/18618'  # 2020.08.13 Not well big rocks prediction
    #
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/8336'  #
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/9027'  # HUGE(split by 100m
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/11914'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/12072'  # middle
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/12107'  #
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/12140'  #
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/12237'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/13251'  #
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/14451'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/15495'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/16376'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/16511'  # small, VARIANCE ROCKS SIZE. IS IT STOCKPILE ?
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/18379'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/18514'  # small
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/19110'  # small. VARIANCE ROCKS SIZE. IS IT STOCKPILE ?
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/19945'  # middle
    # proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/20147'  # middle
    proj_dir = 'F:/DATASET/Strayos/StockPileDatasets/airzaar/20160'  # middle. BAD BOUNDARIES

    exit(test_prediction(proj_dir))
