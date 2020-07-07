import os
import numpy as np
import cv2
from kutils import utilites
from config_rocks import cfg
from kutils.VIAConverter import *
from tools.predict_contours import predict_contours


def test_prediction(src_proj_dir):
    # If enable following flag it will avoid long prediction and will try to read already created result.
    # Useful for debugging
    skip_prediction = False
    memmap_batch_size = 6

    err_code, result_dict = predict_contours(cfg, src_proj_dir, skip_prediction, memmap_batch_size)
    if err_code != 0:
        return -1

    pr_cntrs_list, dataset, solver = result_dict['contours_px'], result_dict['dataset'], result_dict['solver']
    image, _ = dataset[0]
    image_fname = dataset.get_fname(0)

    img_temp = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)
    for class_ind, class_ctrs in enumerate(pr_cntrs_list):
        cv2.drawContours(img_temp, class_ctrs, -1, dataset.get_color(class_ind), 0)

    result_png = 'classes_' + os.path.splitext(os.path.basename(solver.weights_path))[0] + '.png'
    cv2.imwrite(os.path.join(src_proj_dir, result_png), img_temp[..., ::-1])

    utilites.visualize(
        title='{}'.format(src_proj_dir),
        result=img_temp
    )

    print('Creating VIA-json...')
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
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7965'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12989'  # BAD PRODUCTION RESULT
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7964'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12189'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7969'
    proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/dev-oktai/7128'  # big rocks

    exit(test_prediction(proj_dir))
