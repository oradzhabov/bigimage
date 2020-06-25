import os
import numpy as np
import cv2
from kutils import PrepareData
from kmodel import data
from config import cfg
from kmodel.train import read_sample, denormalize, visualize
from kmodel.kutils import get_contours
from kmodel.smooth_tiled_predictions import predict_img_with_smooth_windowing
from solvers import *
from kutils.VIAConverter import *
from data_provider import *


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def test_prediction(src_proj_dir):
    dest_img_fname = os.path.join(src_proj_dir,
                                  'tmp_mppx{:.2f}.png'.format(cfg.mppx))
    dest_himg_fname = os.path.join(src_proj_dir,
                                   'htmp_mppx{:.2f}.png'.format(cfg.mppx)) if cfg.use_heightmap else None
    is_success = PrepareData.build_from_project(src_proj_dir, cfg.mppx, dest_img_fname, dest_himg_fname)
    if not is_success:
        exit(-1)

    model = None
    solver = SegmSolver(cfg)
    model, weights_path, _ = solver.build(compile_model=False)
    if model is None:
        exit(-1)

    dataset = SemanticSegmentationSingleDataProvider(read_sample,
                                                     dest_img_fname,
                                                     dest_himg_fname,
                                                     cfg,
                                                     prep_getter=solver.get_prep_getter())
    image, _ = dataset[0]
    image_fname = dataset.get_fname(0)
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
            use_batch_1=True
        )
        cv2.imwrite(os.path.join(src_proj_dir, predict_png), (pr_mask * 255).astype(np.uint8))
    else:
        pr_mask = cv2.imread(os.path.join(src_proj_dir, predict_png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    postproc_getter = solver.get_post_getter()
    post_processor = postproc_getter()
    pr_mask = post_processor(pr_mask)

    img_temp = (denormalize(image[..., :3]) * 255).astype(np.uint8)
    pr_cntrs_list = get_contours((pr_mask * 255).astype(np.uint8))
    for class_ind, class_ctrs in enumerate(pr_cntrs_list):
        cv2.drawContours(img_temp, class_ctrs, -1, dataset.get_color(class_ind), 5)

    result_png = 'classes_' + os.path.splitext(os.path.basename(solver.weights_path))[0] + '.png'
    cv2.imwrite(os.path.join(src_proj_dir, result_png), img_temp[..., ::-1])

    visualize(
        title='{}'.format(src_proj_dir),
        result=img_temp
    )

    print('Creating VIA-json...')
    via_item = create_json_item(image_fname, pr_cntrs_list, cfg.classes)
    output_filename = 'via_' + os.path.splitext(os.path.basename(solver.weights_path))[0] + '.json'
    with open(os.path.join(src_proj_dir, output_filename), 'w', newline=os.linesep) as f:
        json.dump(via_item, f)


if __name__ == "__main__":

    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dyno/1341'  # small size
    proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12105'  # unseen during training BIG
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12120'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12363'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12266'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/10762'  # unseen during training
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12945'  # unseen during training(quite big)
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3554'  # big size
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3637'  # huge size(4GB-GPU impossible)
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12976'  # BAD PRODUCTION RESULT. SMALL
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7966'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12977'  # BAD PRODUCTION RESULT. SMALL
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7965'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12989'  # BAD PRODUCTION RESULT
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7964'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12189'
    # proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/qa/7969'
    test_prediction(proj_dir)
