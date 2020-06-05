import os
import numpy as np
import cv2
from kutils import PrepareData
from kmodel import data
from config import cfg
from kmodel.train import read_sample, denormalize, visualize
from kmodel.kutils import get_contours
from kmodel.smooth_tiled_predictions import predict_img_with_smooth_windowing
from solvers import SegmSolver, ProdSolver

if __name__ == "__main__":

    # src_proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dyno/1341'  # small size
    src_proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.unseen/airzaar/12105'  # unseen during training
    # src_proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3554'  # big size
    # src_proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3637'  # huge size(4GB-GPU impossible)

    dest_img_fname = os.path.join(src_proj_dir,
                                  'tmp_mppx{:.2f}.png'.format(cfg.mppx))
    dest_himg_fname = os.path.join(src_proj_dir,
                                   'htmp_mppx{:.2f}.png'.format(cfg.mppx)) if cfg.use_heightmap else None
    is_success = PrepareData.build_from_project(src_proj_dir, cfg.mppx, dest_img_fname, dest_himg_fname)
    if not is_success:
        exit(-1)

    model, _, _, prep_getter = SegmSolver().build(cfg, compile_model=False)
    if model is None:
        exit(-1)

    dataset = data.DataSingle(read_sample, dest_img_fname, dest_himg_fname, cfg, prep_getter=prep_getter)
    image, _ = dataset[0]

    pr_mask = predict_img_with_smooth_windowing(
        image,
        window_size=512,  # todo: 512 enough for 4GB GPU. But it will be better if use 1024
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=cfg.cls_nb,
        pred_func=(
            lambda img_batch_subdiv: model.predict(img_batch_subdiv)
        )
    )

    img_temp = (denormalize(image[..., :3]) * 255).astype(np.uint8)
    pr_cntrs_list = get_contours((pr_mask * 255).astype(np.uint8))
    for class_ind, class_ctrs in enumerate(pr_cntrs_list):
        cv2.drawContours(img_temp, class_ctrs, -1, dataset.get_color(class_ind), 5)

    cv2.imwrite(os.path.join(src_proj_dir, 'debug_mpiles_result.png'), img_temp[..., ::-1])

    visualize(
        title='{}'.format(src_proj_dir),
        result=img_temp
    )
