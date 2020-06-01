import os
import numpy as np
import cv2
from kutils import PrepareData
from kmodel import model
from kmodel import data
from kmodel import config
from kmodel.train import read_sample, denormalize, visualize
from kmodel.kutils import get_contours
from kmodel.smooth_tiled_predictions import predict_img_with_smooth_windowing


if __name__ == "__main__":
    dst_mppx = 0.1
    src_proj_dir = 'F:/PROJECTS/Strayos/CUSTOMER.SUPPORT/2020.05.27/problemMuckpile/12105/output'
    # src_proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3554'  # big size
    # src_proj_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs/dev-site/3637'  # huge size(4GB-GPU impossible)

    dest_img_fname = os.path.join(src_proj_dir, 'tmp.png')
    dest_himg_fname = os.path.join(src_proj_dir, 'htmp.png')
    is_success = PrepareData.build_from_project(src_proj_dir, dst_mppx, dest_img_fname, dest_himg_fname)
    if not is_success:
        exit(-1)

    data_reader = read_sample
    dataset = data.Dataset2(data_reader, dest_img_fname, dest_himg_fname, backbone=config.cfg.backbone)
    image, _ = dataset[0]

    model, weights_path, metrics = model.create_model(conf=config.cfg, compile_model=True)

    pr_mask = predict_img_with_smooth_windowing(
        image,
        window_size=512,  # 512 enough for 4GB GPU. But it will be better if use 1024
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=1,
        pred_func=(
            lambda img_batch_subdiv: model.predict(img_batch_subdiv)
        )
    )

    pr_cntrs = get_contours((pr_mask.squeeze() * 255).astype(np.uint8))

    img_temp = (denormalize(image.squeeze()[..., :3]) * 255).astype(np.uint8)
    cv2.drawContours(img_temp, pr_cntrs, -1, (0, 0, 255), 5)
    cv2.imwrite(os.path.join(src_proj_dir, 'mpiles_result.png'), img_temp)

    visualize(
        title='{}'.format(dataset.get_fname(0)),
        result=img_temp
    )
