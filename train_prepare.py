import logging
import os
from kutils import PrepareData
from kutils import VIAConverter
from config_stockpile import cfg


if __name__ == "__main__":
    json_mppx = 0.05
    # PrepareData.prepare_dataset(cfg.root_projects_dir, cfg.data_dir, cfg.mppx, cfg.data_subset)
    if True:  # create VIA-json by shapefiles.
        import json

        maskdir = os.path.join(cfg.data_dir, 'masks.{}'.format(cfg.data_subset))
        if not os.path.exists(maskdir):
            os.makedirs(maskdir)

        customer_list = ['airzaar']
        proj_list = [8018, 8022, 8048, 8225, 8227, 8715, 9100, 9151, 9154, 9155, 9200, 9221, 9264, 9306, 9323, 9459,
                     9778, 10177, 10439, 10730, 11541, 11654, 11667, 11686, 9098, 13940, 12072, 12107, 12140]

        via_items = PrepareData.map_shapefiles_to_via(cfg.root_projects_dir, customer_list, proj_list,
                                                      'shp.shp', cfg.mppx)

        via_json_fname = os.path.join(maskdir, 'via_shp_{}.json'.format(cfg.data_subset))
        with open(via_json_fname, 'w') as outfile:
            json.dump(via_items, outfile)

        # Since before in this block we provided mppx in via-json equal to cfg.mppx, we should specify it directly
        json_mppx = cfg.mppx

    # Find json-file in mask-subdir
    via_annotation_file = None
    maskdir = os.path.join(cfg.data_dir, 'masks.{}'.format(cfg.data_subset))
    for file in os.listdir(maskdir):
        if os.path.splitext(file)[1].lower() == '.json':
            via_annotation_file = os.path.join(maskdir, file)
            break

    img_fnames = VIAConverter.get_imgs(via_annotation_file)

    PrepareData.prepare_dataset(cfg.root_projects_dir, cfg.data_dir, cfg.mppx, cfg.data_subset, img_fnames)

    logging.info('Prepare mask files...')
    # Regression task could be declared without attribute `class_names`.
    class_names = cfg.class_names if hasattr(cfg, 'class_names') else None
    VIAConverter.convert_to_images(via_annotation_file, json_mppx, class_names,
                                   mask_postprocess=cfg.mask_postprocess)
