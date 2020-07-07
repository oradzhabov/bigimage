import os
from kutils import PrepareData
from kutils import VIAConverter
from config_rocks import cfg


if __name__ == "__main__":
    # PrepareData.prepare_dataset(cfg.root_projects_dir, cfg.data_dir, cfg.mppx, cfg.data_subset)

    # Find json-file in mask-subdir
    via_annotation_file = None
    maskdir = os.path.join(cfg.data_dir, 'masks.{}'.format(cfg.data_subset))
    for file in os.listdir(maskdir):
        if os.path.splitext(file)[1].lower() == '.json':
            via_annotation_file = os.path.join(maskdir, file)
            break

    img_fnames = VIAConverter.get_imgs(via_annotation_file)

    PrepareData.prepare_dataset(cfg.root_projects_dir, cfg.data_dir, cfg.mppx, cfg.data_subset, img_fnames)

    print('Prepare mask files...')
    VIAConverter.convert_to_images(via_annotation_file, 0.05, cfg.classes,
                                   mask_postprocess=cfg.mask_postprocess)
