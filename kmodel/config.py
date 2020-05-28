from .EasyDict import EasyDict

cfg = EasyDict()

cfg.backbone = 'mobilenet'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.batch_size = 4
cfg.lr = 0.0001
cfg.epochs = 400
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileLocal'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-04-13'
# cfg.img_wh = 512
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24/mppx0.25'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24/mppx0.10'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24/mppx0.05'
cfg.img_wh = 512  # 512
cfg.img_wh_crop = 1024
cfg.pyramid_block_filters = 256  # default 256
