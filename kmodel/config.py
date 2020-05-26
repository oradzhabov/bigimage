from .EasyDict import EasyDict

cfg = EasyDict()

cfg.backbone = 'mobilenet'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.batch_size = 8
cfg.lr = 0.001
cfg.epochs = 20
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileLocal'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-04-13'
#cfg.img_wh = 512
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24/mppx0.25'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24/mppx0.10'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24/mppx0.05'
cfg.img_wh = 256  # 512
cfg.img_wh_crop = 512
cfg.pyramid_block_filters = 256  # default 256
