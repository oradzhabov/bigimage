from .EasyDict import EasyDict

cfg = EasyDict()

cfg.backbone = 'efficientnetb3'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.batch_size = 1
cfg.lr = 0.00001
cfg.epochs = 30
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileLocal'
# cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-04-13'
#cfg.img_wh = 512
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24'
cfg.img_wh = 128
cfg.img_wh_crop = cfg.img_wh * 4
cfg.pyramid_block_filters = 256  # default 256
