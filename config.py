from kutils.EasyDict import EasyDict

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
cfg.mppx = 0.1
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24a/mppx{:.2f}'.format(cfg.mppx)
cfg.data_subset = 'all_piles'
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = False
cfg.min_mask_ratio = 0.0
cfg.img_wh = 512
cfg.img_wh_crop = 1024
# ==================================================================================================================== #
#                                                   Network Block
# ==================================================================================================================== #
cfg.backbone = 'mobilenet'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = None  # 'imagenet'
cfg.encoder_freeze = False
cfg.pyramid_block_filters = 256  # default 256
# ==================================================================================================================== #
#                                               Training Params Block
# ==================================================================================================================== #
cfg.test_aspect = 0.33
cfg.batch_size = 4
cfg.minimize_train_aug = False
cfg.lr = 0.0005
cfg.epochs = 4000
