from kutils.EasyDict import EasyDict

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
cfg.mppx = 0.1
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24a/mppx{:.2f}'.format(cfg.mppx)
cfg.data_subset = 'muckpiles_multiclass_3'  # 'all_piles'  # 'rocks_1stPart_clean'
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = True
# dict with key/value: 'class'/['cls_1','cls_2'], or None
cfg.classes = {'class': ['muckpile', 'highwall']}  # {'class': ['muckpile', 'pile', 'car']}
cfg.cls_nb = (len(cfg.classes['class']) + 1 if len(cfg.classes['class']) > 1 else 1) if cfg.classes is not None else 1
cfg.min_mask_ratio = 0.0
cfg.img_wh = 512
cfg.img_wh_crop = 1024
# ==================================================================================================================== #
#                                                   Network Block
# ==================================================================================================================== #
cfg.backbone = 'efficientnetb3'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.pyramid_block_filters = 256  # default 256. User only for FPN-architecture
# ==================================================================================================================== #
#                                               Training Params Block
# ==================================================================================================================== #
cfg.test_aspect = 0.33
cfg.batch_size = 2
cfg.minimize_train_aug = False
cfg.lr = 0.0001
cfg.epochs = 4000
cfg.solution_dir = './solutions/{}/mppx{:.2f}/wh{}/{}/rgb{}/{}cls'.format(cfg.data_subset,
                                                                          cfg.mppx,
                                                                          cfg.img_wh,
                                                                          cfg.backbone,
                                                                          'a' if cfg.use_heightmap else '',
                                                                          cfg.cls_nb)
