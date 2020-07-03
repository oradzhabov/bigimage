from . import definitions
from kutils.EasyDict import EasyDict
from kutils.mask_to_dist import mask_to_dist
from solvers import *
from data_provider import *
from augmentation import *

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
cfg.mppx = 0.01
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24a/mppx{:.2f}'.format(cfg.mppx)
cfg.data_subset = 'rocks_1stPart_clean-dev-site_3611_3638'  # 'rocks_1stPart_clean-dev-site_3611' 'rocks_1stPart_clean'
cfg.mask_postprocess = mask_to_dist  # mask needs to be mapped from binary to normalized distance
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = False
# dict with key/value: 'class'/['cls_1','cls_2'], or None
cfg.classes = None  # {'class': ['muckpile', 'highwall']}  # {'class': ['muckpile', 'pile', 'car']}
cfg.cls_nb = (len(cfg.classes['class']) + 1 if len(cfg.classes['class']) > 1 else 1) if cfg.classes is not None else 1
# cannot set 0 for rocks, because there are lot of unlabeled areas with real rocks(all rocks cannot be tagged)
cfg.min_mask_ratio = 0.01
cfg.img_wh = 256
cfg.img_wh_crop = 512  # to minimize 'fake' area during scale down aug, crop it with bigger size than img_wh
cfg.solver = RegrRocksSolver
cfg.provider = RegressionSegmentationDataProvider
cfg.provider_single = RegressionSegmentationSingleDataProvider
cfg.aug = RocksAug
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
cfg.seed = 42
cfg.test_aspect = 0.25
cfg.batch_size = 4
cfg.minimize_train_aug = False
cfg.lr = 0.001
cfg.epochs = 4000
cfg.solution_dir = '{}/solutions/{}/mppx{:.2f}/wh{}/{}/rgb{}/{}cls'.format(definitions.BIM_ROOT_DIR,
                                                                           cfg.data_subset,
                                                                           cfg.mppx,
                                                                           cfg.img_wh,
                                                                           cfg.backbone,
                                                                           'a' if cfg.use_heightmap else '',
                                                                           cfg.cls_nb)
