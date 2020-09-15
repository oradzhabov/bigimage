from definitions import BIM_ROOT_DIR
from kutils.EasyDict import EasyDict
from solvers import *
from data_provider import *
from augmentation import *

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/StockPileDatasets'
cfg.mppx = 0.1
cfg.data_dir = 'F:/DATASET/Strayos/StockPileDatasets.Result/2020-09-10/mppx{:.2f}'.format(cfg.mppx)
cfg.data_subset = 'stockpiles_1_segm'  # 'all_piles'  # 'rocks_1stPart_clean'
cfg.mask_postprocess = None
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = True
# dict with key/value: 'class'/['cls_1','cls_2'], or None
cfg.classes = None  # {'class': ['stockpile']}
cfg.cls_nb = (len(cfg.classes['class']) + 1 if len(cfg.classes['class']) > 1 else 1) if cfg.classes is not None else 1
#
cfg.min_mask_ratio = 0.0  # 0.0, 0.1, 0.25
cfg.img_wh = 512
cfg.img_wh_crop = 1024
cfg.solver = SegmSolver
cfg.provider = SemanticSegmentationDataProvider
cfg.provider_single = SemanticSegmentationSingleDataProvider
cfg.aug = BasicAug
cfg.pred_scale_factors = [1.0]  # List of scale factors. Each item should be less(or eq) to zero, order - descent
# ==================================================================================================================== #
#                                                   Network Block
# ==================================================================================================================== #
cfg.backbone = 'efficientnetb3'  # 'efficientnetb5'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.pyramid_block_filters = 256  # default 256. User only for FPN-architecture
# ==================================================================================================================== #
#                                               Training Params Block
# ==================================================================================================================== #
cfg.seed = 42
cfg.test_aspect = 0.33
cfg.batch_size = 2
cfg.batch_size_multiplier = 16
cfg.minimize_train_aug = False
cfg.lr = 0.0001  # 0.0002, 0.0005, 0.001
cfg.epochs = 4000
cfg.solution_dir = '{}/solutions/{}/mppx{:.2f}/wh{}/{}/rgb{}/{}cls'.format(BIM_ROOT_DIR,
                                                                           cfg.data_subset,
                                                                           cfg.mppx,
                                                                           cfg.img_wh,
                                                                           cfg.backbone,
                                                                           'a' if cfg.use_heightmap else '',
                                                                           cfg.cls_nb)