from definitions import BIM_ROOT_DIR
from kutils.EasyDict import EasyDict
from solvers import *
from data_provider import *
from augmentation import *
import keras

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
cfg.mppx = 0.1
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24a/mppx{:.2f}'.format(cfg.mppx)
cfg.data_subset = 'muckpiles_multiclass_3'  # 'all_piles'  # 'rocks_1stPart_clean'
cfg.mask_postprocess = None
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = True
# dict with key/value: 'class'/['cls_1','cls_2'], or None. Dict assumes using background class and softmax activation
cfg.class_names = {'class': ['muckpile', 'highwall']}  # {'class': ['muckpile', 'pile', 'car']}
cfg.cls_nb = len(cfg.class_names['class']) + 1 if cfg.class_names is not None else 1
cfg.apply_class_weights = False
cfg.min_data_ratio = 0.0
cfg.min_mask_ratio = 0.0
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
cfg.backbone = 'efficientnetb5'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.pyramid_block_filters = 256  # default 256. User only for FPN-architecture
cfg.freeze_bn = False
# ==================================================================================================================== #
#                                               Training Params Block
# ==================================================================================================================== #
cfg.seed = 42
cfg.test_aspect = 0.33
cfg.batch_size = 2
cfg.batch_size_multiplier = 1
cfg.minimize_train_aug = False
cfg.lr = 0.0001
cfg.epochs = 4000
cfg.optimizer = keras.optimizers.Adam(cfg.lr)
cfg.solution_dir = '{}/solutions/{}/mppx{:.2f}/wh{}/{}/rgb{}/{}cls'.format(BIM_ROOT_DIR,
                                                                           cfg.data_subset,
                                                                           cfg.mppx,
                                                                           cfg.img_wh,
                                                                           cfg.backbone,
                                                                           'a' if cfg.use_heightmap else '',
                                                                           cfg.cls_nb)
