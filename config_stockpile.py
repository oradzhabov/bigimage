from .definitions import BIM_ROOT_DIR
from .kutils.EasyDict import EasyDict
from . import bin_keras
from .data_provider import *
from .augmentation import *

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/StockPileDatasets'
cfg.mppx = 0.1
cfg.data_dir = 'F:/DATASET/Strayos/StockPileDatasets.Result/2020-09-10/mppx{:.2f}'.format(cfg.mppx)
# cfg.data_subset = 'stockpiles_2_segm'  # best 2020.10.06
# cfg.data_subset = 'stockpiles_3_segm'  # extra class "water" did not improve general accuracy
# cfg.data_subset = 'stockpiles_2a_segm'  # train/val split 0.8/0.2 became worse accuracy that it was with 0.66/0.33
cfg.data_subset = 'stockpiles_4_segm'  # add samples, min_data_ratio 0.25, train/val 0.66/0.33
cfg.mask_postprocess = None
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = True
# dict with key/value: 'class'/['cls_1','cls_2'], or None. Dict assumes using background class and softmax activation.
# dict with 1-class means using softmax, which can be weighted for class imbalance reduction
# cfg.class_names = {'class': ['stockpile', 'water']}
cfg.class_names = {'class': ['stockpile']}
cfg.cls_nb = len(cfg.class_names['class']) + 1 if cfg.class_names is not None else 1
#
cfg.apply_class_weights = True
cfg.min_data_ratio = 0.25
cfg.min_mask_ratio = 0.0
cfg.thin_out_train_ratio = 1.0  # 0- drop out all samples, 1- don't drop samples
cfg.img_wh = 512
cfg.img_wh_crop = 1024
cfg.solver = bin_keras.SegmSolver
cfg.provider = SemanticSegmentationDataProvider
cfg.provider_single = SemanticSegmentationSingleDataProvider
cfg.aug = BasicAug
cfg.pred_scale_factors = [1.0]  # List of scale factors. Each item should be less(or eq) to zero, order - descent
# ==================================================================================================================== #
#                                                   Network Block
# ==================================================================================================================== #
# useful regarding EfficientTent: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
# batch_size(64), efficientnetb6(for 512 cfg.img_wh)
cfg.backbone = 'efficientnetb6'  # 'efficientnetb5'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.pyramid_block_filters = 256  # default 256. User only for FPN-architecture
cfg.freeze_bn = False
cfg.dropout_rate_mult = 1.0  # [0 ... 1 ... inf) => [0 ... same ... 1]
# ==================================================================================================================== #
#                                               Training Params Block
# ==================================================================================================================== #
cfg.seed = 42
cfg.test_aspect = 0.33
cfg.batch_size = 2
cfg.batch_size_multiplier = 32  # To simulate the enlarging of BS, gradient accumulation will be utilized
cfg.minimize_train_aug = False
cfg.epochs = 400
cfg.lr = 0.0001  # Initial LR
# TIPS:
# * SGD with proper LR/BS should provide smooth loss-function(accuracy metric could be not smooth). If loss looks not
# smooth, LR/BS should be tweaked.
# * Adam not good for warm restarts(in Snapshot Ensembles or restart from previous checkpoint).
cfg.optimizer = bin_keras.modules['optimizers'].Adam(cfg.lr)
cfg.solution_dir = '{}/solutions/{}/mppx{:.2f}/wh{}/{}/rgb{}/{}cls'.format(BIM_ROOT_DIR,
                                                                           cfg.data_subset,
                                                                           cfg.mppx,
                                                                           cfg.img_wh,
                                                                           cfg.backbone,
                                                                           'a' if cfg.use_heightmap else '',
                                                                           cfg.cls_nb)
cfg.callbacks = [
    # bin_keras.LR_PolynomialDecay(cfg.epochs, cfg.lr, 1.0)
    # bin_keras.LR_SnapshotEnsembleDecay(cfg.epochs, cfg.epochs // 40, cfg.lr, cfg.solution_dir)
]
