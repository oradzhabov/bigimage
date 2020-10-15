from .definitions import BIM_ROOT_DIR
from .kutils.EasyDict import EasyDict
from .kutils.mask_to_dist import mask_to_dist
from . import bin_keras
from .data_provider import *
from .augmentation import *

cfg = EasyDict()

# ==================================================================================================================== #
#                                                 Source Data Block
# ==================================================================================================================== #
cfg.root_projects_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs'
cfg.mppx = 0.01
cfg.data_dir = 'F:/DATASET/Strayos/MuckPileDatasets.outputs.Result.4/2020-05-24a/mppx{:.2f}'.format(cfg.mppx)
cfg.data_subset = 'rocks_1stPart_clean-dev-site_3611_3638_dev-oktai_7128'  # 'rocks_1stPart_clean-dev-site_3611' 'rocks_1stPart_clean'
cfg.mask_postprocess = mask_to_dist  # mask needs to be mapped from binary to normalized distance
# ==================================================================================================================== #
#                                                Sample Space Block
# ==================================================================================================================== #
cfg.use_heightmap = False
# dict with key/value: 'class'/['cls_1','cls_2'], or None. Dict assumes using background class and softmax activation.
# If there is no attribute `class_names` it means that task is not classification but regression
cfg.cls_nb = 1
cfg.apply_class_weights = False
cfg.min_data_ratio = 0.0
# cannot set 0 for rocks, because there are lot of unlabeled areas with real rocks(all rocks cannot be tagged)
cfg.min_mask_ratio = 0.1
cfg.thin_out_train_ratio = 1.0  # 0- drop out all samples, 1- don't drop samples
cfg.img_wh = 256
# To minimize 'fake' area during scale down aug, crop it with bigger size than img_wh
# Note: if crop with 1024 and img_wh 512, it will degrade validation metrics because in fact not all rocks labeled and
# lot of non-labeled rocks will disturb ANN (data conflict)
cfg.img_wh_crop = 512
cfg.solver = bin_keras.RegrRocksSolver
cfg.provider = RegressionSegmentationDataProvider
cfg.provider_single = RegressionSegmentationSingleDataProvider
cfg.aug = RocksAug
cfg.pred_scale_factors = [1.0, 0.25]  # List of scale factors. Each item should be less(or eq) to zero, order - descent
cfg.window_size_2pow_max = 10  # Power of 2 maximum window size during inference
# ==================================================================================================================== #
#                                                   Network Block
# ==================================================================================================================== #
cfg.backbone = 'efficientnetb3'  # 'resnet34'  # 'mobilenet'  # 'efficientnetb3'
cfg.encoder_weights = 'imagenet'
cfg.encoder_freeze = False
cfg.pyramid_block_filters = 256  # default 256. User only for FPN-architecture
cfg.freeze_bn = False
# ==================================================================================================================== #
#                                               Training Params Block
# ==================================================================================================================== #
cfg.seed = 42
cfg.test_aspect = 0.33
cfg.batch_size = 4
cfg.batch_size_multiplier = 1
cfg.minimize_train_aug = False
cfg.lr = 0.001  # start from 0.001
cfg.epochs = 4000
# Adam() not good for warm restarts(in Ensembles or prev checkpoint)
cfg.optimizer = bin_keras.modules['optimizers'].Adam(cfg.lr)
cfg.solution_dir = '{}/solutions/{}/mppx{:.2f}/wh{}/{}/rgb{}/{}cls'.format(BIM_ROOT_DIR,
                                                                           cfg.data_subset,
                                                                           cfg.mppx,
                                                                           cfg.img_wh,
                                                                           cfg.backbone,
                                                                           'a' if cfg.use_heightmap else '',
                                                                           cfg.cls_nb)
