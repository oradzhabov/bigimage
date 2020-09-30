import logging
import os
import sys
import numpy as np
import segmentation_models as sm
import keras
from . import ISolver
from .optimizers import AccumOptimizer
sys.path.append(sys.path[0] + "/..")
from kutils import utilites


def freeze_bn(obj):
    for layer in obj.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        elif isinstance(layer, keras.Model):
            freeze_bn(layer)


class SegmSolver(ISolver):
    def __init__(self, conf):
        super(SegmSolver, self).__init__(conf)
        self.weights_path = 'unet.h5'
        self.activation = 'sigmoid' if self.conf.cls_nb == 1 else 'softmax'

        """
        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if self.conf.cls_nb == 1 else sm.losses.CategoricalFocalLoss()
        # Practice hints:
        # 'focal_loss' could maximize F1, but it has no any affect to F2 (even for train-set) when dataset extended by
        # "rich" background samples. Even 0.001 value of 'focal_loss' could make 0.8 F1 and 0.1 F2.
        # If add 'dice_loss', total loss will grown from 0.001 (focal_loss only) to 0.9. F2 became Increases with heavy
        # background imbalance.
        self.total_loss = dice_loss + focal_loss
        # self.total_loss = focal_loss
        # total_loss = 'binary_crossentropy'

        # Value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
        threshold = 0.5
        self.metrics = [sm.metrics.IOUScore(threshold=threshold), sm.metrics.FScore(threshold=threshold),
                        sm.metrics.f2_score]
        """
    def _create_metrics(self, **kwargs):
        # How to control the FocalLoss parameters:
        # https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/
        # https://leimao.github.io/blog/Focal-Loss-Explained/
        class_weights = kwargs['class_weights'] if 'class_weights' in kwargs else None
        if class_weights is not None:
            if self.conf.cls_nb != len(class_weights):
                logging.info('Class Nb {} not matched with Weights Nb {}.'
                             'Class Weights will not be used in model'.format(self.conf.cls_nb, len(class_weights)))
                class_weights = None
            else:
                logging.info('Metrics uses class weights: {}'.format(class_weights))
        alpha = 1.0
        gamma = 2.0
        if self.conf.cls_nb == 1:
            focal_loss = sm.losses.BinaryFocalLoss(alpha=alpha, gamma=gamma)
        else:
            focal_loss = sm.losses.CategoricalFocalLoss(alpha=alpha, gamma=gamma)

        if self.conf.cls_nb == 1:
            ce_loss = sm.losses.BinaryCELoss()
        else:
            ce_loss = sm.losses.CategoricalCELoss(class_weights=class_weights)

        # Why we need use BCE(or FL) instead of DiceLoss at least at the beginning:
        # https://stats.stackexchange.com/a/344244
        self.total_loss = ce_loss
        #
        # Dice(beta=2) = 1 - F2-score
        # self.total_loss = ce_loss + sm.losses.DiceLoss(beta=2, class_weights=class_weights)

        # Value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
        # here should not be weights
        threshold = 0.5
        self.metrics = [sm.metrics.IOUScore(threshold=threshold),
                        sm.metrics.FScore(beta=1, threshold=threshold),
                        sm.metrics.FScore(beta=2, threshold=threshold)]

    def _create(self, compile_model=True, **kwargs):
        solution_path = os.path.normpath(os.path.abspath(self.conf.solution_dir))
        if not os.path.isdir(solution_path):
            os.makedirs(solution_path)
        self.weights_path = os.path.join(solution_path, self.weights_path)

        logging.info('Trying to find model\'s weight by path \"{}\"'.format(self.weights_path))
        weights_init_path = self.weights_path if os.path.isfile(self.weights_path) else None

        base_model = sm.Unet(self.conf.backbone,
                             input_shape=(None, None, 3),
                             classes=self.conf.cls_nb,
                             activation=self.activation,
                             encoder_weights=self.conf.encoder_weights if weights_init_path is None else None,
                             encoder_freeze=self.conf.encoder_freeze,
                             # pyramid_block_filters=conf.pyramid_block_filters,  # default 256
                             weights=None,
                             # pyramid_dropout=0.25
                             )

        if self.conf.use_heightmap:
            # Add extra input layer to map custom-channel number to 3-channels
            # input model which can use pre-trained weights
            inp = keras.layers.Input(shape=(None, None, 4))
            l1 = keras.layers.Conv2D(32, (1, 1), use_bias=False)(inp)
            l1 = keras.layers.BatchNormalization()(l1)
            l1 = keras.layers.Conv2D(3, (1, 1), use_bias=False)(l1)
            l1 = keras.layers.BatchNormalization()(l1)
            out = base_model(l1)

            self.model = keras.models.Model(inp, out, name=base_model.name)
        else:
            self.model = base_model

        if weights_init_path is not None:
            # loading model weights
            self.model.load_weights(weights_init_path)

            logging.info('Model has been initialized from file: \"{}\"'.format(weights_init_path))

            # Sometime "The BathcNormalization layers need to be kept frozen.
            # (more details: https://keras.io/guides/transfer_learning/). If they are also turned to trainable, the
            # first epoch after unfreezing will significantly reduce accuracy."
            # from: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
            if hasattr(self.conf, 'freeze_bn'):
                if self.conf.freeze_bn:
                    freeze_bn(self.model)
                    logging.info('BatchNormalization layers have been frozen')
        else:
            # Provide model info for first call of model
            self.model.summary()
            logging.info('Model\'s weights has not been found by path \"{}\"'.format(self.weights_path))

        if compile_model:
            # todo: actually compilation could be moved out to parent class.
            #  But there are some limitations - IOUScore/FScore threshold

            if weights_init_path is not None and isinstance(self.conf.optimizer, keras.optimizers.Adam):
                logging.warning('There is no sense to expect a good training convergence '
                                'after warm restarting with Adam optimizer')
                # src: https://ruder.io/deep-learning-optimization-2017/

            # define optimizer
            optimizer = self.conf.optimizer
            if self.conf.batch_size_multiplier > 1:
                optimizer = AccumOptimizer.AccumOptimizer(optimizer, self.conf.batch_size_multiplier)

            # compile model with defined optimizer, loss and metrics
            self.model.compile(optimizer, self.total_loss, self.metrics)

    def get_prep_getter(self):
        return sm.get_preprocessing

    def post_predict(self, pr_result):
        if pr_result.shape[2] > 1:
            # Left maximum component of multi-channels input
            mg = np.meshgrid(np.arange(pr_result.shape[1]), np.arange(pr_result.shape[0]))
            vx = mg[0].reshape(-1)
            vy = mg[1].reshape(-1)
            vz = np.argmax(pr_result, axis=2).reshape(-1)
            pr_result *= 0
            pr_result[vy, vx, vz] = 1
        else:
            # Binarize 1-channels input
            pr_result = np.where(pr_result > 0.5, 1.0, 0.0)

        return pr_result.astype(np.bool)

    def get_contours(self, pr_mask_list):
        pr_mask = self._get_avg_prob_field(pr_mask_list)
        return utilites.get_contours((pr_mask * 255).astype(np.uint8))

    def monitoring_metric(self):
        return 'val_f1-score', 'max'
