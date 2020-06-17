import os
import segmentation_models as sm
import keras
from . import ISolver


class SegmSolver(ISolver):
    def __init__(self, conf):
        super(SegmSolver, self).__init__(conf)

        self.weights_path = 'unet_{}_mppx{:.2f}_wh{}_rgb{}_{}cls_{}.h5'.format(self.conf.backbone,
                                                                               self.conf.mppx,
                                                                               self.conf.img_wh,
                                                                               'a' if self.conf.use_heightmap else '',
                                                                               self.conf.cls_nb,
                                                                               self.conf.data_subset)

    def _create(self, compile_model=True):
        solution_path = os.path.normpath(os.path.abspath(self.conf.solution_dir))
        if not os.path.isdir(solution_path):
            os.makedirs(solution_path)
        self.weights_path = os.path.join(solution_path, self.weights_path)

        activation = 'sigmoid' if self.conf.cls_nb == 1 else 'softmax'

        weights_init_path = self.weights_path if os.path.isfile(self.weights_path) else None

        base_model = sm.Unet(self.conf.backbone,
                             input_shape=(None, None, 3),
                             classes=self.conf.cls_nb,
                             activation=activation,
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

            print('Model has been initialized from file: {}'.format(weights_init_path))
        else:
            # Provide model info for first call of model
            self.model.summary()

        if compile_model:
            # todo: actually compilation could be moved out to parent class.
            #  But there are some limitations - IOUScore/FScore threshold

            # define optimizer
            optimizer = keras.optimizers.Adam(self.conf.lr)

            # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
            dice_loss = sm.losses.DiceLoss()
            focal_loss = sm.losses.BinaryFocalLoss() if self.conf.cls_nb == 1 else sm.losses.CategoricalFocalLoss()
            total_loss = dice_loss + focal_loss
            # total_loss = focal_loss
            # total_loss = 'binary_crossentropy'

            # Value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
            threshold = 0.5
            self.metrics = [sm.metrics.IOUScore(threshold=threshold), sm.metrics.FScore(threshold=threshold), sm.metrics.f2_score]

            # compile model with defined optimizer, loss and metrics
            self.model.compile(optimizer, total_loss, self.metrics)

        # base_model.load_weights('./unet_mobilenet_wh512_rgb_f1083.h5')

    def get_prep_getter(self):
        return sm.get_preprocessing
