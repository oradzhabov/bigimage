import numpy as np
from .. import get_submodules_from_kwargs


def preprocess(ximg):
    return ximg.astype(np.float32) / 255.0


def get_preprocessing_production(_):
    return preprocess


def get_solver(**kwarguments):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    from .SegmSolver import get_solver as get_segm_solver

    segm_solver = get_segm_solver(**kwarguments)

    class NativeUnet(segm_solver):
        def __init__(self, conf):
            super(NativeUnet, self).__init__(conf)
            self.weights_path = 'native_unet.h5'

        def _create_model(self, **kwargs):
            input_channels = 4 if self.conf.use_heightmap else 3
            dropout_value = 0.5
            # n_ch_exps = [6, 7, 8, 9, 10]
            n_ch_exps = [4, 5, 6]
            model = NativeUnet.create_model_ff(None, input_channels, dropout_value, n_ch_exps, self.conf.cls_nb)
            return model

        def get_prep_getter(self):
            return get_preprocessing_production

        @staticmethod
        def create_model_ff(img_hw, input_channels, dropout_value, n_ch_exps=[6, 7, 8, 9, 10], output_channels=1):
            k_size = (3, 3)  # size of filter kernel
            k_init = 'he_normal'  # kernel initializer

            if _backend.image_data_format() == 'channels_first':
                ch_axis = 1
                input_shape = (input_channels, img_hw, img_hw)
            elif _backend.image_data_format() == 'channels_last':
                ch_axis = 3
                input_shape = (img_hw, img_hw, input_channels)

            inp = _layers.Input(shape=input_shape)
            # lambd = Lambda(lambda x: x / 255) (inp)

            # encoder
            encodeds = []
            enc = inp
            for l_idx, n_ch in enumerate(n_ch_exps):
                enc = _layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(enc)
                enc = _layers.BatchNormalization()(enc)
                enc = _layers.Activation('relu')(enc)

                enc = _layers.Dropout(dropout_value)(enc)

                enc = _layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(enc)
                enc = _layers.BatchNormalization()(enc)
                enc = _layers.Activation('relu')(enc)

                encodeds.append(enc)
                if n_ch < n_ch_exps[-1]:  # do not run max pooling on the last encoding/downsampling step
                    enc = _layers.MaxPooling2D(pool_size=(2, 2))(enc)

            # decoder
            dec = enc
            decoder_n_chs = n_ch_exps[::-1][1:]
            for l_idx, n_ch in enumerate(decoder_n_chs):
                l_idx_rev = len(n_ch_exps) - l_idx - 2  #
                dec = _layers.Conv2DTranspose(filters=2 ** n_ch, kernel_size=k_size, strides=(2, 2), padding='same', use_bias=False, kernel_initializer=k_init)(dec)
                dec = _layers.BatchNormalization()(dec)
                dec = _layers.concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
                dec = _layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(dec)
                dec = _layers.BatchNormalization()(dec)
                dec = _layers.Activation('relu')(dec)

                dec = _layers.Dropout(dropout_value)(dec)

                dec = _layers.Conv2D(filters=2 ** n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(dec)
                dec = _layers.BatchNormalization()(dec)
                dec = _layers.Activation('relu')(dec)

            activation = 'sigmoid' if output_channels == 1 else 'softmax'
            outp = _layers.Conv2DTranspose(filters=output_channels, kernel_size=k_size, activation=activation, padding='same', kernel_initializer='glorot_normal')(dec)

            model = _models.Model(inputs=[inp], outputs=[outp])

            return model

    return NativeUnet
