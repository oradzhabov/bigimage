import os
import segmentation_models as sm
import keras


def create_model(conf, compile_model=True):
    weights_path = './fpn_{}_wh{}_pbf{}.h5'.format(conf.backbone,
                                                   conf.img_wh,
                                                   conf.pyramid_block_filters)
    n_classes = 1
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    weights_init_path = weights_path if os.path.isfile(weights_path) else None

    base_model = sm.FPN(conf.backbone,
                        input_shape=(None, None, 3),
                        classes=n_classes,
                        activation=activation,
                        encoder_weights=conf.encoder_weights if weights_init_path is None else None,
                        encoder_freeze=conf.encoder_freeze,
                        pyramid_block_filters=conf.pyramid_block_filters,  # default 256
                        weights=None,
                        pyramid_dropout=0.25
                        )

    # Add extra input layer to map custom-channel number to 3-channels input model which can use pre-trained weights
    inp = keras.layers.Input(shape=(None, None, 4))
    l1 = keras.layers.Conv2D(3, (1, 1), use_bias=False)(inp)
    l1 = keras.layers.BatchNormalization()(l1)
    out = base_model(l1)

    model = keras.models.Model(inp, out, name=base_model.name)

    if weights_init_path is not None:
        # loading model weights
        model.load_weights(weights_init_path)

        print('Model has been initialized from file: {}'.format(weights_init_path))
    else:
        # Provide model info for first call of model
        model.summary()

    preprocess_input = sm.get_preprocessing(conf.backbone)

    metrics = None
    if compile_model:
        # define optimizer
        optimizer = keras.optimizers.Adam(conf.lr)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        # total_loss = dice_loss + (3 * focal_loss)
        total_loss = focal_loss
        # total_loss = 'binary_crossentropy'

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile model with defined optimizer, loss and metrics
        model.compile(optimizer, total_loss, metrics)

    return model, weights_path, preprocess_input, metrics
