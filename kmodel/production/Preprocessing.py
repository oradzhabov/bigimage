import numpy as np
import cv2
import segmentation_models as sm
import keras
from .model import create_model_ff


def preprocess(ximg):
    # map to BGRA, since production uses BGRA channels order instead of this pipeline(RGBx)
    ximg = np.concatenate((ximg[..., :3][..., ::-1], ximg[..., 3:]), axis=-1)

    mean_bgra = [84.062949, 85.64098229, 83.47105794, 127.5]
    std_bgra = [73.83440617, 74.93718162, 75.32351374, 127.5]

    ximg = np.float32(ximg)
    ximg[:, :, 0] = (ximg[:, :, 0] - mean_bgra[0]) / std_bgra[0]
    ximg[:, :, 1] = (ximg[:, :, 1] - mean_bgra[1]) / std_bgra[1]
    ximg[:, :, 2] = (ximg[:, :, 2] - mean_bgra[2]) / std_bgra[2]
    ximg[:, :, 3] = (ximg[:, :, 3] - mean_bgra[3]) / std_bgra[3]

    # Put here resizing according to working scale(mppx = 0.25)
    trained_mppx = 0.25
    sc_factor = trained_mppx / 0.25  # todo: ???
    nw = int(ximg.shape[1] * sc_factor)
    nh = int(ximg.shape[0] * sc_factor)
    nw = nw + (nw % 2)
    nh = nh + (nh % 2)

    ximg = cv2.resize(ximg, (nw, nh), interpolation=cv2.INTER_CUBIC)

    return ximg


def get_preprocessing_production(backbone):
    return preprocess


def create_model_production(conf, compile_model=True):
    n_classes = 1
    model = create_model_ff(img_hw=None, input_channels=4, dropout_value=0.0)

    metrics = None
    if compile_model:
        # define optimizer
        optimizer = keras.optimizers.Adam(conf.lr)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        # dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        # total_loss = dice_loss + (3 * focal_loss)
        total_loss = focal_loss
        # total_loss = 'binary_crossentropy'

        metrics = [sm.metrics.IOUScore(threshold=0.996), sm.metrics.FScore(threshold=0.996)]

        # compile model with defined optimizer, loss and metrics
        model.compile(optimizer, total_loss, metrics)

    model.load_weights('./mp_cntr_ff2_p1a_3_weights-ep495-loss0.00879-val_loss0.00905-val_acc0.99146-val_mean_iou0.95021.h5')

    return model, None, metrics
