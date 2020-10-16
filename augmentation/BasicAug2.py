import albumentations as alb
import cv2
from . import IAug
from .albumentations2 import *


class BasicAug2(IAug):
    def get_training_augmentation(self, conf, is_stub=False):
        train_transform = [

            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            # Equal probability to obtain one from 4 possible position: always_apply=True
            alb.RandomRotate90(always_apply=True),
            #
            # apply shadow augmentation before any color augmentation
            RandomShadow2(shadow_roi=(0, 0, 1, 1), shadow_dimension=3, always_apply=True),
            #
            alb.OneOf(
                [
                    alb.Blur(blur_limit=[3, 5], p=1),
                    alb.GaussNoise(var_limit=20, mean=0, p=1),
                ],
                p=0.9,  # remain possibility to not apply these filters
            ),
            #
            # scale_limit ((float, float) or float) – scaling factor range. If scale_limit is a single float value,
            # the range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).
            # rotate_limit ((int, int) or int) – rotation range. If rotate_limit is a single int value,
            # the range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
            # shift_limit ((float, float) or float) – shift factor range for both height and width. If shift_limit is
            # a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper
            # bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
            # * Remain shifting to guaranty that aug will operate shifting even if random crop will not help when crop-
            # size will be equal to window size.
            # * Rotation 45 will guaranty covering all possible rotations if group-d4 has been implemented before
            alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, border_mode=0, shift_limit=0.0625,
                                 interpolation=cv2.INTER_AREA, p=1),

            # Pad side of the image / max if side is less than desired number.
            alb.PadIfNeeded(min_height=conf.img_wh, min_width=conf.img_wh, always_apply=True, border_mode=0),

            # Apply one of Brightness or Contrast filter to avoid very low/high performing and remain quite big ranges
            # of each filter
            alb.OneOf(
                [
                    RandomBrightnessContrast2(brightness_limit=0.2, contrast_limit=0.0, p=1),
                    RandomBrightnessContrast2(brightness_limit=0.0, contrast_limit=0.2, p=1),
                ],
                p=1
            ),

            RandomGamma2(p=1),

            HueSaturationValue2(hue_shift_limit=15, sat_shift_limit=(-20, 10), val_shift_limit=0, p=1.0),

            # * Orthographic images could be stitched with distortions. Apply some distortions to be able recognize
            # mask by distorted source sample.
            alb.GridDistortion(p=1, interpolation=cv2.INTER_AREA, border_mode=0),
            # * Crop a random part of the input.
            alb.RandomCrop(height=conf.img_wh, width=conf.img_wh, always_apply=True),
        ]
        if is_stub:
            return alb.Compose([alb.PadIfNeeded(min_height=conf.img_wh, min_width=conf.img_wh,
                                                always_apply=True, border_mode=0),
                                alb.RandomCrop(height=conf.img_wh, width=conf.img_wh, always_apply=True)])
        return alb.Compose(train_transform)

    def get_validation_augmentation(self, conf, is_stub=False):
        # Since batch-size in validation is 1, validation could be performed by whole crop-size.
        # To provide pos
        test_transform = [
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(always_apply=False, p=0.5),
            alb.PadIfNeeded(conf.img_wh_crop, conf.img_wh_crop, always_apply=True, border_mode=0),
            # alb.RandomCrop(height=conf.img_wh_crop, width=conf.img_wh_crop, always_apply=True),
        ]
        if is_stub:
            return alb.Compose([alb.PadIfNeeded(conf.img_wh_crop, conf.img_wh_crop, always_apply=True, border_mode=0)])
        return alb.Compose(test_transform)
