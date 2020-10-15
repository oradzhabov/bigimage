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
            alb.GaussNoise(var_limit=20, mean=0, p=1),
            alb.OneOf(
                [
                    alb.Blur(blur_limit=3, p=1),
                    alb.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            #
            # scale_limit ((float, float) or float) – scaling factor range. If scale_limit is a single float value,
            # the range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).
            # rotate_limit ((int, int) or int) – rotation range. If rotate_limit is a single int value,
            # the range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
            # shift_limit ((float, float) or float) – shift factor range for both height and width. If shift_limit is
            # a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper
            # bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
            # alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, shift_limit=0.5, p=1, border_mode=0),
            alb.ShiftScaleRotate(scale_limit=0.1, rotate_limit=90, p=1, border_mode=0,
                                 interpolation=cv2.INTER_LANCZOS4),

            # Pad side of the image / max if side is less than desired number.
            alb.PadIfNeeded(min_height=conf.img_wh, min_width=conf.img_wh, always_apply=True, border_mode=0),

            RandomBrightnessContrast2(brightness_limit=(-0.2, 0.2), contrast_limit=0.2, p=1),
            RandomGamma2(p=1),



            # alb.OneOf(
            #    [
            #        alb.CLAHE(p=1),
            #        alb.RandomBrightness(p=1),
            #        alb.RandomGamma(p=1),
            #    ],
            #    p=0.9,
            # ),

            # alb.OneOf(
            #    [
            #        alb.IAASharpen(p=1),
            #        alb.Blur(blur_limit=3, p=1),
            #        alb.MotionBlur(blur_limit=3, p=1),
            #    ],
            #    p=0.9,
            # ),
            HueSaturationValue2(hue_shift_limit=15, sat_shift_limit=(-20, 10), val_shift_limit=0, p=1.0),
            ## HueSaturationValue2(hue_shift_limit=90, sat_shift_limit=(-180, 10), val_shift_limit=0, p=1.0),
            # alb.OneOf(
            #    [
            #        alb.RandomContrast(p=1),
            #        alb.HueSaturationValue(p=1),
            #    ],
            #    p=0.9,
            # ),
            # alb.Lambda(mask=round_clip_0_1),
            # Crop a random part of the input.
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
