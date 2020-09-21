import sys
import os
import cv2
import numpy as np
import random
from abc import ABCMeta, abstractmethod
sys.path.append(sys.path[0] + "/..")
from kmodel.data import read_image


class ISolver(metaclass=ABCMeta):

    def __init__(self, conf):
        self.model = None
        self.weights_path = ''
        self.metrics = None
        self.conf = conf
        self.activation = None
        self.total_loss = None
        #
        random.seed(self.conf.seed)
        np.random.seed(self.conf.seed)

    def signature(self):
        wpath_name = os.path.splitext(os.path.basename(self.weights_path))[0]
        signature = '{}_{}_mppx{:.2f}_wh{}_rgb{}_{}cls_{}'.format(wpath_name,
                                                                  self.conf.backbone,
                                                                  self.conf.mppx,
                                                                  self.conf.img_wh,
                                                                  'a' if self.conf.use_heightmap else '',
                                                                  self.conf.cls_nb,
                                                                  self.conf.data_subset)
        return signature

    def _get_avg_prob_field(self, pr_mask_list):
        def get_ndarray(pr_mask_item):
            result = read_image(pr_mask_item['img']) if isinstance(pr_mask_item['img'], str) else pr_mask_item['img']

            # Sometime(e.g. 2-channels output) data stored with bigger channels num. Trunc used channels.
            if len(result.shape) > 2:
                result = result[..., :self.conf.cls_nb]

            result = result.squeeze()
            if pr_mask_item['img_dtype'] != np.uint8:
                # if pr_mask_item['img_dtype'] in [np.float16, np.float32, np.float64]:
                if pr_mask_item['img_dtype'] in (np.float16, np.float32, np.float64):
                    result = result.astype(pr_mask_item['img_dtype']) / 255.0
                elif pr_mask_item['img_dtype'] == np.bool:
                    result = result.astype(pr_mask_item['img_dtype'])
                else:
                    raise NotImplementedError
            return result

        # To speed up the performance, returns result if it is single
        if len(pr_mask_list) == 1:
            return get_ndarray(pr_mask_list[0])

        # Get average probability field
        pr_mask = None
        pr_mask_dtype = None
        for item in pr_mask_list:
            if pr_mask is None:
                pr_mask = get_ndarray(item).copy()
                # Predict data overflow
                pr_mask_dtype = pr_mask.dtype
                if pr_mask_dtype is np.uint8 or pr_mask_dtype is np.int8:
                    pr_mask = pr_mask.astype(np.float16)
            else:
                pr_mask += get_ndarray(item)
        if pr_mask is not None:
            pr_mask /= len(pr_mask_list)
            pr_mask = pr_mask.astype(pr_mask_dtype)
        return pr_mask

    @abstractmethod
    def _create_metrics(self, **kwargs):
        """ This method responds to creating parameters self.metrics and self.total_loss """
        raise NotImplementedError

    @abstractmethod
    def _create(self, compile_model=True, **kwargs):
        raise NotImplementedError

    def post_predict(self, pr_result):
        raise pr_result

    @abstractmethod
    def get_prep_getter(self):
        raise NotImplementedError

    @abstractmethod
    def get_contours(self, pr_mask_list):
        raise NotImplementedError

    @abstractmethod
    def monitoring_metric(self):
        raise NotImplementedError

    def build(self, compile_model=True, **kwargs):
        self._create_metrics(**kwargs)
        self._create(compile_model, **kwargs)
        return self.model, self.weights_path, self.metrics
