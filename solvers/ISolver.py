import os
import numpy as np
import random
from abc import ABCMeta, abstractmethod


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

    @abstractmethod
    def _create(self, compile_model=True, **kwargs):
        raise NotImplementedError

    def post_predict(self, pr_result):
        raise pr_result

    @abstractmethod
    def get_prep_getter(self):
        raise NotImplementedError

    @abstractmethod
    def get_contours(self, pr_mask):
        raise NotImplementedError

    @abstractmethod
    def monitoring_metric(self):
        raise NotImplementedError

    def build(self, compile_model=True, **kwargs):
        self._create(compile_model, **kwargs)
        return self.model, self.weights_path, self.metrics
