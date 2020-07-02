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
