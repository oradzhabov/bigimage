import numpy as np
from abc import ABCMeta, abstractmethod


class ISolver(metaclass=ABCMeta):

    def __init__(self, conf):
        self.model = None
        self.weights_path = ''
        self.metrics = None
        self.conf = conf

    @staticmethod
    def round(pr_mask):
        return np.where(pr_mask > 0.5, 1.0, 0.0)

    @staticmethod
    def round_getter():
        return ISolver.round

    @abstractmethod
    def _create(self, compile_model=True):
        raise NotImplementedError

    @abstractmethod
    def get_prep_getter(self):
        raise NotImplementedError

    def get_post_getter(self):
        return ISolver.round_getter

    def build(self, compile_model=True):
        self._create(compile_model=compile_model)
        return self.model, self.weights_path, self.metrics, self.get_prep_getter()
