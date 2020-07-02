from abc import ABCMeta, abstractmethod


class IAug(metaclass=ABCMeta):

    @abstractmethod
    def get_training_augmentation(self, conf, is_stub=False):
        raise NotImplementedError

    @abstractmethod
    def get_validation_augmentation(self, conf, is_stub=False):
        raise NotImplementedError
