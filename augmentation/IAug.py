from abc import abstractmethod
import sys
sys.path.append(sys.path[0] + "/..")
from kutils.JSONEncoder import JSONEncoder


class IAug(JSONEncoder):

    @abstractmethod
    def get_training_augmentation(self, conf, is_stub=False):
        raise NotImplementedError

    @abstractmethod
    def get_validation_augmentation(self, conf, is_stub=False):
        raise NotImplementedError
