from abc import ABCMeta, abstractmethod


class ISolver(metaclass=ABCMeta):

    def __init__(self):
        self.model = None
        self.weights_path = ''
        self.metrics = None

    @abstractmethod
    def _create(self, conf, compile_model=True):
        raise NotImplementedError

    @abstractmethod
    def get_prep_getter(self):
        raise NotImplementedError

    def build(self, conf, compile_model=True):
        self._create(conf, compile_model=compile_model)
        return self.model, self.weights_path, self.metrics, self.get_prep_getter()
