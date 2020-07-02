from abc import ABCMeta, abstractmethod


class IDataProvider(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def get_fname(self, i):
        raise NotImplementedError

    @abstractmethod
    def get_color(self, class_ind):
        raise NotImplementedError

    @abstractmethod
    def show(self, i):
        raise NotImplementedError

    @abstractmethod
    def show_predicted(self, show_random_items_nb):
        raise NotImplementedError
