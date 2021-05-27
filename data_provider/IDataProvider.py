from abc import abstractmethod
from ..kutils.JSONEncoder import JSONEncoder


class IDataProvider(JSONEncoder):

    def __init__(self):
        self.conf = None

    @abstractmethod
    def __getitem__(self, i):
        raise NotImplementedError

    @abstractmethod
    def get_scaled_image(self, i, sc_factor):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def get_src_data(self):
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
