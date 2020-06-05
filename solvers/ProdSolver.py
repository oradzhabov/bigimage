from . import ISolver
from .production import create_model_production, get_preprocessing_production


class ProdSolver(ISolver):
    def __init__(self):
        super(ProdSolver, self).__init__()

    def _create(self, conf, compile_model=True):
        if not conf.use_heightmap:
            raise Exception('Production utilizes height map. Enable it before in config before running')
        if conf.mppx != 0.25:
            raise Exception('Production utilizes 0.25 mppx. Setup it before in config before running')

        self.model, self.weights_path, self.metrics = create_model_production(conf, compile_model)

    def get_prep_getter(self):
        return get_preprocessing_production
