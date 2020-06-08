from . import ISolver
from .production import create_model_production, get_preprocessing_production


class ProdSolver(ISolver):
    def __init__(self):
        super(ProdSolver, self).__init__()

        # Recofigure weights-file by synonym of file used in production
        # './mp_cntr_ff2_p1a_3_weights-ep495-loss0.00879-val_loss0.00905-val_acc0.99146-val_mean_iou0.95021.h5'
        self.weights_path = './mp_cntr_production_ff2.h5'

    def _create(self, conf, compile_model=True):
        if not conf.use_heightmap or conf.mppx != 0.25:
            raise Exception('Production utilizes height map and 0.25 mppx. Check it in config before running')

        self.model, self.metrics = create_model_production(conf, compile_model)
        self.model.load_weights(self.weights_path)

    def get_prep_getter(self):
        return get_preprocessing_production
