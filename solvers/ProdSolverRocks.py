from . import ISolver
from .production import create_model_production_rock, get_preprocessing_production_rock, get_postprocessing_prod_rock


class ProdSolverRocks(ISolver):
    def __init__(self, conf):
        super(ProdSolverRocks, self).__init__(conf)

        # Reconfigure weights-file by synonym of file used in production
        # './weights-ep203-loss0.73074513-val_loss0.76001111-val_acc0.86737848-val_mean_iou0.60081122.h5'
        self.weights_path = './rocks_cntr_production.h5'

    def _create(self, compile_model=True, **kwargs):
        assert self.conf.use_heightmap, 'Production utilize height map. Check it in config before running'
        # assert self.conf.mppx == 0.25, 'Production utilizes 0.25 mppx. Check it in config before running'
        assert self.conf.classes is None, 'Production utilizes None for classes. Check it in config before running'

        self.model, self.metrics = create_model_production_rock(self.conf, compile_model)
        self.model.load_weights(self.weights_path)

    def get_prep_getter(self):
        return get_preprocessing_production_rock

    def get_post_getter(self):
        return get_postprocessing_prod_rock

    def monitoring_metric(self):
        return 'val_f1-score', 'max'
