from .. import get_submodules_from_kwargs


def get_solver(**kwarguments):
    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    from .SegmSolver import get_solver as get_segm_solver

    SegmSolver = get_segm_solver(**kwarguments)

    class RegrSolver(SegmSolver):
        def __init__(self, conf):
            super(RegrSolver, self).__init__(conf)

            self.weights_path = 'runet.h5'
            self.activation = 'linear'

        def _create_metrics(self, **kwargs):
            self.total_loss = 'mse'
            self.metrics = ['mae']

        def post_predict(self, pr_result):
            return pr_result

        def get_contours(self, pr_mask_list):
            raise NotImplemented

        def monitoring_metric(self):
            return 'val_mean_absolute_error', 'min'

    return RegrSolver
