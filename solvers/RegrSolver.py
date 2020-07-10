from . import SegmSolver


class RegrSolver(SegmSolver):
    def __init__(self, conf):
        super(RegrSolver, self).__init__(conf)

        self.weights_path = 'runet.h5'
        self.activation = 'linear'

        self.total_loss = 'mse'
        self.metrics = ['mae']

    def post_predict(self, pr_result):
        return pr_result

    def get_contours(self, pr_mask):
        raise NotImplemented

    def monitoring_metric(self):
        return 'val_mean_absolute_error', 'min'
