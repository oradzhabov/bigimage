def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Loss"}
    if x in translations:
        return translations[x]
    else:
        return x


def get_plot_losses(**kwarguments):
    from IPython.display import clear_output
    from .. import get_submodules_from_kwargs

    # Runtime custom callbacks
    # https://github.com/deepsense-ai/intel-ai-webinar-neural-networks/blob/master/live_loss_plot.py
    # Fixed code to enable non-flat loss plots on keras model.fit_generator()
    # import matplotlib
    # matplotlib.use('Agg')  # Disable TclTk because it sometime crash training!

    import matplotlib.pyplot as plt

    _backend, _layers, _models, _keras_utils, _optimizers, _legacy, _callbacks = get_submodules_from_kwargs(kwarguments)

    # Actually it could crash training. To avoid crash add 'matplotlib.use('Agg')' at the start
    class PlotLosses(_callbacks.Callback):
        def __init__(self, imgfile, figsize=None):
            super(PlotLosses, self).__init__()
            self.figsize = figsize
            self.imgfile = imgfile
            self.base_metrics = []
            self.logs = []

        def on_train_begin(self, logs=None):
            self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
            self.logs = []

        def on_epoch_end(self, epoch, logs=None):
            self.logs.append(logs.copy())

            clear_output(wait=True)
            fig = plt.figure(figsize=self.figsize)

            for metric_id, metric in enumerate(self.base_metrics):
                plt.subplot(len(self.base_metrics), 1, metric_id + 1)

                metric_extr_val_value = None
                plt.plot(range(1, len(self.logs) + 1),
                         [log[metric] for log in self.logs],
                         label="training")
                if self.params['do_validation']:
                    val_log_list = [log['val_' + metric] for log in self.logs]
                    plt.plot(range(1, len(self.logs) + 1),
                             val_log_list,
                             label="validation")
                    metric_extr_val_value = ' (val_min {:.3f})'.format(min(val_log_list)) if metric == 'loss' else \
                        ' (val_max {:.3f})'.format(max(val_log_list))
                plt.title(translate_metric(metric) + metric_extr_val_value)
                plt.xlabel('epoch')
                plt.legend(loc='center left')

            plt.tight_layout()
            fig.savefig(self.imgfile)
            # draw the plot. Actually it could crash training. To avoid crash add 'matplotlib.use('Agg')' at the start
            plt.draw()
            plt.pause(0.5)  # show it for N seconds
            plt.close(fig)

    return PlotLosses
