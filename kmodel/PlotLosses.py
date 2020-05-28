from keras.callbacks import Callback
from IPython.display import clear_output

# Runtime custom callbacks
# https://github.com/deepsense-ai/intel-ai-webinar-neural-networks/blob/master/live_loss_plot.py
# Fixed code to enable non-flat loss plots on keras model.fit_generator()
# import matplotlib
# matplotlib.use('Agg')  # Disable TclTk because it sometime crash training!

import matplotlib.pyplot as plt


def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x


# Actually it could crash training. To avoid crash add 'matplotlib.use('Agg')' at the start
class PlotLosses(Callback):
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
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center left')
        
        plt.tight_layout()
        fig.savefig(self.imgfile)
        # draw the plot. Actually it could crash training. To avoid crash add 'matplotlib.use('Agg')' at the start
        plt.draw()
        plt.pause(2)  # show it for N seconds
        plt.close(fig)
