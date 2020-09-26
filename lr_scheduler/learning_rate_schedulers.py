import os
import matplotlib.pyplot as plt
import numpy as np
import keras
import math
import logging

# src:
# https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/


class LearningRateDecay:
    """
    how to use:

    # Create linear scheduler with initial lr = 1e-1
    scheduler = PolynomialDecay(max_epochs=epochs, initAlpha=1e-1, power=1)

    # Fit model with following callbacks
    callbacks = [LearningRateScheduler(scheduler)]
    """
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):
    def __init__(self, init_alpha=0.01, factor=0.25, drop_every=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)


class PolynomialDecay(LearningRateDecay):
    def __init__(self, max_epochs=100, init_alpha=0.01, power=1.0):
        """
            max_epochs: The total number of epochs will be training for.
            init_alpha: The initial learning rate.
            power: The power/exponent of the polynomial.
        Note that if you set power=1.0
        then you have a linear learning rate decay.
        """
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.max_epochs = max_epochs
        self.init_alpha = init_alpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.init_alpha * decay
        # return the new learning rate
        return float(alpha)


class SnapshotEnsemble(keras.callbacks.Callback, LearningRateDecay):
    # std:
    # https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/

    def __init__(self, n_epochs, n_cycles, lrate_max, save_dir):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.save_dir = save_dir

    @staticmethod
    def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
        # calculate learning rate for epoch
        epochs_per_cycle = math.floor(n_epochs / n_cycles)
        cos_inner = (math.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return lrate_max / 2 * (math.cos(cos_inner) + 1)

    def __call__(self, epoch):
        return SnapshotEnsemble.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        lr = SnapshotEnsemble.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)

        # Set learning rate
        # Depending on Keras version(KERAS 2.3.0 renamed param self.lr to self.learning_rate)
        if hasattr(self.model.optimizer, 'learning_rate'):
            keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        else:
            keras.backend.set_value(self.model.optimizer.lr, lr)

        logging.info('SnapshotEnsemble LR changed to {}'.format(lr))

    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs=None):
        # check if we can save model
        epochs_per_cycle = math.floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # save model to file
            filename = "snapshot_weights_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            filename = os.path.join(self.save_dir, filename)
            # self.model.save(filename)
            self.model.save_weights(filename)
            logging.info('Saved weights %s, epoch %d' % (filename, epoch))
