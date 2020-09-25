# src: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np


class LearningRateDecay:
    """
    how to use:

    # Create linear scheduler with initial lr = 1e-1
    schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)

    # Fit model with following callbacks
    callbacks = [LearningRateScheduler(schedule)]
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
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)


class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        """
            maxEpochs: The total number of epochs we’ll be training for.
            initAlpha: The initial learning rate.
            power: The power/exponent of the polynomial.
        Note that if you set power=1.0
        then you have a linear learning rate decay.
        """
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power
    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        # return the new learning rate
        return float(alpha)


