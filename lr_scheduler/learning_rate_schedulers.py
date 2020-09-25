# src: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np


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
