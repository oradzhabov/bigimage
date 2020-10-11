from matplotlib import pyplot as plt
import math
import logging
# from keras.callbacks import LambdaCallback
# import keras.backend as K
import numpy as np

from .. import get_submodules_from_kwargs
from .. import bin_keras


# https://www.machinecurve.com/index.php/2020/02/20/finding-optimal-learning-rates-with-the-learning-rate-range-test/
def tf1(**kwarguments):
    backend, layers, models, keras_utils, optimizers, legacy, callbacks = get_submodules_from_kwargs(kwarguments)

    class LRFinder:
        """
        Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
        See for details:
        https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
        """

        def __init__(self, model):
            self.model = model
            self.losses = []
            self.lrs = []
            self.best_loss = 1e9
            self.lr_mult = None
            self.steps_per_update = 1
            if isinstance(model.optimizer, bin_keras.AccumGradOptimizer):
                self.steps_per_update = model.optimizer.steps_per_update

        def on_batch_end(self, batch, logs):
            # Log the loss
            loss = logs['loss']

            # Process case when optimizer assumes accumulating gradients
            if batch % self.steps_per_update != 0:
                self.losses[-1] += loss
                return

            # Average previous results
            if len(self.losses) > 0:
                self.losses[-1] /= float(self.steps_per_update)

            # Log the learning rate
            lr = backend.get_value(self.model.optimizer.lr)
            self.lrs.append(lr)

            self.losses.append(loss)

            # Check whether the loss got too large or NaN
            if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 400):
                logging.info('Process interrupted with loss {}'.format(loss))
                self.model.stop_training = True
                return

            if loss < self.best_loss:
                self.best_loss = loss

            # Increase the learning rate for the next batch
            lr *= self.lr_mult
            backend.set_value(self.model.optimizer.lr, lr)

        def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1, **kw_fit):
            # If x_train contains data for multiple inputs, use length of the first input.
            # Assumption: the first element in the list is single input; NOT a list of inputs.
            n = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

            # Compute number of batches and LR multiplier
            num_batches = epochs * n / batch_size
            self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
            # Save weights into a file
            initial_weights = self.model.get_weights()

            # Remember the original learning rate
            original_lr = backend.get_value(self.model.optimizer.lr)

            # Set the initial learning rate
            backend.set_value(self.model.optimizer.lr, start_lr)

            callback = callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

            self.model.fit(x_train, y_train,
                           batch_size=batch_size, epochs=epochs,
                           callbacks=[callback],
                           **kw_fit)

            # Restore the weights to the state before model fitting
            self.model.set_weights(initial_weights)

            # Restore the original learning rate
            backend.set_value(self.model.optimizer.lr, original_lr)

        def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
            if steps_per_epoch is None:
                try:
                    steps_per_epoch = len(generator)
                except (ValueError, NotImplementedError) as e:
                    raise e('`steps_per_epoch=None` is only valid for a'
                            ' generator based on the '
                            '`keras.utils.Sequence`'
                            ' class. Please specify `steps_per_epoch` '
                            'or use the `keras.utils.Sequence` class.')
            self.lr_mult = (float(end_lr) / float(start_lr)) ** (
                        float(1) / float(epochs * steps_per_epoch / self.steps_per_update))

            # Save weights into a file
            # initial_weights = self.model.get_weights()

            # Remember the original learning rate
            # original_lr = backend.get_value(self.model.optimizer.lr)

            # Set the initial learning rate
            backend.set_value(self.model.optimizer.lr, start_lr)

            callback = callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

            self.model.fit_generator(generator=generator,
                                     epochs=epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     callbacks=[callback],
                                     **kw_fit)

            # Restore the weights to the state before model fitting
            # self.model.set_weights(initial_weights)

            # Restore the original learning rate
            # backend.set_value(self.model.optimizer.lr, original_lr)

        def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
            """
            Plots the loss.
            Parameters:
                n_skip_beginning - number of batches to skip on the left.
                n_skip_end - number of batches to skip on the right.
            """
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
            plt.xscale(x_scale)
            plt.show()

        def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
            """
            Plots rate of change of the loss function.
            Parameters:
                sma - number of batches for simple moving average to smooth out the curve.
                n_skip_beginning - number of batches to skip on the left.
                n_skip_end - number of batches to skip on the right.
                y_lim - limits for the y axis.
            """
            derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
            lrs = self.lrs[n_skip_beginning:-n_skip_end]
            plt.ylabel("rate of loss change")
            plt.xlabel("learning rate (log scale)")
            plt.plot(lrs, derivatives)
            plt.xscale('log')
            plt.ylim(y_lim)
            plt.show()

        def get_derivatives(self, sma):
            assert sma >= 1
            derivatives = [0] * sma
            for i in range(sma, len(self.lrs)):
                derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
            return derivatives

        def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
            derivatives = self.get_derivatives(sma)
            # best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end]) # ros: I've modified to following:
            best_der_idx = np.argmax(derivatives[n_skip_beginning:-n_skip_end])[0]
            return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]

    return LRFinder


def tf2(**kwarguments):
    backend, layers, models, keras_utils, optimizers, legacy, callbacks = get_submodules_from_kwargs(kwarguments)

    class LRFinder:
        """
        Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
        See for details:
        https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
        """

        def __init__(self, model):
            self.model = model
            self.losses = []
            self.lrs = []
            self.best_loss = 1e9
            self.lr_mult = None
            self.steps_per_update = 1
            if isinstance(model.optimizer, bin_keras.AccumGradOptimizer):
                self.steps_per_update = model.optimizer.steps_per_update

        def on_batch_end(self, batch, logs):
            # Log the loss
            loss = logs['loss']

            # Process case when optimizer assumes accumulating gradients
            if batch % self.steps_per_update != 0:
                self.losses[-1] += loss
                return

            # Average previous results
            if len(self.losses) > 0:
                self.losses[-1] /= float(self.steps_per_update)

            # Log the learning rate
            lr = backend.get_value(self.model.optimizer.learning_rate)
            self.lrs.append(lr)

            self.losses.append(loss)

            # Check whether the loss got too large or NaN
            if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 400):
                logging.info('Process interrupted with loss {}'.format(loss))
                self.model.stop_training = True
                return

            if loss < self.best_loss:
                self.best_loss = loss

            # Increase the learning rate for the next batch
            lr *= self.lr_mult
            backend.set_value(self.model.optimizer.learning_rate, lr)

        def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
            # If x_train contains data for multiple inputs, use length of the first input.
            # Assumption: the first element in the list is single input; NOT a list of inputs.
            n = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

            # Compute number of batches and LR multiplier
            num_batches = epochs * n / batch_size
            self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
            # Save weights into a file
            self.model.save_weights('tmp.h5')

            # Remember the original learning rate
            original_lr = backend.get_value(self.model.optimizer.learning_rate)

            # Set the initial learning rate
            backend.set_value(self.model.optimizer.learning_rate, start_lr)

            callback = callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

            self.model.fit(x_train, y_train,
                           batch_size=batch_size, epochs=epochs,
                           callbacks=[callback])

            # Restore the weights to the state before model fitting
            self.model.load_weights('tmp.h5')

            # Restore the original learning rate
            backend.set_value(self.model.optimizer.learning_rate, original_lr)

        def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
            if steps_per_epoch is None:
                try:
                    steps_per_epoch = len(generator)
                except (ValueError, NotImplementedError) as e:
                    raise e('`steps_per_epoch=None` is only valid for a'
                            ' generator based on the '
                            '`keras.utils.Sequence`'
                            ' class. Please specify `steps_per_epoch` '
                            'or use the `keras.utils.Sequence` class.')
            self.lr_mult = (float(end_lr) / float(start_lr)) ** (
                    float(1) / float(epochs * steps_per_epoch / self.steps_per_update))

            # Save weights into a file
            self.model.save_weights('tmp.h5')

            # Remember the original learning rate
            original_lr = backend.get_value(self.model.optimizer.learning_rate)

            # Set the initial learning rate
            backend.set_value(self.model.optimizer.learning_rate, start_lr)

            callback = callbacks.LambdaCallback(on_batch_end=lambda batch,
                                                logs: self.on_batch_end(batch, logs))

            self.model.fit_generator(generator=generator,
                                     epochs=epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     callbacks=[callback],
                                     **kw_fit)

            # Restore the weights to the state before model fitting
            self.model.load_weights('tmp.h5')

            # Restore the original learning rate
            backend.set_value(self.model.optimizer.learning_rate, original_lr)

        def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
            """
            Plots the loss.
            Parameters:
                n_skip_beginning - number of batches to skip on the left.
                n_skip_end - number of batches to skip on the right.
            """
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
            plt.xscale(x_scale)
            plt.show()

        def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
            """
            Plots rate of change of the loss function.
            Parameters:
                sma - number of batches for simple moving average to smooth out the curve.
                n_skip_beginning - number of batches to skip on the left.
                n_skip_end - number of batches to skip on the right.
                y_lim - limits for the y axis.
            """
            derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
            lrs = self.lrs[n_skip_beginning:-n_skip_end]
            plt.ylabel("rate of loss change")
            plt.xlabel("learning rate (log scale)")
            plt.plot(lrs, derivatives)
            plt.xscale('log')
            plt.ylim(y_lim)
            plt.show()

        def get_derivatives(self, sma):
            assert sma >= 1
            derivatives = [0] * sma
            for i in range(sma, len(self.lrs)):
                derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
            return derivatives

        def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
            derivatives = self.get_derivatives(sma)
            best_der_idx = np.argmax(derivatives[n_skip_beginning:-n_skip_end])[0]
            return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]

    return LRFinder
