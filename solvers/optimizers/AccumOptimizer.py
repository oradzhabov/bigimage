from .. import get_submodules_from_kwargs


def get_accum_optimizer(**kwarguments):
    def dummy_dec(func):
        return func

    backend, layers, models, keras_utils, optimizers, legacy, callbacks = get_submodules_from_kwargs(kwarguments)

    legacy_get_updates_support = legacy.interfaces.legacy_get_updates_support if legacy is not None else dummy_dec

    # src: https://stackoverflow.com/a/56946898/5630599
    class AccumOptimizer(optimizers.Optimizer):
        """Optimizer
        #
            optimizer
            steps_per_update
        #
        Inheriting Optimizer class, wrapping the original optimizer
        to achieve a new corresponding optimizer of gradient accumulation.
        # Arguments
            optimizer: an instance of keras optimizer (supporting
                        all keras optimizers currently available);
            steps_per_update: the steps of gradient accumulation
        # Returns
            a new keras optimizer.
        """
        def __init__(self, optimizer, steps_per_update=1, **kwargs):
            super(AccumOptimizer, self).__init__(**kwargs)
            # super(AccumOptimizer, self).init(name='AccumOptimizer', **kwargs)
            self.optimizer = optimizer
            self.updates = []
            self.accum_grads = []
            with backend.name_scope(self.__class__.__name__):
                self.steps_per_update = steps_per_update
                self.iterations = backend.variable(0, dtype='int64', name='iterations')
                self.cond = backend.equal(self.iterations % self.steps_per_update, 0)

                # Depending on Keras version(KERAS 2.3.0 renamed param self.lr to self.learning_rate)
                if hasattr(self.optimizer, 'learning_rate'):
                    self.learning_rate = self.optimizer.learning_rate
                    self.optimizer.learning_rate = backend.switch(self.cond, self.optimizer.learning_rate, 0.)
                else:
                    self.lr = self.optimizer.lr
                    self.optimizer.lr = backend.switch(self.cond, self.optimizer.lr, 0.)

                for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                    if hasattr(self.optimizer, attr):
                        value = getattr(self.optimizer, attr)
                        setattr(self, attr, value)
                        setattr(self.optimizer, attr, backend.switch(self.cond, value, 1 - 1e-7))
                for attr in self.optimizer.get_config():
                    if not hasattr(self, attr):
                        value = getattr(self.optimizer, attr)
                        setattr(self, attr, value)

                # Cover the original get_gradients method with accumulative gradients.
                def get_gradients(loss, params):
                    return [ag / self.steps_per_update for ag in self.accum_grads]
                self.optimizer.get_gradients = get_gradients

        @legacy_get_updates_support
        def get_updates(self, loss, params):
            self.updates = [
                backend.update_add(self.iterations, 1),
                backend.update_add(self.optimizer.iterations, backend.cast(self.cond, 'int64')),
            ]
            # (gradient accumulation)
            self.accum_grads = [backend.zeros(backend.int_shape(p), dtype=backend.dtype(p)) for p in params]
            grads = self.get_gradients(loss, params)
            for g, ag in zip(grads, self.accum_grads):
                self.updates.append(backend.update(ag, backend.switch(self.cond, g, ag + g)))
            # optimizer (inheriting updates of original optimizer)
            self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
            self.weights.extend(self.optimizer.weights)
            return self.updates

        def get_config(self):
            iterations = backend.eval(self.iterations)
            backend.set_value(self.iterations, 0)
            config = self.optimizer.get_config()
            backend.set_value(self.iterations, iterations)
            return config

    return AccumOptimizer
