import functools
import logging

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None
_KERAS_OPTIMIZERS = None
_KERAS_LEGACY = None
_KERAS_CALLBACKS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    optimizers = kwargs.get('optimizers', _KERAS_OPTIMIZERS)
    legacy = kwargs.get('legacy', _KERAS_LEGACY)
    callbacks = kwargs.get('callbacks', _KERAS_CALLBACKS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils', 'optimizers', 'legacy', 'callbacks']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils, optimizers, legacy, callbacks


def inject_keras_modules(func):
    import keras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        kwargs['optimizers'] = keras.optimizers
        kwargs['legacy'] = keras.legacy
        kwargs['callbacks'] = keras.callbacks

        try:
            # Cross-pipeline strategy allows to not crash if not required packages was not installed
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(e)
            return None

    return wrapper


def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        kwargs['optimizers'] = tfkeras.optimizers
        kwargs['legacy'] = None  # tf.keras has no module legacy
        kwargs['callbacks'] = tfkeras.callbacks

        try:
            # Cross-pipeline strategy allows to not crash if not required packages was not installed
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(e)
            return None

    return wrapper


def init_keras_custom_objects():
    import keras
    from .solvers.optimizers import AccumOptimizer

    custom_objects = {
        'AccumOptimizer': inject_keras_modules(AccumOptimizer.tf1)()
    }

    try:
        keras.utils.generic_utils.get_custom_objects().update(custom_objects)
    except AttributeError:
        keras.utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    from .solvers.optimizers import AccumOptimizer

    custom_objects = {
        'AccumOptimizer': inject_tfkeras_modules(AccumOptimizer.tf2)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)
