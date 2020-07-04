import numpy as np
from itertools import chain
from math import log, floor
import keras
import keras.backend as K
import operator as op
from functools import reduce
from keras.models import Model
import nvidia_smi  # pip3 install nvidia-ml-py3


def get_free_gpu_mem(gpu_index):
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    # print("Total GPU memory:", info.total)
    # print("Free GPU memory:", info.free)
    # print("Used GPU memory:", info.used)

    nvidia_smi.nvmlShutdown()

    return info.free


# https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        #for s in l.output_shape:
        for s in l.get_output_at(-1).shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    # gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count

    return total_memory


def estimate_batch_size(gpu_index: int, model: Model, inpu_shape: tuple,
                        scale_by: float = 5.0,
                        precision: int = 4) -> int:

    """
    inp = keras.layers.Input(shape=(inpu_shape[0], inpu_shape[1], inpu_shape[2]))
    out = model(inp)
    model_wh = keras.models.Model(inp, out)
    model_wh.summary()
    """
    old_batch_input_shape = model._layers[0].batch_input_shape
    model._layers[0].batch_input_shape = (None, *inpu_shape)
    model_wh = keras.models.model_from_json(model.to_json())
    model_wh.summary()

    m1 = get_model_memory_usage(1, model_wh)
    m2 = get_model_memory_usage(2, model_wh)
    print('m1: ', m1)
    print('m2: ', m2)
    del model_wh
    K.clear_session()
    model._layers[0].batch_input_shape = old_batch_input_shape

    m21 = m2 - m1

    free_mem = get_free_gpu_mem(gpu_index)
    print('free_mem: ', free_mem)

    free_mem_1 = free_mem - m1

    batch_size = int(free_mem_1 // m21)

    batch_size = max(1, batch_size)

    print('batch_size: ', batch_size)
    exit(0)
    return batch_size
