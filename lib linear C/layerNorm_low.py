import numpy as np
from math import sqrt

__all__ = ['layer_norm']

'''
# C++ init?
def layer_norm(input, normalized_shape: tuple[int], weight, bias, eps: float = 1e-05):
    size: int = normalized_shape[0]
    out = np.empty(normalized_shape, dtype=np.float32)
    layer_norm_low_level(input, out, size, weight, bias, eps)
    return out

# Применяется C-тип float

def layer_norm_low_level(x, out, size: int, weight, bias, eps):
    mean: float = 0.0
    for elem in x:
        mean += elem

    mean = mean / size


    var: float = 0.0

    for i in range(size):
        var += (x[i] - mean)**2

    var = var / size



    for i in range(size):
        out[i] = ((x[i] - mean) / sqrt(var + eps)) * weight[i] + bias[i]


'''




def layer_norm(input: np.ndarray, normalized_shape: tuple[int], weight: np.ndarray, bias: np.ndarray, eps: float = 1e-05):
    # Корректная работа с N-мерными тензорами: нормализация по последним осям
    axis = tuple(range(-len(normalized_shape), 0))
    mean = np.mean(input, axis=axis, keepdims=True)
    var = np.var(input, axis=axis, keepdims=True)
    out = (input - mean) / np.sqrt(var + eps)
    out = out * weight + bias
    return out