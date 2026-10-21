import ctypes
import numpy as np
import os

# Определяем путь к DLL относительно текущего файла
_dll_path = os.path.join(os.path.dirname(__file__), "linear_lib.dll")
_lib = ctypes.CDLL(_dll_path)

_lib.linear_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # weight
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # batch
    ctypes.c_int,                    # in_features
    ctypes.c_int,                    # out_features
    ctypes.c_int                     # num_threads
]
_lib.linear_forward.restype = None

def linear(input: np.ndarray, weight: np.ndarray, bias: np.ndarray = None, num_threads: int = 4) -> np.ndarray:
    """
    Drop-in замена для linear из multi_head_attention_forward_low.py
    Вычисляет: input @ weight.T + bias
    """
    original_shape = input.shape
    
    # Приводим к 2D для C-функции (математически эквивалентно transpose_last_two_axes для matmul)
    if input.ndim != 2:
        input_2d = input.reshape(-1, input.shape[-1])
    else:
        input_2d = input

    batch = input_2d.shape[0]
    in_features = input_2d.shape[1]
    out_features = weight.shape[0] # weight имеет форму (out_features, in_features)

    # Выделяем память для результата
    output = np.empty((batch, out_features), dtype=np.float32)

    # Обработка bias (None превращается в NULL pointer)
    bias_ptr = ctypes.cast(bias.ctypes.data, ctypes.POINTER(ctypes.c_float)) if bias is not None else None

    # Вызов C-функции
    _lib.linear_forward(
        ctypes.cast(input_2d.ctypes.data, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(weight.ctypes.data, ctypes.POINTER(ctypes.c_float)),
        bias_ptr,
        ctypes.cast(output.ctypes.data, ctypes.POINTER(ctypes.c_float)),
        batch,
        in_features,
        out_features,
        num_threads
    )

    # Возвращаем исходную размерность, если она была N-мерной
    if input.ndim != 2:
        return output.reshape(*original_shape[:-1], out_features)
    
    return output