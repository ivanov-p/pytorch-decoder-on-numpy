# import torch

import numpy as np

__all__ = ["scaled_dot_product_attention"]


def scaled_dot_product_attention(query: np.ndarray, 
                                 key: np.ndarray, 
                                 value: np.ndarray,
                                 attn_mask: np.ndarray|None = None,
                                 dropout_p=0.0,
                                 is_causal: bool =False,
                                 scale: float|None = None,
                                 enable_gqa=False) -> np.ndarray:
    '''
    iv.functional.scaled_dot_product_attention()
    
    Функция вычисляет масштабированное скалярное внимание (scaled dot product attention)
    для тензоров запроса (query), ключа (key) и значения (value).
    При необходимости используется маска внимания (attn_mask).
    Если указана вероятность отсева больше 0.0, применяется dropout.
    Аргумент scale (масштабный коэффициент) можно передать только как именованный аргумент.
    '''

    
    L: int = 0
    S: int = 0
    query_size_m1: int = 0

    L, S = query.shape[-2], key.shape[-2]
    query_size_m1 = query.shape[-1]


    if 0 in (query.shape[-2], key.shape[-2], query.shape[-1]):
        # отрабатывание пустые query, key
        out_shape = query.shape[:-1] + (value.shape[-1],)
        out_dtype = np.result_type(query.dtype, value.dtype, np.float32)
        return np.zeros(out_shape, dtype=out_dtype)


    scale = 0.0 if scale is None else scale

    attn_bias = np.zeros((L, S), dtype=np.float32)

    if is_causal:
        assert attn_mask is None

        temp_mask = np.tril(np.ones((L, S), dtype=np.bool))
        attn_bias = np.ma.masked_array(attn_bias, mask=np.logical_not(temp_mask))
        attn_bias = np.ma.filled(attn_bias, fill_value=-1e9)

    if attn_mask is not None:
        if attn_mask.dtype == np.bool_ or attn_mask.dtype == bool:
            attn_bias = np.ma.masked_array(attn_bias, mask=attn_mask)
            attn_bias = np.ma.filled(attn_bias, fill_value=-1e9)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = np.repeat(key, query.shape[-3]//key.shape[-3], -3)
        value = np.repeat(value, query.shape[-3]//value.shape[-3], -3)

    return scaled_dot_product_attention_low_level(query,
                                                  key,
                                                  value,
                                                  #attn_mask,
                                                  dropout_p,
                                                  #is_causal,
                                                  scale,
                                                  #enable_gqa,
                                                  #L,
                                                  #S,
                                                  query_size_m1,
                                                  attn_bias,
                                                  #temp_mask
                                                  )






import math
def scaled_dot_product_attention_low_level(query,
                                           key,
                                           value,
                                           #attn_mask,
                                           dropout_p,
                                           #is_causal: bool,
                                           scale: float,
                                           #enable_gqa: bool,
                                           #L: int,
                                           #S: int,
                                           query_size_m1: int,
                                           attn_bias: np.ndarray,
                                           #temp_mask: np.ndarray
                                           ):

    if scale == 0.0:
        scale_factor = 1 / math.sqrt(query_size_m1)
    else:
        scale_factor = scale
    

    attn_weight = query @ like_torch_transpose(key, -2, -1) * scale_factor
    attn_weight += attn_bias

    attn_weight = softmax(attn_weight, dim=-1)

    if dropout_p > 0.0:
        mask = np.random.rand(*attn_weight.shape) > dropout_p
        attn_weight = np.where(mask, attn_weight / (1.0 - dropout_p), 0.0)

    return attn_weight @ value



def like_torch_transpose(x, d0, d1):
    axes = list(range(x.ndim))
    axes[d0], axes[d1] = axes[d1], axes[d0]
    return np.transpose(x, tuple(axes))


def softmax(x, dim=0):
    exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return exp_x / exp_x.sum(axis=dim, keepdims=True)
