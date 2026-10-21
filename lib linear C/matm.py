import numpy as np
from linear_low import linear_low_level

__all__ = ['matmul']

def matmul(a: np.ndarray,
           b: np.ndarray):
    '''
    В текущей реализациии оба аргумента должны быть двумерными массивами.
    '''
    bias = np.zeros((len(b),), dtype=np.float32)
    c = np.zeros((len(a), len(b)), dtype=np.float32)
    linear_low_level(a,
                    b.T,
                    bias,
                    c,
                    len(a),
                    len(a[0]),
                    len(b))
    return c