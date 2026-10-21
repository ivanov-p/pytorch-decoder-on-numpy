import numpy as np
from layerNorm_low import layer_norm
from base import Module

__all__ = ['LayerNorm']





class LayerNorm(Module):
    def __init__(self, normalized_shape: tuple[int] | int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = np.ones(self.normalized_shape, dtype=np.float32)
        if bias:
            self.bias = np.zeros(self.normalized_shape, dtype=np.float32)
        else:
            self.bias = None

    def __call__(self, input: np.ndarray) -> np.ndarray:
        b = self.bias if self.bias is not None else np.zeros(self.normalized_shape, dtype=np.float32)
        return layer_norm(input, self.normalized_shape, self.weight, b, self.eps)





'''
class LayerNorm:
    # Python-level
    
    def __init__(self, normalized_shape: tuple[int] | int):
        self.eps: float = 1e-5

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)

        self.weight = np.ones(self.normalized_shape)
        self.bias = np.zeros(self.normalized_shape)

    def __call__(self, input):
        return layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
'''