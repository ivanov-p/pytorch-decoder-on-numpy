import numpy as np

__all__ = ['relu', 'gelu', '_get_activation_fn']

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def gelu(x: np.ndarray) -> np.ndarray:
    # Приближенная формула GELU, используемая в PyTorch
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def _get_activation_fn(activation: str):
    if activation == "relu":
        return relu
    elif activation == "gelu":
        return gelu
    else:
        raise ValueError(f"Activation function {activation} not supported")
