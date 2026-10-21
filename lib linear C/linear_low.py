import numpy as np

__all__ = ['linear']


# C++ init?
def linear(x, weights, b):
    batch = len(x)
    in_feat = len(x[0])
    out_feat = len(weights)
    result = np.empty((batch, out_feat), dtype=np.float32)
    linear_low_level(x,
                     weights,
                     b,
                     result,
                     batch,
                     in_feat,
                     out_feat)
    return result




def linear_low_level(
        x,
        weights,
        b,
        result,
        batch: int,
        in_feat: int,
        out_feat: int):
    
    for i in range(batch):
        for j in range(out_feat):
            s: float = 0.0
            for k in range(in_feat):
                s += x[i][k] * weights[j][k]
            s += b[j]

            result[i][j] = s