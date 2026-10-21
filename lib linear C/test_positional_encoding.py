import torch
import numpy as np
import math
from PositionalEncoding import PositionalEncoding

# PyTorch-версия (из вашего gpt.py)
class TorchPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


def test_positional_encoding():
    d_model = 32
    max_len = 100
    batch_size = 2
    seq_len = 10

    # PyTorch
    pt_pe = TorchPositionalEncoding(d_model, max_len)
    pt_pe.eval()

    # NumPy
    np_pe = PositionalEncoding(d_model, max_len)

    # Одинаковые входные данные
    np.random.seed(42)
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    x_pt = torch.tensor(x_np)

    # Forward
    with torch.no_grad():
        out_pt = pt_pe(x_pt)
    out_np = np_pe(x_np)

    # Сравниваем не только выходы, но и сами буферы pe
    pe_diff = np.max(np.abs(pt_pe.pe.numpy() - np_pe.pe))
    out_diff = np.max(np.abs(out_pt.numpy() - out_np))

    print(f"PE Buffer Max Diff: {pe_diff:.2e}")
    print(f"Output Max Diff:    {out_diff:.2e}")
    print(f"Input shape:  {x_np.shape}")
    print(f"Output shape: {out_np.shape}")
    print(f"PE shape:     {np_pe.pe.shape}")

    # Для float32 и тригонометрических функций допуск 1e-5 является стандартом
    if out_diff < 1e-5 and pe_diff < 1e-5:
        print("POSITIONAL ENCODING OK (разница в пределах машинного эпсилон для float32)")
    else:
        print("POSITIONAL ENCODING IS BROKEN")


if __name__ == "__main__":
    test_positional_encoding()