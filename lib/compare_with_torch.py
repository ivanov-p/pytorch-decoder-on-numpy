import torch
import numpy as np


from TransformerDecoderLayer import TransformerDecoderLayer as NumpyTransformerDecoderLayer
from MultiheadAttention import MultiheadAttention
from Linear import Linear
from LayerNorm import LayerNorm

def copy_weights_from_torch(pt_layer, np_layer):
    """Копирует веса из PyTorch слоя в NumPy слой"""
    
    # 1. Self Attention
    np_layer.self_attn.in_proj_weight = pt_layer.self_attn.in_proj_weight.detach().numpy()
    np_layer.self_attn.in_proj_bias = pt_layer.self_attn.in_proj_bias.detach().numpy()
    np_layer.self_attn.out_proj_weight = pt_layer.self_attn.out_proj.weight.detach().numpy()
    np_layer.self_attn.out_proj_bias = pt_layer.self_attn.out_proj.bias.detach().numpy()

    # 2. Multi-Head Attention (Cross Attention)
    np_layer.multihead_attn.in_proj_weight = pt_layer.multihead_attn.in_proj_weight.detach().numpy()
    np_layer.multihead_attn.in_proj_bias = pt_layer.multihead_attn.in_proj_bias.detach().numpy()
    np_layer.multihead_attn.out_proj_weight = pt_layer.multihead_attn.out_proj.weight.detach().numpy()
    np_layer.multihead_attn.out_proj_bias = pt_layer.multihead_attn.out_proj.bias.detach().numpy()

    # 3. Feed Forward
    np_layer.linear1.weights = pt_layer.linear1.weight.detach().numpy()
    np_layer.linear1.bias = pt_layer.linear1.bias.detach().numpy()
    np_layer.linear2.weights = pt_layer.linear2.weight.detach().numpy()
    np_layer.linear2.bias = pt_layer.linear2.bias.detach().numpy()

    # 4. Layer Norms
    np_layer.norm1.weight = pt_layer.norm1.weight.detach().numpy()
    np_layer.norm1.bias = pt_layer.norm1.bias.detach().numpy()
    
    np_layer.norm2.weight = pt_layer.norm2.weight.detach().numpy()
    np_layer.norm2.bias = pt_layer.norm2.bias.detach().numpy()
    
    np_layer.norm3.weight = pt_layer.norm3.weight.detach().numpy()
    np_layer.norm3.bias = pt_layer.norm3.bias.detach().numpy()


def main():
    # Параметры для теста (делаем небольшими для скорости)
    d_model = 64
    nhead = 4
    dim_feedforward = 128
    seq_len_tgt = 5
    seq_len_mem = 7
    batch_size = 2

    print("Инициализация моделей...")
    # PyTorch модель
    pt_layer = torch.nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
        batch_first=False, dropout=0.0 # dropout=0.0 для детерминированного сравнения!
    )
    pt_layer.eval() # Важно: отключаем dropout и train-режимы

    # Наша NumPy модель
    np_layer = NumpyTransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
        batch_first=False, dropout=0.0
    )
    np_layer.eval()

    # Копируем веса из PyTorch в NumPy
    copy_weights_from_torch(pt_layer, np_layer)
    print("Веса успешно скопированы!")

    # Генерируем одинаковые входные данные
    np.random.seed(42)
    torch.manual_seed(42)
    
    tgt_np = np.random.randn(seq_len_tgt, batch_size, d_model).astype(np.float32)
    mem_np = np.random.randn(seq_len_mem, batch_size, d_model).astype(np.float32)

    tgt_pt = torch.tensor(tgt_np)
    mem_pt = torch.tensor(mem_np)

    # Создаем простую causal mask (верхний треугольник True)
    # True в mask означает "игнорировать" (заполнить -inf)
    mask_np = np.triu(np.ones((seq_len_tgt, seq_len_tgt), dtype=bool), k=1)
    mask_pt = torch.tensor(mask_np)

    print("Запуск прямого прохода (Forward pass)...")
    
    # 1. Считаем в PyTorch
    with torch.no_grad():
        out_pt = pt_layer(tgt_pt, mem_pt, tgt_mask=mask_pt)
    
    # 2. Считаем в нашей NumPy реализации
    out_np = np_layer(tgt_np, mem_np, tgt_mask=mask_np)

    # 3. Сравниваем результаты
    out_pt_np = out_pt.numpy()
    
    # allclose проверяет, что все элементы отличаются не более чем на atol + rtol * abs(b)
    is_close = np.allclose(out_np, out_pt_np, atol=1e-5, rtol=1e-5)
    
    # Вычислим максимальную разницу для наглядности
    max_diff = np.max(np.abs(out_np - out_pt_np))

    print("\n" + "="*50)
    if is_close:
        print("УСПЕХ! Результаты численно идентичны!")
        print(f"Максимальная разница между PyTorch и NumPy: {max_diff:.2e}")
    else:
        print("РАСХОЖДЕНИЕ! Результаты отличаются.")
        print(f"Максимальная разница: {max_diff:.2e}")
        print("Первые 5 элементов NumPy :", out_np.flatten()[:5])
        print("Первые 5 элементов PyTorch:", out_pt_np.flatten()[:5])
    print("="*50)

if __name__ == "__main__":
    main()