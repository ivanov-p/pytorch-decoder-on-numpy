import torch
import numpy as np
import copy

from TransformerDecoder import TransformerDecoder
from TransformerDecoderLayer import TransformerDecoderLayer
from LayerNorm import LayerNorm


def copy_decoder_weights(pt_decoder, np_decoder):
    """Копирует веса из PyTorch TransformerDecoder в NumPy-версию."""
    for i in range(len(np_decoder.layers)):
        pt_layer = pt_decoder.layers[i]
        np_layer = np_decoder.layers[i]

        # Self Attention
        np_layer.self_attn.in_proj_weight = pt_layer.self_attn.in_proj_weight.detach().numpy()
        np_layer.self_attn.in_proj_bias = pt_layer.self_attn.in_proj_bias.detach().numpy()
        np_layer.self_attn.out_proj_weight = pt_layer.self_attn.out_proj.weight.detach().numpy()
        np_layer.self_attn.out_proj_bias = pt_layer.self_attn.out_proj.bias.detach().numpy()

        # Multi-Head Attention (Cross Attention)
        np_layer.multihead_attn.in_proj_weight = pt_layer.multihead_attn.in_proj_weight.detach().numpy()
        np_layer.multihead_attn.in_proj_bias = pt_layer.multihead_attn.in_proj_bias.detach().numpy()
        np_layer.multihead_attn.out_proj_weight = pt_layer.multihead_attn.out_proj.weight.detach().numpy()
        np_layer.multihead_attn.out_proj_bias = pt_layer.multihead_attn.out_proj.bias.detach().numpy()

        # Feed Forward
        np_layer.linear1.weights = pt_layer.linear1.weight.detach().numpy()
        np_layer.linear1.bias = pt_layer.linear1.bias.detach().numpy()
        np_layer.linear2.weights = pt_layer.linear2.weight.detach().numpy()
        np_layer.linear2.bias = pt_layer.linear2.bias.detach().numpy()

        # Layer Norms
        np_layer.norm1.weight = pt_layer.norm1.weight.detach().numpy()
        np_layer.norm1.bias = pt_layer.norm1.bias.detach().numpy()
        np_layer.norm2.weight = pt_layer.norm2.weight.detach().numpy()
        np_layer.norm2.bias = pt_layer.norm2.bias.detach().numpy()
        np_layer.norm3.weight = pt_layer.norm3.weight.detach().numpy()
        np_layer.norm3.bias = pt_layer.norm3.bias.detach().numpy()

    # Финальный norm (если есть)
    if np_decoder.norm is not None and pt_decoder.norm is not None:
        np_decoder.norm.weight = pt_decoder.norm.weight.detach().numpy()
        np_decoder.norm.bias = pt_decoder.norm.bias.detach().numpy()


def test_transformer_decoder():
    d_model = 32
    nhead = 4
    dim_feedforward = 64
    num_layers = 3
    seq_len_tgt = 5
    seq_len_mem = 7
    batch_size = 2

    print("Инициализация моделей...")

    # PyTorch
    pt_decoder_layer = torch.nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=0.0, batch_first=True
    )
    pt_norm = torch.nn.LayerNorm(d_model)
    pt_decoder = torch.nn.TransformerDecoder(pt_decoder_layer, num_layers=num_layers, norm=pt_norm)
    pt_decoder.eval()

    # NumPy
    np_decoder_layer = TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=0.0, batch_first=True
    )
    np_norm = LayerNorm(d_model)
    np_decoder = TransformerDecoder(np_decoder_layer, num_layers=num_layers, norm=np_norm)
    np_decoder.eval()

    # Копируем веса
    copy_decoder_weights(pt_decoder, np_decoder)
    print("Веса скопированы!")

    # Входные данные
    np.random.seed(42)
    tgt_np = np.random.randn(batch_size, seq_len_tgt, d_model).astype(np.float32)
    mem_np = np.random.randn(batch_size, seq_len_mem, d_model).astype(np.float32)

    # Causal mask для tgt
    mask_np = np.triu(np.ones((seq_len_tgt, seq_len_tgt), dtype=bool), k=1)

    tgt_pt = torch.tensor(tgt_np)
    mem_pt = torch.tensor(mem_np)
    mask_pt = torch.tensor(mask_np)

    print("Forward pass...")

    # PyTorch
    with torch.no_grad():
        out_pt = pt_decoder(tgt_pt, mem_pt, tgt_mask=mask_pt)

    # NumPy
    out_np = np_decoder(tgt_np, mem_np, tgt_mask=mask_np)

    # Сравнение
    diff = np.max(np.abs(out_pt.numpy() - out_np))
    print(f"\nOutput Max Diff: {diff:.2e}")
    print(f"Input shape:  {tgt_np.shape}")
    print(f"Output shape: {out_np.shape}")
    print(f"Num layers:   {num_layers}")

    if diff < 1e-5:
        print("TRANSFORMER DECODER OK")
    else:
        print("TRANSFORMER DECODER IS BROKEN")


if __name__ == "__main__":
    test_transformer_decoder()