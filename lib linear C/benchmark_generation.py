import time
import numpy as np
import torch
import torch.nn as nn
import math

# Импортируем ваши NumPy компоненты
from Embedding import Embedding
from PositionalEncoding import PositionalEncoding
from TransformerDecoder import TransformerDecoder
from TransformerDecoderLayer import TransformerDecoderLayer
from Linear import Linear
from LayerNorm import LayerNorm


# ============================================================================
# 1. Вспомогательные функции для копирования весов из PyTorch в NumPy
# ============================================================================

def copy_weights_from_torch(pt_layer, np_layer):
    """Копирует веса из PyTorch TransformerDecoderLayer в NumPy TransformerDecoderLayer"""
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


def copy_decoder_weights(pt_decoder, np_decoder):
    """Копирует веса из PyTorch TransformerDecoder в NumPy TransformerDecoder"""
    for i in range(len(np_decoder.layers)):
        pt_layer = pt_decoder.layers[i]
        np_layer = np_decoder.layers[i]
        copy_weights_from_torch(pt_layer, np_layer)

    # Финальный norm
    if np_decoder.norm is not None and pt_decoder.norm is not None:
        np_decoder.norm.weight = pt_decoder.norm.weight.detach().numpy()
        np_decoder.norm.bias = pt_decoder.norm.bias.detach().numpy()


# ============================================================================
# 2. PyTorch модель для генерации (аналог gpt.py с autoregressive generation)
# ============================================================================

class TorchPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class BuiltInDecoderLM_Torch(nn.Module):
    """Полная авторегрессионная языковая модель на PyTorch для benchmark."""
    def __init__(self, vocab_size, d_model=2, n_layers=2, n_heads=2,
                 d_ff=2, dropout=0.0, max_len=5):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = TorchPositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=final_norm
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def generate_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src):
        batch_size, seq_len = src.size()
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        tgt_mask = self.generate_mask(seq_len).to(src.device)
        output = self.decoder(
            tgt=x,
            memory=torch.zeros(batch_size, 0, self.d_model).to(x.device),
            tgt_mask=tgt_mask
        )
        logits = self.output_projection(output)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        """Авторегрессионная генерация токенов."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            # Обрезаем до max_len если нужно (для простоты не делаем здесь)
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


# ============================================================================
# 3. NumPy модель для генерации (аналог gpt_numpy.py с autoregressive generation)
# ============================================================================

def generate_causal_mask_np(sz: int) -> np.ndarray:
    mask = np.triu(np.ones((sz, sz), dtype=np.float32), k=1)
    mask = np.where(mask == 1.0, -np.inf, 0.0)
    return mask


class BuiltInDecoderLM_Numpy:
    """Полная авторегрессионная языковая модель на NumPy для benchmark."""
    def __init__(self, vocab_size, d_model=2, n_layers=2, n_heads=2,
                 d_ff=2, dropout=0.0, max_len=5):
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        final_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=final_norm,
        )
        self.output_projection = Linear(d_model, vocab_size)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        batch_size, seq_len = src.shape
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        tgt_mask = generate_causal_mask_np(seq_len)
        memory = np.zeros((batch_size, 0, self.d_model), dtype=np.float32)
        output = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=False,
        )
        logits = self.output_projection(output)
        return logits

    def generate(self, input_ids: np.ndarray, max_new_tokens: int) -> np.ndarray:
        """Авторегрессионная генерация токенов."""
        generated = input_ids.copy()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token_logits = logits[:, -1, :]
            next_token = np.argmax(next_token_logits, axis=-1)
            next_token = next_token.reshape(-1, 1)
            generated = np.concatenate([generated, next_token], axis=1)
        return generated


# ============================================================================
# 4. Функция бенчмарка для одного набора параметров
# ============================================================================

def run_benchmark_for_config(config_name, vocab_size, d_model, n_layers, n_heads,
                              d_ff, max_len, batch_size, seq_len_input, max_new_tokens,
                              num_repeats=3):
    """
    Замеряет время генерации для одной конфигурации.
    Возвращает dict со средним временем для torch и numpy.
    """
    print(f"\n{'='*70}")
    print(f"Конфигурация: {config_name}")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, n_layers={n_layers}, "
          f"n_heads={n_heads}, d_ff={d_ff}")
    print(f"  max_len={max_len}, batch_size={batch_size}, seq_len_input={seq_len_input}, "
          f"max_new_tokens={max_new_tokens}")
    print(f"  Повторов замера: {num_repeats}")
    print(f"{'='*70}")

    # --- Создаём PyTorch модель ---
    torch.manual_seed(42)
    pt_model = BuiltInDecoderLM_Torch(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=max_len
    )
    pt_model.eval()

    # --- Создаём NumPy модель ---
    np.random.seed(42)
    np_model = BuiltInDecoderLM_Numpy(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.0,
        max_len=max_len
    )

    # --- Копируем веса из PyTorch в NumPy ---
    copy_decoder_weights(pt_model.decoder, np_model.decoder)
    # Эмбеддинги
    np_model.token_embedding.weight = pt_model.token_embedding.weight.detach().numpy()
    # Позиционное кодирование — вычисляется детерминированно, совпадает автоматически
    # Выходной проекционный слой
    np_model.output_projection.weights = pt_model.output_projection.weight.detach().numpy()
    np_model.output_projection.bias = pt_model.output_projection.bias.detach().numpy()

    # --- Генерируем одинаковые входные данные ---
    np.random.seed(42)
    torch.manual_seed(42)
    input_ids_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len_input), dtype=np.int64)
    input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)

    # --- Warm-up (прогрев кэша и JIT) ---
    try:
        with torch.no_grad():
            _ = pt_model.generate(input_ids_pt, min(2, max_new_tokens))
        _ = np_model.generate(input_ids_np, min(2, max_new_tokens))
    except Exception as e:
        print(f"  Warm-up ошибка: {e}")

    # --- Замер времени PyTorch ---
    times_pt = []
    for rep in range(num_repeats):
        start = time.perf_counter()
        with torch.no_grad():
            out_pt = pt_model.generate(input_ids_pt, max_new_tokens)
        end = time.perf_counter()
        times_pt.append(end - start)

    avg_time_pt = np.mean(times_pt)
    std_time_pt = np.std(times_pt)

    # --- Замер времени NumPy ---
    times_np = []
    for rep in range(num_repeats):
        start = time.perf_counter()
        out_np = np_model.generate(input_ids_np, max_new_tokens)
        end = time.perf_counter()
        times_np.append(end - start)

    avg_time_np = np.mean(times_np)
    std_time_np = np.std(times_np)

    speedup = avg_time_pt / avg_time_np if avg_time_np > 0 else float('inf')

    print(f"  PyTorch:   {avg_time_pt:.6f} ± {std_time_pt:.6f} сек")
    print(f"  NumPy:     {avg_time_np:.6f} ± {std_time_np:.6f} сек")
    print(f"  Ускорение (PT/NP): {speedup:.2f}x")

    # Проверка корректности: сравним результаты
    out_pt_np = out_pt.numpy()
    match = np.allclose(out_pt_np, out_np, atol=1e-4, rtol=1e-4)
    print(f"  Результаты совпадают: {'OK' if match else 'FAIL'}")

    return {
        "config": config_name,
        "params": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "max_len": max_len,
            "batch_size": batch_size,
            "seq_len_input": seq_len_input,
            "max_new_tokens": max_new_tokens,
        },
        "time_torch_sec": avg_time_pt,
        "time_numpy_sec": avg_time_np,
        "speedup": speedup,
        "results_match": match,
    }


# ============================================================================
# 5. Основные конфигурации для бенчмарка (5 объёмов данных)
# ============================================================================

def main():
    print("ЗАПУСК БЕНЧМАРКА: Генерация токенов — PyTorch vs NumPy")
    print("=" * 70)
    print("Примечание: все замеры в eval режиме, dropout=0.0, без градиентов.")
    print("Для честного сравнения веса копируются из PyTorch в NumPy.")
    print("=" * 70)

    # 5 конфигураций от маленькой к большой
    configs = [
        # (name, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, batch_size, seq_len_input, max_new_tokens)
        ("Tiny",       10,   8,   1,   2,   16,  10,   1,   3,   3),
        ("Small",      32,  16,   2,   4,   32,  20,   2,   5,   5),
        ("Medium",     64,  32,   2,   4,   64,  30,   2,   8,   8),
        ("Large",     128,  64,   3,   8,  128,  50,   4,  10,  10),
        ("XLarge",    256, 128,   4,   8,  256, 100,   4,  15,  15),
    ]

    results = []
    for cfg in configs:
        name, vs, dm, nl, nh, df, ml, bs, si, mt = cfg
        res = run_benchmark_for_config(
            config_name=name,
            vocab_size=vs,
            d_model=dm,
            n_layers=nl,
            n_heads=nh,
            d_ff=df,
            max_len=ml,
            batch_size=bs,
            seq_len_input=si,
            max_new_tokens=mt,
            num_repeats=3
        )
        results.append(res)

    # --- Итоговая таблица ---
    print("\n\n" + "=" * 90)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 90)
    header = f"{'Конфиг':<10} {'Vocab':>6} {'d_mod':>6} {'Layers':>7} {'Batch':>6} {'SeqIn':>6} {'NewTok':>7} {'Torch(s)':>10} {'NumPy(s)':>10} {'Speedup':>8} {'Match':>6}"
    print(header)
    print("-" * 90)
    for r in results:
        p = r["params"]
        print(f"{r['config']:<10} {p['vocab_size']:>6} {p['d_model']:>6} {p['n_layers']:>7} "
              f"{p['batch_size']:>6} {p['seq_len_input']:>6} {p['max_new_tokens']:>7} "
              f"{r['time_torch_sec']:>10.6f} {r['time_numpy_sec']:>10.6f} "
              f"{r['speedup']:>8.2f}x {'OK' if r['results_match'] else 'FAIL':>6}")
    print("=" * 90)

    # Сохранение результатов в JSON
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nРезультаты сохранены в benchmark_results.json")


if __name__ == "__main__":
    main()