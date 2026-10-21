import numpy as np
import math

# Импорты ваших реализованных компонентов
from base import Module
from Embedding import Embedding
from PositionalEncoding import PositionalEncoding
from TransformerDecoder import TransformerDecoder
from TransformerDecoderLayer import TransformerDecoderLayer
from Linear import Linear

__all__ = ['BuiltInDecoderLM', 'generate_causal_mask']


def generate_causal_mask(sz: int) -> np.ndarray:
    """
    Создаёт маску для маскированного самовнимания (causal mask).
    Аналог:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
    """
    # Верхний треугольник (выше диагонали) будет 1, остальное 0
    mask = np.triu(np.ones((sz, sz), dtype=np.float32), k=1)
    # Заменяем 1 на -inf, а 0 оставляем как 0
    mask = np.where(mask == 1.0, -np.inf, 0.0)
    return mask


class BuiltInDecoderLM(Module):
    """
    Аналог BuiltInDecoderLM из вашего gpt.py, но на NumPy.
    Авторегрессионная языковая модель на базе стека декодеров.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 2,
        n_layers: int = 2,
        n_heads: int = 2,
        d_ff: int = 2,
        dropout: float = 0.1,
        max_len: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 1. Эмбеддинги
        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 2. Слой декодера (batch_first=True для удобства: (batch, seq_len, dim))
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Стек декодеров + финальная нормализация (опционально, но полезно)
        from LayerNorm import LayerNorm
        final_norm = LayerNorm(d_model)
        
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
            norm=final_norm,
        )

        # 4. Выходной слой (проекция на пространство словаря)
        self.output_projection = Linear(d_model, vocab_size)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        """
        Args:
            src: входные индексы токенов, форма (batch_size, seq_len), dtype=int
        
        Returns:
            logits: форма (batch_size, seq_len, vocab_size), dtype=float32
        """
        batch_size, seq_len = src.shape

        # Токенные эмбеддинги + масштабирование (как в оригинальном Transformer)
        x = self.token_embedding(src) * math.sqrt(self.d_model)

        # Позиционное кодирование
        x = self.pos_encoding(x)

        # Маска для маскированного внимания (causal mask)
        tgt_mask = generate_causal_mask(seq_len)

        # Пустой memory для авторегрессионной модели (длина последовательности = 0)
        # Наши компоненты уже умеют корректно обрабатывать размерность 0
        memory = np.zeros((batch_size, 0, self.d_model), dtype=np.float32)

        # Проход через стек декодеров
        # Передаем tgt_is_causal=False, так как мы явно передаём tgt_mask
        output = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=False,
        )

        # Проекция на пространство словаря
        logits = self.output_projection(output)
        
        return logits


# ==============================================================================
# Блок тестирования (аналог вашего кода внизу gpt.py)
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Инициализация модели BuiltInDecoderLM на NumPy...")
    
    # Параметры из вашего примера
    model = BuiltInDecoderLM(vocab_size=3, d_model=2, n_layers=2, n_heads=2, d_ff=2, max_len=5)
    model.eval() # Отключаем dropout для детерминированного результата

    # Пример данных: batch=1, seq_len=2, значения индексов от 0 до 2
    input_ids = np.random.randint(0, 3, size=(1, 2), dtype=np.int64)
    print(f"Входные данные (input_ids):\n{input_ids}")
    print(f"Форма входных данных: {input_ids.shape}")

    # Прямой проход
    logits = model(input_ids)

    print(f"\n✅ Успешно!")
    print(f"Выходная форма (logits): {logits.shape}")  # Ожидаем [1, 2, 3]
    print(f"Пример выходных логитов:\n{logits}")
    
    # Проверка формы
    assert logits.shape == (1, 2, 3), f"Ожидалась форма (1, 2, 3), получена {logits.shape}"
    print("\n🎉 Все проверки пройдены! Модель работает корректно.")