import numpy as np
import math
from base import Module

__all__ = ['PositionalEncoding']


class PositionalEncoding(Module):
    """
    Аналог PositionalEncoding из вашего gpt.py.
    
    Синусоидальное позиционное кодирование:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: размерность эмбеддинга.
        max_len: максимальная длина последовательности, для которой
                 заранее вычисляются позиционные кодирования.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Предварительно вычисляем позиционные кодирования
        # pe shape: (1, max_len, d_model)
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        # position shape: (max_len, 1) — позиция каждого токена
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]

        # div_term shape: (d_model/2,) — знаменатель для чётных/нечётных индексов
        # Это эквивалент 1 / 10000^(2i/d_model)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )

        # Чётные индексы — синусы, нечётные — косинусы
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Добавляем размерность batch: (1, max_len, d_model)
        # В PyTorch это register_buffer — у нас просто атрибут
        self.pe = pe[np.newaxis, :, :]  # (1, max_len, d_model)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: тензор формы (batch, seq_len, d_model) — эмбеддинги токенов.
            
        Returns:
            Тензор той же формы, но с добавленным позиционным кодированием.
        """
        seq_len = x.shape[1]
        
        # Проверка, что длина последовательности не превышает max_len
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                f"Increase max_len in PositionalEncoding."
            )

        # Берём первые seq_len позиций из буфера pe
        # self.pe[:, :seq_len, :] имеет форму (1, seq_len, d_model)
        # x имеет форму (batch, seq_len, d_model)
        # Broadcasting автоматически добавит batch-измерение
        return x + self.pe[:, :seq_len, :]