import numpy as np
from base import Module

__all__ = ['Embedding']


class Embedding(Module):
    """
    Аналог torch.nn.Embedding.
    
    Простая таблица эмбеддингов: преобразует индексы токенов (целые числа)
    в плотные векторы заданной размерности.
    
    Args:
        num_embeddings: размер словаря (количество уникальных токенов).
        embedding_dim: размерность вектора эмбеддинга для каждого токена.
        padding_idx: если задан, то эмбеддинг с этим индексом всегда будет
                     нулевым вектором (как в PyTorch).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Инициализация весов — как в PyTorch: нормальное распределение N(0, 1)
        rng = np.random.default_rng()
        self.weight = rng.standard_normal(
            (num_embeddings, embedding_dim), dtype=np.float32
        )

        # Если задан padding_idx, зануляем соответствующую строку
        if padding_idx is not None:
            self.weight[padding_idx] = 0.0

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input: тензор целочисленных индексов произвольной формы,
                   например (batch_size, seq_len).
                   
        Returns:
            Тензор формы (*input.shape, embedding_dim), dtype=np.float32.
        """
        
        input = np.asarray(input, dtype=np.int64)
        
        # Проверка выхода за границы словаря
        if input.min() < 0 or input.max() >= self.num_embeddings:
            raise ValueError(
                f"Embedding index out of range: "
                f"got min={input.min()}, max={input.max()}, "
                f"but num_embeddings={self.num_embeddings}"
            )

        
        
        out = self.weight[input]

        # Гарантируем, что padding_idx всегда нулевой
        # (на случай, если веса были изменены вручную)
        if self.padding_idx is not None:
            out[..., self.padding_idx, :] = 0.0

        return out