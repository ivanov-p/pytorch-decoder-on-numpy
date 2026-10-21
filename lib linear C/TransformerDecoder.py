import copy
import numpy as np
from base import Module

__all__ = ['TransformerDecoder']


class TransformerDecoder(Module):
    """
    Аналог torch.nn.TransformerDecoder.
    
    Стек из num_layers копий TransformerDecoderLayer.
    Опционально применяет финальную нормализацию (LayerNorm) в самом конце.
    
    Args:
        decoder_layer: экземпляр TransformerDecoderLayer или список слоёв.
                       Если передан один экземпляр, он будет скопирован
                       num_layers раз через copy.deepcopy.
        num_layers: количество слоёв в стеке (игнорируется, если decoder_layer
                    уже является списком).
        norm: опциональный модуль нормализации (обычно LayerNorm), применяемый
              после последнего слоя декодера. По умолчанию None.
    """

    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        norm=None,
    ):
        super().__init__()

        # Если передан один слой — клонируем его num_layers раз (как в PyTorch)
        if isinstance(decoder_layer, list):
            self.layers = decoder_layer
            if len(decoder_layer) != num_layers:
                raise ValueError(
                    f"Length of decoder_layer list ({len(decoder_layer)}) "
                    f"does not match num_layers ({num_layers})"
                )
        else:
            self.layers = [copy.deepcopy(decoder_layer) for _ in range(num_layers)]

        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: np.ndarray,
        memory: np.ndarray,
        tgt_mask: np.ndarray | None = None,
        memory_mask: np.ndarray | None = None,
        tgt_key_padding_mask: np.ndarray | None = None,
        memory_key_padding_mask: np.ndarray | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> np.ndarray:
        """
        Последовательно прогоняет tgt через все слои декодера.
        
        Args:
            tgt: входная последовательность декодера.
            memory: выходная последовательность энкодера (или пустой тензор
                    для авторегрессионных моделей).
            tgt_mask: маска для самовнимания tgt (например, causal mask).
            memory_mask: маска для cross-attention на memory.
            tgt_key_padding_mask: маска паддингов для tgt.
            memory_key_padding_mask: маска паддингов для memory.
            tgt_is_causal: если True, автоматически применяется causal mask для tgt.
            memory_is_causal: если True, автоматически применяется causal mask для memory.
            
        Returns:
            Тензор той же формы, что и tgt.
        """
        output = tgt

        # Последовательно прогоняем через каждый слой
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        # Финальная нормализация (если задана)
        if self.norm is not None:
            output = self.norm(output)

        return output

    # Для удобства вызова как функции
    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)