import numpy as np
from typing import Callable, Union


from base import Module, Dropout
from Linear import Linear
from LayerNorm import LayerNorm
from MultiheadAttention import MultiheadAttention
from activations import _get_activation_fn

class TransformerDecoderLayer(Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[np.ndarray], np.ndarray]] = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        
        
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        
        
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        
        
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = "relu"
        super().__setstate__(state)

    def __call__(
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
        x = tgt
        
        if self.norm_first:
            # Pre-LN
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            # Post-LN
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: np.ndarray,
        attn_mask: np.ndarray | None,
        key_padding_mask: np.ndarray | None,
        is_causal: bool = False,
    ) -> np.ndarray:
        # x является и query, и key, и value
        x_attn, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        return self.dropout1(x_attn)

    def _mha_block(
        self,
        x: np.ndarray,
        mem: np.ndarray,
        attn_mask: np.ndarray | None,
        key_padding_mask: np.ndarray | None,
        is_causal: bool = False,
    ) -> np.ndarray:
        # x - query, mem - key и value
        x_attn, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )
        return self.dropout2(x_attn)

    def _ff_block(self, x: np.ndarray) -> np.ndarray:
        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x_ff)










import numpy as np


decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=False)


# Формат: (seq_len, batch_size, d_model)
seq_len_tgt = 20
batch_size = 32
d_model = 512
seq_len_mem = 10

tgt = np.random.rand(seq_len_tgt, batch_size, d_model).astype(np.float32)
memory = np.random.rand(seq_len_mem, batch_size, d_model).astype(np.float32)

# Создаем causal mask для tgt (чтобы токены не видели будущие)
# True = маскировать (игнорировать), False = разрешить внимание
tgt_mask = np.triu(np.ones((seq_len_tgt, seq_len_tgt), dtype=bool), k=1)


out = decoder_layer(
    tgt=tgt,
    memory=memory,
    tgt_mask=tgt_mask,
    tgt_is_causal=False # Если True, маска сгенерируется внутри автоматически
)

print("Успешно!")
print(f"Входной shape tgt:   {tgt.shape}")
print(f"Входной shape memory:{memory.shape}")
print(f"Выходной shape out:  {out.shape}")

# Проверка для batch_first=True
decoder_layer_bf = TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
tgt_bf = np.random.rand(batch_size, seq_len_tgt, d_model).astype(np.float32)
memory_bf = np.random.rand(batch_size, seq_len_mem, d_model).astype(np.float32)

out_bf = decoder_layer_bf(tgt=tgt_bf, memory=memory_bf, tgt_is_causal=True)
print(f"\nbatch_first=True успешно!")
print(f"Выходной shape out_bf: {out_bf.shape}")