import numpy as np
from multi_head_attention_forward_low import multi_head_attention_forward
from base import Module

class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        rng = np.random.default_rng()
        # in_proj_weight shape: (3 * embed_dim, embed_dim)
        self.in_proj_weight = rng.random((3 * embed_dim, embed_dim), dtype=np.float32)
        self.in_proj_bias = rng.random(3 * embed_dim, dtype=np.float32) if bias else None
        
        # out_proj_weight shape: (embed_dim, embed_dim)
        self.out_proj_weight = rng.random((embed_dim, embed_dim), dtype=np.float32)
        self.out_proj_bias = rng.random(embed_dim, dtype=np.float32) if bias else None

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        key_padding_mask: np.ndarray | None = None,
        need_weights: bool = True,
        attn_mask: np.ndarray | None = None,
        is_causal: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        
        if self.batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            query = np.transpose(query, (1, 0, 2))
            key = np.transpose(key, (1, 0, 2))
            value = np.transpose(value, (1, 0, 2))
            
        attn_output, attn_output_weights = multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout if self.training else 0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        
        if self.batch_first:
            # (seq, batch, feature) -> (batch, seq, feature)
            attn_output = np.transpose(attn_output, (1, 0, 2))
            
        return attn_output, attn_output_weights