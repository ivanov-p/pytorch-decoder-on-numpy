import numpy as np

from scaled_dot_product_attention_low import scaled_dot_product_attention

def multi_head_attention_forward(

    query:  np.ndarray,
    key:  np.ndarray,
    value:  np.ndarray,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight:  np.ndarray | None, # !!!!!!!!!!!!!! NOT SET None
    in_proj_bias:  np.ndarray | None,
    bias_k:  np.ndarray | None,
    bias_v:  np.ndarray | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight:  np.ndarray,
    out_proj_bias:  np.ndarray | None,
    training: bool = True,
    key_padding_mask:  np.ndarray | None = None,
    need_weights: bool = True, # !!!!!!!!!!!!!!!!!!  SET False
    attn_mask:  np.ndarray | None = None,
    use_separate_proj_weight: bool = False, # !!!!! SET False
    q_proj_weight:  np.ndarray | None = None,
    k_proj_weight:  np.ndarray | None = None,
    v_proj_weight:  np.ndarray | None = None,
    static_k:  np.ndarray | None = None, # !!!!!!!!!!!!!! SET None
    static_v:  np.ndarray | None = None, # !!!!!!!!!!!!!! SET None
    average_attn_weights: bool = True,
    is_causal: bool = False,

) -> tuple[ np.ndarray,  np.ndarray | None]:

    r"""Forward method for MultiHeadAttention.

    See :class:`torch.nn.MultiheadAttention` for details.
    
    
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is a binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`

            Note: `needs_weight` defaults to `True`, but should be set to `False`

            For best performance when attention weights are not needed.

            *Setting needs_weights to `True`

            leads to a significant performance degradation.*

        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.

            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.

        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.

            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True

  
  

    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          
          If a Float np.ndarray is provided, it will be directly added to the value.
          If a Bool np.ndarray is provided, the positions with the

          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked

          positions. If a Bool np.ndarray is provided, positions with ``True``

          are not allowed to attend while ``False`` values will be unchanged. If a Float np.ndarray

          is provided, it will be added to the attention weight.

        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,

          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,

          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

  

        Outputs:

        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,

          E is the embedding dimension.

        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns

          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or

          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and

          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per

          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

    """

    
  

    is_batched = True

    '''
    WE EXPECT THAT
    _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    ALWAYS RETURN True
    '''

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.

  

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape # AFTER THAT embed_dim MUST BE not isinstance(embed_dim, torch. np.ndarray)
    src_len, _, _ = key.shape

  

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

  

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None

  
  

    head_dim = embed_dim // num_heads
  

    #
    # compute in-projection
    #


    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)  

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    # pyrefly: ignore [bad-argument-type, no-matching-overload]

    
    q = transpose(q.reshape(tgt_len, bsz * num_heads, head_dim), [0, 1])
  

    # pyrefly: ignore [bad-argument-type, no-matching-overload]
    k = transpose(k.reshape(k.shape[0], bsz * num_heads, head_dim), [0, 1])

  
    # pyrefly: ignore [bad-argument-type, no-matching-overload]
    v = transpose(v.reshape(v.shape[0], bsz * num_heads, head_dim), [0, 1])

    
    # update source sequence length after adjustments
    src_len = k.shape[1]
  
    

    #
    # (deep breath) calculate attention and out projection
    #

  

    # attn_mask can be either (L,S) or (N*num_heads, L, S)
    # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
    # in order to match the input for SDPA of (N, num_heads, L, S)

  
  

    # pyrefly: ignore [bad-argument-type]
    q = q.reshape(bsz, num_heads, tgt_len, head_dim)

    # pyrefly: ignore [bad-argument-type]
    k = k.reshape(bsz, num_heads, src_len, head_dim)

    # pyrefly: ignore [bad-argument-type]
    v = v.reshape(bsz, num_heads, src_len, head_dim)

  

    attn_output = scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, is_causal
    )

    # Free q, k, v and their backing projection storage before the
    # .contiguous() call below allocates.  In self-attention the three
    #  np.ndarrays are views of a single packed projection, so releasing all
    # references here lets the allocator reclaim that memory immediately.

    del q, k, v

    attn_output = np.transpose(attn_output, (2, 0, 1, 3))
    attn_output = np.ascontiguousarray(attn_output)
    attn_output = attn_output.reshape(bsz * tgt_len, embed_dim)

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.reshape(tgt_len, bsz, attn_output.shape[1])
  
    return attn_output, None





def _none_or_dtype(tensor: np.ndarray):
    return None if tensor is None else tensor.dtype




def _canonical_mask(**kwargs):
    return None





def _in_projection_packed(
    q:  np.ndarray,
    k:  np.ndarray,
    v:  np.ndarray,
    w:  np.ndarray,
    b:  np.ndarray | None = None,
) -> list[ np.ndarray]:
    r"""Perform the in-projection step of the attention operation, using packed weights.

    Output is a triple containing projection  np.ndarrays for query, key and value.

    Args:
        q, k, v: query, key and value  np.ndarrays to be projected. For self-attention,
            these are typically the same  np.ndarray; for encoder-decoder attention,
            k and v are typically the same  np.ndarray. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single  np.ndarray. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single  np.ndarray
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output  np.ndarray will have the
            same shape as the corresponding input  np.ndarray.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            
            proj = proj.reshape(*proj.shape[:-1], 3, E)
            proj = np.expand_dims(proj, axis=0) # ?

                # proj.unflatten(-1, (3, E))
                #.unsqueeze(0)
            
            proj = transpose(proj, [0, -2])

            proj = np.squeeze(proj, axis=-2)
            proj = np.ascontiguousarray(proj)

            # pyrefly: ignore [bad-return]
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = np.split(w, [E])

            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = np.split(b, [E])
            
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.reshape(*kv_proj.shape[:-1], 2, E)
            kv_proj = np.expand_dims(kv_proj, axis=0)
            kv_proj = transpose(kv_proj, [0, -2])
            kv_proj = np.squeeze(kv_proj, axis=-2)
            kv_proj = np.ascontiguousarray(kv_proj)

            # pyrefly: ignore [bad-return]
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = np.split(w, [E, 2 * E])
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = np.split(b, [E, 2 * E])
        # pyrefly: ignore [bad-return]
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)



def linear(input, weight, bias=None):
    out = input @ transpose_last_two_axes(weight)
    if bias is not None:
        out = out + bias
    return out




def transpose_last_two_axes(x: np.ndarray) -> np.ndarray:
    axis1 = x.ndim - 1      # [-1]
    axis2 = x.ndim - 2      # [-2]


    axes = list(range(x.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    return np.transpose(x, axes)


def transpose(x: np.ndarray, axes_to_set: list[int]) -> np.ndarray:
    if len(axes_to_set) != 2:
        raise ValueError("len(axes_to_set) must be 2")

    axes_temp = axes_to_set.copy()

    for i in range(len(axes_temp)):
        if axes_temp[i] < 0:
            axes_temp[i] = x.ndim + axes_temp[i]

    axes = list(range(x.ndim))
    axis1 = axes_temp[0]
    axis2 = axes_temp[1]
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    return np.transpose(x, axes)