import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FLASH = True
except Exception:  # pragma: no cover
    # testing is gitignored (oopsie), TODO: push the tests
    flash_attn_varlen_func = None
    _HAS_FLASH = False



"""Block-diagonal (document-packed) causal attention.

During pretraining several documents are packed into each ``max_length`` sequence. A token must
only attend within its own document, causally. Previously this was expressed as a dense
``[B, 1, S, S]`` boolean mask handed to ``F.scaled_dot_product_attention`` -- but a custom mask
disqualifies SDPA's FlashAttention backend, forcing the memory-efficient/math kernel that
materializes and reads the full S*S mask every call.

Here we instead pass the packing structure as ``cu_seqlens`` (cumulative segment boundaries over
the flattened ``B*S`` token axis) and call ``flash_attn_varlen_func``. This skips all cross-document
compute (attention cost scales with sum(segment_len^2) rather than S^2) and never materializes a
mask.
"""

# block diagonal (document-packed) causal attention is implemented in flash-attn as a variable length attention
# each token only attends to tokens in its own segment (document) and the segments are packed into a batch of sequences
# [B, 1, S, S] masks cant be used, as it disqualifies FlashAttention and forces the SDPA fallback
# the segments are defined by the cumulative segment lengths (cu_seqlens) over the flattened B*S token axis
# => O(sum(segment_len^2)) rather than O(S^2) compute and memory and no full mask is materialized :D

def _default_cu_seqlens(batch_size: int, seq_len: int, device) -> torch.Tensor:
    # one full-length segment per sample -> plain causal attention (the no-packing case)
    return torch.arange(0, (batch_size + 1) * seq_len, seq_len, device=device, dtype=torch.int32)


def cu_seqlens_from_doc_ids(document_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Build flash-attn ``cu_seqlens`` from a batch-aligned ``[B, S]`` segment-id tensor.

    Each contiguous run of equal ids within a row is one packed document; rows (samples) never
    share a segment (a boundary is forced at every sample start). ``document_ids`` is passed
    through the dataloader instead of ``cu_seqlens`` because it is batch-aligned (dim 0 == B), so
    ``accelerate``'s batch splitting/device placement handles it like ``input_ids`` rather than
    truncating a ragged ``cu_seqlens`` to the batch size.
    """
    B, S = document_ids.shape
    device = document_ids.device
    flat = document_ids.reshape(-1)
    pos = torch.arange(B * S, device=device)
    boundary = torch.ones(B * S, dtype=torch.bool, device=device)
    boundary[1:] = flat[1:] != flat[:-1]
    boundary |= (pos % S == 0)  # force a break at each sample seam even if ids collide
    starts = boundary.nonzero().flatten()
    ends = torch.cat([starts[1:], torch.tensor([B * S], device=device, dtype=starts.dtype)])
    seg_lens = ends - starts
    cu = torch.zeros(starts.numel() + 1, dtype=torch.int32, device=device)
    cu[1:] = seg_lens.cumsum(0).to(torch.int32)
    # max_seqlen only has to be an upper bound for flashs scheduling, and segments never span a sample, so S works. the true max would cost a .item() host sync every step (which is too much overhead)
    return cu, S


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Document packed causal attention

    Args:
        q: queries, shape ``[B, Hq, S, D]``.
        k, v: keys/values, shape ``[B, Hkv, S, D]`` (GQA: ``Hkv`` may divide ``Hq``, flash handles the head broadcast natively, so KV heads are not pre repeated)
        cu_seqlens: int32 tensor ``[num_segments + 1]`` of cumulative segment lengths over the flattened ``B*S`` token axis (row major: all of sample 0s tokens, then sample 1, etc).
            ``None`` falls back to one segment per sample (plain causal attn)
        max_seqlen: longest segment length (int). Ignored when ``cu_seqlens`` is ``None``
        dropout_p: attention dropout probability
        softmax_scale: attention scale. Defaults to ``D ** -0.5``
        causal: apply causal masking within each segment

    Returns:
        Attention output, shape ``[B, S, Hq, D]``

    Notes: 
        - Hq: number of query heads 
        - Hkv: number of key/value heads
        - GQA: Hkv must divide Hq and Hq > Hkv
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    if cu_seqlens is None:
        cu_seqlens = _default_cu_seqlens(B, S, q.device)
        max_seqlen = S

    if _HAS_FLASH:
        # flash wants [total_tokens, H, D], transpose+reshape forces contiguity
        qf = q.transpose(1, 2).reshape(B * S, Hq, D)
        kf = k.transpose(1, 2).reshape(B * S, Hkv, D)
        vf = v.transpose(1, 2).reshape(B * S, Hkv, D)
        cu = cu_seqlens.to(device=q.device, dtype=torch.int32)
        out = flash_attn_varlen_func(
            qf, kf, vf,
            cu, cu,
            int(max_seqlen), int(max_seqlen),
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        return out.reshape(B, S, Hq, D)

    # fallback (no flash-attn): rebuild the block mask and use SDPA => slowwww
    return _sdpa_fallback(q, k, v, cu_seqlens, B, S, Hq, Hkv, dropout_p, softmax_scale, causal)


def cached_attention(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    kv_cache,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Causal attention against a growable ``LayerKVCache``, for single-sequence (unpacked)
    incremental decoding.

    ``q`` and ``k_new``/``v_new`` cover only the newly-appended tokens for this call (``Lq``
    positions); ``kv_cache`` already holds every earlier token at this depth (see
    ``modules/model/kv_cache.py`` for why that is valid under this model's causal structure). The
    new K/V are appended to the cache first, then every query attends causally over the FULL
    (old + new) key/value set -- aligned so query row ``i`` (absolute position
    ``cache_len_before + i``) may see key column ``j`` iff ``j <= cache_len_before + i``.

    Args:
        q: queries, shape ``[B, Hq, Lq, D]``.
        k_new, v_new: this call's new keys/values, shape ``[B, Hkv, Lq, D]``.
        kv_cache: a ``LayerKVCache`` to append to and read the full history from.
        dropout_p: attention dropout probability.
        softmax_scale: attention scale. Defaults to ``D ** -0.5``.

    Returns:
        Attention output, shape ``[B, Lq, Hq, D]`` (matches ``varlen_attention``'s layout).
    """
    B, Hq, Lq, D = q.shape
    Hkv = k_new.shape[1]
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    cache_len_before = kv_cache.length
    k, v = kv_cache.update(k_new, v_new)
    total_len = k.shape[2]

    if Hkv != Hq:
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        v = v.repeat_interleave(Hq // Hkv, dim=1)

    if Lq == 1:
        # the single new token is the last (and therefore latest) position in the cache -> every
        # key is already causally visible, no mask needed
        attn_mask, is_causal = None, False
    elif cache_len_before == 0:
        # plain prefill from an empty cache -> standard top-left-aligned causal mask
        attn_mask, is_causal = None, True
    else:
        # continuing an already-primed cache with more than one new token (e.g. accepting several
        # MTP-drafted tokens at once) -> explicit bottom-right-aligned causal mask
        device = q.device
        i = torch.arange(Lq, device=device).unsqueeze(1)
        j = torch.arange(total_len, device=device).unsqueeze(0)
        attn_mask = (j <= (cache_len_before + i))[None, None, :, :]
        is_causal = False

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=softmax_scale, is_causal=is_causal,
    )
    return out.transpose(1, 2)  # [B, Lq, Hq, D]


def _sdpa_fallback(q, k, v, cu_seqlens, B, S, Hq, Hkv, dropout_p, softmax_scale, causal):
    # default scaled product attention with a block diagonal mask (one block per document segment)
    device = q.device
    # derive a per token segment id from the internal boundaries, then mask same segment pairs
    seg_id = torch.zeros(B * S, dtype=torch.long, device=device)
    internal = cu_seqlens[1:-1].long()
    if internal.numel() > 0:
        seg_id[internal] = 1
    seg_id = torch.cumsum(seg_id, dim=0).view(B, S)             # [B, S]
    same = seg_id[:, :, None] == seg_id[:, None, :]             # [B, S, S]
    if causal:
        same = same & torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    attn_mask = same[:, None, :, :]                             # [B, 1, S, S]

    if Hkv != Hq:
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        v = v.repeat_interleave(Hq // Hkv, dim=1)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=softmax_scale
    )
    return out.transpose(1, 2)  # [B, S, Hq, D]
