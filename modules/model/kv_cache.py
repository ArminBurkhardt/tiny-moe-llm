"""Incremental key/value cache for single-sequence (unpacked) autoregressive decoding.

Every attention call in this model -- the dense decoder's self-attention, and the MoE loop's
shared/self/cross/IR-expert attention -- ultimately runs through ``Gemma4TextAttention`` (see
``modules/model/gemma4.py``). All of it is causal, so a past token's output at a given depth
(decoder layer, or MoE loop) never changes once later tokens are appended -- which is exactly what
makes a growable KV cache valid here, including through the MoE's loop recurrence.

``LayerKVCache`` caches one such attention call's K/V. Composed into ``KVCache``, which mirrors the
model's structure: one slot per dense-decoder layer, plus one slot per (loop, non-MLP-expert) pair
(``shared_attn`` and each self/cross/IR expert get an independent cache per loop, since the MoE
block is the SAME weights re-applied ``n_loops`` times over an evolving hidden state, not
``n_loops`` independent layers).

Every ``forward()`` call site in this package defaults ``kv_cache=None``, reproducing the exact
packed/varlen training path -- this module is additive and never touches training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from modules.model.transformer import TinyMoETransformer


class LayerKVCache:
    """Growable [B, H, T, D] key/value cache for one attention call."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends new K/V along the sequence axis and returns the full cache so far."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None


class _MoELoopCache:
    """Caches for one loop iteration's always-on ``shared_attn`` plus its non-MLP expert slots."""

    def __init__(self, num_attn_slots: int):
        self.shared_attn = LayerKVCache()
        self.slots = [LayerKVCache() for _ in range(num_attn_slots)]

    def reset(self):
        self.shared_attn.reset()
        for slot in self.slots:
            slot.reset()


class KVCache:
    """Top-level cache threaded through ``TinyMoETransformer.forward`` for cached generation.

    Scoped to a single generation session (one prompt's prefill + decode loop): ``n_loops`` and
    the number of MoE attention slots are fixed at construction, matching whatever loop count and
    checkpoint the session was built for. Build a fresh one per ``generate()`` call rather than
    trying to reuse it across unrelated prompts.
    """

    def __init__(self, num_layers: int, n_loops: int, num_attn_slots: int):
        self.decoder = [LayerKVCache() for _ in range(num_layers)]
        self.moe = [_MoELoopCache(num_attn_slots) for _ in range(n_loops)]

    @property
    def length(self) -> int:
        """Number of tokens already cached (same across every slot by construction)."""
        return self.decoder[0].length if self.decoder else 0

    def reset(self):
        for layer in self.decoder:
            layer.reset()
        for loop in self.moe:
            loop.reset()

    @classmethod
    def for_model(cls, model: "TinyMoETransformer", n_loops: int | None = None) -> "KVCache":
        n_loops = model.moe.n_loops if n_loops is None else int(n_loops)
        return cls(
            num_layers=len(model.gemma_decoder.layers),
            n_loops=n_loops,
            num_attn_slots=model.moe.first_mlp_index,
        )
