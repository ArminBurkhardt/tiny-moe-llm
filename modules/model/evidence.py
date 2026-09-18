"""Retrieved evidence, as the MoE block reads it.

The block already has a per-call injection port: ``other``, re-read at every loop, which today
carries this model's own per-token embeddings (``TinyMoETransformer._moe_ple``). Evidence replaces
what that port carries, and the swap is not a content swap alone -- evidence is a *different set of
tokens* than the queries, so three things stop being shared:

  * **segmentation.** ``varlen_attention`` pinned the key length to the query length by passing one
    ``cu_seqlens`` for both sides. Evidence needs its own, one evidence segment per query segment,
    paired by position: document *i* of the batch reads evidence set *i* and nothing else.
  * **position basis.** Each retrieved chunk restarts at position 0, so within-chunk order survives
    (a copied span still reads left to right) while cross-chunk geometry is absent -- two chunks
    from different documents have no relative position, and giving them one invents an ordering the
    retriever never meant.
  * **causality.** A retrieved passage is not before or after the token reading it, so the read is
    bidirectional over the evidence and the query side keeps its own causal order intact.

**With no evidence attached the forward pass is bit-identical to the model without this module.**
That is the property that lets one checkpoint serve both modes and makes the replay fraction of a
finetune genuinely protect the trunk rather than train the port, so nothing here may touch a code
path that a ``None`` evidence batch reaches.
"""
from dataclasses import dataclass

import torch


@dataclass
class EvidenceBatch:
    """One batch's retrieved evidence, already embedded and segmented.

    Built by ``TinyMoETransformer.build_evidence`` and handed down unchanged; the MoE block reads it
    at every loop, which is what makes the evidence re-read rather than consumed once.

    Attributes:
        states: ``[B, S_ev, H]`` evidence tokens in the block's hidden space.
        cu_seqlens: int32 ``[num_segments + 1]`` over the flattened ``B * S_ev`` axis. **Must carry
            the same number of segments as the query side's** -- flash pairs them by position, so a
            mismatch silently points a document at another document's evidence rather than failing.
        max_seqlen: longest evidence segment, an upper bound is fine (same contract as the query
            side, and for the same no-host-sync reason).
        position_embeddings: ``(cos, sin)`` for the evidence axis, gathered at the per chunk
            positions rather than sliced from 0.
        chunk_ids: ``[B, S_ev]``, which retrieved chunk each evidence token came from. Read by the
            selector to gate a chunk's tokens by its relevance, and by nothing else.
        chunk_keys: ``[B, num_chunks, embed_dim]`` external embedder vectors, one per retrieved
            chunk -- the selector's side of the port. ``None`` when only the reader is attached.
    """

    states: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    chunk_ids: torch.Tensor = None
    chunk_keys: torch.Tensor = None

    @property
    def num_tokens(self) -> int:
        return self.states.shape[1]


def chunk_position_ids(chunk_ids: torch.Tensor) -> torch.Tensor:
    """Per chunk positions restarting at 0, from a ``[B, S_ev]`` chunk id tensor.

    Position within a chunk is the token's offset from that chunk's first token. Computed by
    subtracting a running start rather than by a Python loop over chunks: the chunk count is
    data dependent, and reading it on the host would cost a sync in the step path.
    """
    B, S = chunk_ids.shape
    device = chunk_ids.device
    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    # a chunk starts where its id differs from the previous token's; cummax of the start positions
    # carries each chunk's own start forward across its tokens
    starts = torch.zeros_like(pos)
    starts[:, 1:] = torch.where(
        chunk_ids[:, 1:] != chunk_ids[:, :-1], pos[:, 1:], torch.zeros_like(pos[:, 1:])
    )
    return pos - torch.cummax(starts, dim=1).values


def evidence_cu_seqlens(segment_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``cu_seqlens`` over a ``[B, S_ev]`` evidence segment id tensor.

    Deliberately the same construction as ``attention.cu_seqlens_from_doc_ids`` -- including forcing
    a break at every row seam -- so the two sides' segment numbering agrees by position. It is a
    separate function only because the evidence axis is a different length than the query axis.
    """
    B, S = segment_ids.shape
    device = segment_ids.device
    flat = segment_ids.reshape(-1)
    pos = torch.arange(B * S, device=device)
    boundary = torch.ones(B * S, dtype=torch.bool, device=device)
    boundary[1:] = flat[1:] != flat[:-1]
    boundary |= (pos % S == 0)
    starts = boundary.nonzero().flatten()
    ends = torch.cat([starts[1:], torch.tensor([B * S], device=device, dtype=starts.dtype)])
    cu = torch.zeros(starts.numel() + 1, dtype=torch.int32, device=device)
    cu[1:] = (ends - starts).cumsum(0).to(torch.int32)
    return cu, S
