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

from modules.model.attention import _default_cu_seqlens, _segment_ids


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
        chunk_ids: ``[B, S_ev]``, which retrieved chunk each evidence token came from. Drives the
            per chunk position restart, and indexes ``chunk_keys``. **Numbered globally over the
            batch**, not per row: it is the join between the reader's token axis and the selector's
            chunk axis, and a per-row numbering would make chunk 0 of row 0 and chunk 0 of row 1
            index the same key.
        chunk_keys: ``[num_chunks, embed_dim]`` external embedder vectors, one per retrieved chunk
            -- the selector's side of the port. ``None`` when only the reader is attached.
        chunk_segments: ``[num_chunks]`` which query segment each chunk serves, in the same
            numbering the reader's ``cu_seqlens`` pairing produces. Derived rather than supplied, so
            the selector and the reader cannot disagree about which document owns a chunk.
    """

    states: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    chunk_ids: torch.Tensor = None
    chunk_keys: torch.Tensor = None
    chunk_segments: torch.Tensor = None

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


def chunk_segment_ids(chunk_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                      num_chunks: int) -> torch.Tensor:
    """Which query segment each retrieved chunk serves, ``[num_chunks]``.

    Derived from the evidence side's own ``cu_seqlens`` rather than taken from the caller, so the
    selector's notion of "this chunk belongs to document *i*" is by construction the same one the
    reader's segment pairing uses. Two ways of saying which document owns a chunk is two ways for
    them to disagree, and the disagreement is silent -- the reader would attend to the right
    passage while the selector scored a different document's.

    Requires ``chunk_ids`` to be numbered globally over the batch (see ``EvidenceBatch``): the
    scatter indexes ``chunk_keys`` directly. Every token of a chunk writes its own segment, which is
    the same value for all of them, so the duplicate writes are not order dependent.
    """
    B, S = chunk_ids.shape
    token_segment = _segment_ids(cu_seqlens, B, S, chunk_ids.device).reshape(-1)
    out = torch.zeros(num_chunks, dtype=token_segment.dtype, device=chunk_ids.device)
    return out.scatter_(0, chunk_ids.reshape(-1), token_segment)


def evidence_memory(evidence: EvidenceBatch, cu_seqlens: torch.Tensor, B: int, S: int, device):
    """The selector's view of the corpus: ``(chunk_keys [M, embed_dim], visible [B*S, M])``.

    The reader gets its isolation from flash's positional pairing of two ``cu_seqlens``; the
    selector scores a dense ``[tokens, chunks]`` matrix instead, so it needs the pairing written out
    as a mask. Both are built from the same segment numbering, which is what keeps them consistent.

    Dense over the whole batch's chunks rather than gathered per row: ``M`` is tens, so the masked
    out entries cost ~2% of what the parametric candidate scoring already costs, and a flat chunk
    axis is the shape an append only buffer grows along.

    Returns None when there is nothing to select over, which is the signal the IR module uses to
    take its original read path unchanged.
    """
    if evidence is None or evidence.chunk_keys is None or evidence.chunk_segments is None:
        return None
    if cu_seqlens is None:
        cu_seqlens = _default_cu_seqlens(B, S, device)
    token_segment = _segment_ids(cu_seqlens, B, S, device).reshape(-1)          # [B*S]
    visible = evidence.chunk_segments.unsqueeze(0) == token_segment.unsqueeze(1)
    return evidence.chunk_keys, visible
