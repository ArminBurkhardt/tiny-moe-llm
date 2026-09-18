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

A document that retrieved **nothing** is the fourth case and needs no special handling: it
contributes a zero length evidence segment, and flash returns exact zeros for a query whose segment
has no keys (measured, not assumed -- the SDPA fallback is made to agree in ``attention.py``). So
"no evidence" costs no sentinel token and no placeholder chunk, and reads as exactly the absence it
is rather than as an empty string the model has to learn to ignore.

**With no evidence attached the forward pass is bit-identical to the model without this module.**
That is the property that lets one checkpoint serve both modes and makes the replay fraction of a
finetune genuinely protect the trunk rather than train the port, so nothing here may touch a code
path that a ``None`` evidence batch reaches.
"""
from dataclasses import dataclass

import torch

from modules.model.attention import _default_cu_seqlens, _segment_ids  # noqa: F401 (evidence_memory)


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
        chunk_ids: ``[B, S_ev]``, which retrieved chunk each evidence token came from. Read **only**
            to restart positions at each chunk boundary, so any numbering works as long as adjacent
            chunks differ; evidence padding carries -1 and simply forms one more run.
        chunk_keys: ``[num_chunks, embed_dim]`` external embedder vectors, one per retrieved chunk
            -- the selector's side of the port. ``None`` when only the reader is attached.
        chunk_segments: ``[num_chunks]`` which query segment each chunk serves, in the query side's
            segment numbering. Supplied by whoever built the batch rather than re-derived here: the
            reader's token stream and the selector's chunk list come out of the same packing loop,
            and that loop is the only place both are known at once. Re-deriving it from the token
            stream would look safer and would in fact be a second opinion that can disagree.
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

    **Row padding is not a chunk and is given position 0.** It arrives as one contiguous run of the
    negative id, so treating it as a chunk would number it 0..(padding length), and the padding run
    is as long as the row's evidence budget -- far past the rotary cache, which is sized for the
    model's context. That gather is unchecked, so the result is a device side assert: asynchronous,
    so it surfaces as the process wedging with the GPU idle rather than as an exception anyone can
    read. Padding is only ever attended to by the trailing pad query segment, whose output nothing
    reads, so any in-range position is correct and 0 is the cheapest.
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
    within = pos - torch.cummax(starts, dim=1).values
    return torch.where(chunk_ids >= 0, within, torch.zeros_like(within))


def evidence_cu_seqlens(segment_ids: torch.Tensor, num_segments: int) -> tuple[torch.Tensor, int]:
    """``cu_seqlens`` for the evidence axis, from a ``[B, S_ev]`` map of token -> query segment.

    **Counted per segment, not derived from runs of equal ids.** The evidence side has segments the
    query side does not: a document that retrieved nothing contributes ZERO evidence tokens, and a
    run based construction cannot express a zero length segment at all -- it would simply omit it.
    Omitting one is not a shorter list, because flash pairs the two sides *by position*: every later
    document would silently read the document before it. The count is over ``num_segments``, which
    comes from the query side, so the two lists are the same length by construction rather than by
    the data happening to cooperate.

    Requires the evidence tokens to be laid out sorted by segment over the flattened ``B * S_ev``
    axis, which is what the dataset writes (row major, and within a row in conversation order).

    Args:
        segment_ids: ``[B, S_ev]``, each entry the index of the query segment that token serves.
        num_segments: ``len(query cu_seqlens) - 1``.

    Returns:
        ``(cu_seqlens, max_seqlen)``. ``max_seqlen`` is ``S_ev``, a valid upper bound -- the true
        maximum would cost a host sync every step, same trade as the query side's.
    """
    flat = segment_ids.reshape(-1)
    counts = torch.zeros(num_segments, dtype=torch.long, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    cu = torch.zeros(num_segments + 1, dtype=torch.int32, device=flat.device)
    cu[1:] = counts.cumsum(0).to(torch.int32)
    return cu, int(segment_ids.shape[1])


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
