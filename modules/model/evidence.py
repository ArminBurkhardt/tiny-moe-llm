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
import dataclasses
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from modules.model.attention import _default_cu_seqlens, _segment_ids  # noqa: F401 (evidence_memory)
from modules.model.information_retrieval import _group_starts
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm


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
        chunk_gold: ``[num_chunks]`` bool/uint8, the corpus's per chunk gold flag, aligned with
            ``chunk_keys``/``chunk_segments`` exactly -- same axis, same order, same slots. ``None``
            whenever the corpus predates the gold flag or only the reader is attached (nothing in
            the model reads this itself; it rides along purely so a trainer computing
            ``information_retrieval.evidence_selection_loss`` has it at the same axis the module's
            own weights and ``evidence_memory``'s ``visible`` mask use, rather than having to
            re-derive the ``chunk_keys`` flattening a second time from the raw batch).
    """

    states: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    chunk_ids: torch.Tensor = None
    chunk_keys: torch.Tensor = None
    chunk_segments: torch.Tensor = None
    chunk_gold: torch.Tensor = None

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


def chunk_mean_mass(weights: torch.Tensor, visible: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``[T, M]`` per (query token, chunk) external read mass -> ``[M]``, meaning per chunk.

    The mean is taken over the chunk's OWN document, i.e. over ``visible``'s True entries, not over
    every token in the batch: ``_memory_scores`` already forces a non visible token's weight near
    zero, but averaging over all ``T`` tokens regardless would still let a longer document dilute a
    shorter one's chunk scores for no reason connected to relevance.

    Used by both halves this feeds: the reader's per chunk gate (segment level, see
    ``scatter_chunk_score_to_tokens``) and, for a trainer that wants to supervise it, the selection
    loss (``information_retrieval.evidence_selection_loss`` reads the un-aggregated per token
    weights directly, this function is for the READER side which only needs one number per chunk).
    """
    visible_f = visible.to(weights.dtype)
    denom = visible_f.sum(dim=0).clamp_min(eps)
    return (weights * visible_f).sum(dim=0) / denom


# cap on how many chunks one query segment (document) may hold, purely to size the lookup table
# below without a host sync -- the true per segment count is data dependent (a batch's own
# documents decide it) and reading it to size a tensor would need `.max().item()` every forward.
# 256 matches the largest buffer size anything in this project's plan asks a reader to hold (a
# needle style eval over that many chunks); a document with more chunks than this shares its
# overflow chunks' lookup slot with whichever chunk already has that slot; RANK, so no ranks are
# ever completely skipped.
MAX_CHUNKS_PER_SEGMENT = 256


def _chunk_rank_by_segment(chunk_segments: torch.Tensor, num_segments: int) -> torch.Tensor:
    """0-indexed rank of each chunk WITHIN its segment's own block, in original array order.

    This is the half of the reconciliation that starts from the SELECTOR's flat, whole batch chunk
    list (``chunk_segments``, one entry per row of ``chunk_keys``); ``_evidence_chunk_rank`` below
    computes the matching rank from the READER's per token axis. The two lists are guaranteed to
    agree on ORDER -- "the reader's token stream and the selector's chunk list come out of the same
    packing loop" (see ``EvidenceBatch.chunk_segments``'s docstring) -- never on the actual numbers,
    which is why this is a stable sort rather than a lookup by value.
    """
    order = torch.argsort(chunk_segments, stable=True)
    seg_sorted = chunk_segments[order]
    starts = _group_starts(seg_sorted, num_segments)
    rank_sorted = torch.arange(chunk_segments.numel(), device=chunk_segments.device) - starts[seg_sorted]
    rank = torch.empty_like(rank_sorted)
    rank[order] = rank_sorted
    return rank


def _evidence_chunk_rank(evidence: "EvidenceBatch") -> torch.Tensor:
    """0-indexed rank of each evidence TOKEN's chunk, within its own query segment. ``[B, S_ev]``.

    Deliberately not ``evidence.chunk_ids``' raw value: that field's own docstring only promises
    adjacent chunks differ (and the encoder's counter resets per packed ROW, which can span several
    query segments), so its numbers do not by themselves line up with ``chunk_segments``' per
    segment blocks. What both sides ARE guaranteed to agree on is order, so this counts RUN
    boundaries -- a chunk id differing from its predecessor, or a new query segment starting --
    rather than trusting the id values, and resets the count at every query segment boundary
    (unlike the raw ``chunk_ids``, which only resets at row boundaries).
    """
    B, S_ev = evidence.chunk_ids.shape
    device = evidence.chunk_ids.device
    flat_ids = evidence.chunk_ids.reshape(-1)
    token_seg = _segment_ids(evidence.cu_seqlens, B, S_ev, device).reshape(-1)

    boundary = torch.ones(B * S_ev, dtype=torch.long, device=device)
    if B * S_ev > 1:
        changed = (flat_ids[1:] != flat_ids[:-1]) | (token_seg[1:] != token_seg[:-1])
        boundary[1:] = changed.long()
    run_number = torch.cumsum(boundary, dim=0) - 1  # global run index over the flattened axis

    num_segments = evidence.cu_seqlens.numel() - 1
    # the run index AT the start of each segment, gathered from cu_seqlens' own start offsets --
    # segments are laid out contiguously on this axis by construction (evidence_cu_seqlens' own
    # requirement), so a segment's first token IS the row cu_seqlens points at. An empty segment's
    # clamped index reads some other segment's run number, but an empty segment owns no tokens to
    # rank, so nothing ever looks that value up.
    seg_starts = evidence.cu_seqlens[:-1].long().clamp(max=max(B * S_ev - 1, 0))
    run_at_seg_start = run_number.index_select(0, seg_starts)
    rank = run_number - run_at_seg_start.index_select(0, token_seg)
    return rank.view(B, S_ev)


def scatter_chunk_score_to_tokens(
    evidence: "EvidenceBatch", chunk_score: torch.Tensor, max_chunks_per_segment: int = MAX_CHUNKS_PER_SEGMENT
) -> torch.Tensor:
    """Broadcast one scalar per retrieved chunk onto every evidence token that chunk contributed.

    ``chunk_score``: ``[M]``, aligned with ``evidence.chunk_segments``/``chunk_keys``'s own axis.
    Returns ``[B, S_ev]``. Padding (``chunk_ids < 0``) reads 0 -- multiplying an already isolated
    padding state by anything does not change what the reader's real tokens see, so the value there
    is unobserved, not meaningful.

    Reconciles the two chunk numberings (see ``_chunk_rank_by_segment`` / ``_evidence_chunk_rank``)
    through a FIXED size, per segment lookup table rather than a data dependent one, purely to avoid
    a host sync sizing it: a segment with more than ``max_chunks_per_segment`` chunks folds its
    overflow onto the last slot instead of raising, which biases that one slot's gate rather than
    reading garbage or crashing.
    """
    device = evidence.chunk_ids.device
    B, S_ev = evidence.chunk_ids.shape
    num_segments = evidence.cu_seqlens.numel() - 1

    chunk_rank = _chunk_rank_by_segment(evidence.chunk_segments, num_segments).clamp(
        max=max_chunks_per_segment - 1
    )
    lut = chunk_score.new_zeros(num_segments * max_chunks_per_segment)
    lut_index = evidence.chunk_segments * max_chunks_per_segment + chunk_rank
    lut.scatter_(0, lut_index, chunk_score.to(lut.dtype))

    token_seg = _segment_ids(evidence.cu_seqlens, B, S_ev, device).view(-1)
    token_rank = _evidence_chunk_rank(evidence).view(-1).clamp(min=0, max=max_chunks_per_segment - 1)
    gathered = lut.index_select(0, token_seg * max_chunks_per_segment + token_rank).view(B, S_ev)
    return torch.where(evidence.chunk_ids >= 0, gathered, torch.zeros_like(gathered))


def apply_chunk_gate(evidence: "EvidenceBatch", chunk_gate: torch.Tensor) -> "EvidenceBatch":
    """Multiply the evidence states by a per chunk gate, before the reader's ``k_proj``/``v_proj``.

    ``chunk_gate``: ``[M]``, one multiplicative factor per entry of ``chunk_segments``/
    ``chunk_keys``'s axis (typically ``1 + scale * sigmoid(selector chunk score)``, see
    ``LoopMixtureOfExperts.forward_step``). Structural rather than a mask on the attention itself:
    flash varlen attention takes no additive bias, so gating the STATES the reader's projections
    read is the only cheap way to make a chunk's content responsive to how relevant the selector
    found it, without touching the reader's own segmentation or causality.

    Returns a NEW ``EvidenceBatch`` (states replaced, everything else shared) rather than mutating
    in place -- the original is still what the selector itself scored this call and any other reader
    of it should keep seeing.
    """
    token_gate = scatter_chunk_score_to_tokens(evidence, chunk_gate)
    gated_states = evidence.states * token_gate.unsqueeze(-1).to(evidence.states.dtype)
    return dataclasses.replace(evidence, states=gated_states)


class GroundednessHead(nn.Module):
    """Predicts "a gold chunk is present AND the row is answerable" from the reader's OWN output.

    The label is external and corpus known, not the model's own argmax the way the deleted
    correctness head's BCE target was (see CLAUDE.md's Phase 0 section on why that failed: BCE
    against ``lm_head``'s own most likely token made "reproduce ``p_max``" the reachable optimum by
    construction, and that is what it learned). This head's target is a fact about the CORPUS --
    whether a gold chunk was ever put in front of the model and whether the row actually has an
    answer -- which the model's own logits cannot hand it for free, so it cannot collapse the same
    way.

    Reads the evidence reader's own output at a position (e.g. ``shared_evidence``'s output before
    it is scaled and added into the loop's accumulator), not the block's whole residual stream:
    the question this asks is "does what the reader retrieved grounds an answer", a property of the
    read alone, not of everything else that step's loop accumulated. Not wired into the model's
    forward -- a trainer attaches it and picks the position(s) to read, since that choice (last
    prompt token, every supervised position, ...) depends on the corpus a script is training on.

    Zero-init output projection: the same neutrality pattern as every other head or gate added to
    this model after training started (see ``moe.is_fresh_loop_param``) -- attaching this to an
    already trained checkpoint costs nothing until it is actually trained.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, reader_output: torch.Tensor) -> torch.Tensor:
        """``reader_output``: ``[..., hidden_size]``. Returns the raw logit (``[...]``, last dim
        squeezed) -- pass it through ``torch.sigmoid`` for a probability or straight into
        ``groundedness_loss``/``F.binary_cross_entropy_with_logits`` for training.
        """
        return self.out_proj(self.dropout(self.norm(reader_output))).squeeze(-1)


def groundedness_loss(
    logits: torch.Tensor, gold_present: torch.Tensor, answerable: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """BCE against "a gold chunk is present AND the row is answerable" -- the non leaky label.

    Both ``gold_present`` and ``answerable`` are corpus known (the same ``chunk_gold``/row level
    answerability the corpus builder already records) and neither is read off this model's own
    output, which is what keeps this from collapsing onto ``p_max`` the way the deleted correctness
    head did -- see ``GroundednessHead``'s docstring.

    Args:
        logits: ``[...]``, ``GroundednessHead``'s raw output at whichever positions a trainer chose.
        gold_present: same shape, 1.0/True where the row's evidence buffer held a gold chunk.
        answerable: same shape, 1.0/True where the row's question is actually answerable.
        mask: optional same shape mask (e.g. restricting the loss to one position per document);
            every given row counts equally when omitted.

    Returns:
        scalar loss, 0 (with no grad breaking discontinuity) when ``mask`` zeroes every row.
    """
    target = gold_present.to(logits.dtype) * answerable.to(logits.dtype)
    per_row = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is None:
        return per_row.mean()
    mask = mask.to(per_row.dtype)
    return (per_row * mask).sum() / mask.sum().clamp_min(1e-6)


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
