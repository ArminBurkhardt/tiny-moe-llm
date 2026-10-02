import torch
from torch import nn
import torch.nn.functional as F

from modules.model.moe import LoopMixtureOfExperts
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextModel
from modules.model.modules import SmallLMHead
from modules.model.mtp import MTPHead
from modules.model.evidence import (
    EvidenceBatch, GroundednessHead, chunk_position_ids, evidence_cu_seqlens, groundedness_loss,
)
from modules.model.attention import cu_seqlens_from_doc_ids
from utils import logger

# NOTE: use Transformer Engines checkpoint, not torch.utils.checkpoint for FP8/NVFP4
from transformer_engine.pytorch import checkpoint


class TokenTracker():
    """Counts trained tokens without forcing a host sync on every forward.

    When ``pad_token_id`` is set the per step non pad count is a reduction over a CUDA tensor;
    reading it with ``.item()`` every forward would drain the stream and serialize CPU/GPU. Instead
    the increments accumulate into an on-device scalar and are only pulled to the host on ``sync()``
    (called at log/checkpoint cadence). ``num_tokens`` stays readable/writable as a plain int so
    existing call sites (resume, dry run save/restore) keep working.
    """
    def __init__(self):
        self._cached = 0           # host-side total as of the last sync()
        self._device_count = None  # pending on-device increments not yet drained to the host
        self.pad_token_id = None   # when set, padding tokens are excluded from the count

    def count_tokens(self, input_ids: torch.Tensor):
        if self.pad_token_id is None:
            # numel() is a python int already -> no sync
            self._cached += input_ids.numel()
            return
        # keep the reduction on-device and accumulate; no host transfer here
        n = (input_ids != self.pad_token_id).sum()
        if self._device_count is None or self._device_count.device != n.device:
            self._device_count = torch.zeros((), dtype=torch.long, device=n.device)
        self._device_count += n

    def sync(self):
        """Drain pending on-device counts into the host total. The only host sync; call it at
        logging/checkpoint cadence rather than every step."""
        if self._device_count is not None:
            self._cached += int(self._device_count.item())
            self._device_count.zero_()
        return self._cached

    def reset(self):
        self._cached = 0
        if self._device_count is not None:
            self._device_count.zero_()

    def get_count(self):
        # sync-free read; may lag real-time by up to one logging interval of pending tokens
        return self._cached

    @property
    def num_tokens(self):
        return self._cached

    @num_tokens.setter
    def num_tokens(self, value):
        # explicit set (resume / dry run restore) replaces the host total and clears any pending
        self._cached = int(value)
        if self._device_count is not None:
            self._device_count.zero_()

class TinyMoETransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        max_seq_len: int, 
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_layers: int,
        num_heads: int,
        num_mlp_experts: int,
        num_attn_experts: int,
        top_k: int = 2,
        n_loops: int = 4,
        num_ir_experts: int = 1,
        num_ir_entries: int = 8192,
        ir_dim: int = 256,
        ir_num_clusters: int = 0,
        ir_probe_clusters: int = 4,
        ir_read_top_k: int = 32,
        dropout: float = 0.1,
        ple_embeddings_size: int = None,
        mtp_num_extra_tokens: int = 0,
        lm_head_factor: int = 8,
        moe_intermediate_size: int = None,
        loop_inject: bool = False,
        evidence_port: bool = False,
        evidence_encoder: bool | int = True,
        ir_direct_read: bool = True,
        groundedness_head: bool = False,
        evidence_reader_rotary: bool = True,
    ):
        super().__init__()

        # how build_evidence embeds retrieved evidence, resolved once here rather than re-branched
        # on every call. True/None -> every decoder layer; an explicit int -> only the first N
        # (0 counts as "every layer", so an absent yaml key and an explicit 0 mean the same thing);
        # False -> skip the decoder entirely and fall back to the original per-token PLE embedding.
        # That fallback exists for the encoded-vs-raw comparison this flag was added to make
        # possible, not because it is a design worth keeping -- see _encode_evidence.
        if evidence_encoder is False:
            self.evidence_encoder_layers = None
        elif evidence_encoder is True or evidence_encoder is None:
            self.evidence_encoder_layers = 0
        else:
            self.evidence_encoder_layers = int(evidence_encoder)

        # construction-time invariants (PLAN.md Step 5) -- SmallLMHead chunks both dims into
        # `factor` pieces, so a bad vocab/hidden/lm_head_factor combo would silently truncate
        # instead of raising; catch it here instead of at the first forward.
        assert vocab_size % lm_head_factor == 0, (
            f"vocab_size ({vocab_size}) must be divisible by lm_head_factor ({lm_head_factor})"
        )
        assert hidden_size % lm_head_factor == 0, (
            f"hidden_size ({hidden_size}) must be divisible by lm_head_factor ({lm_head_factor})"
        )
        if mtp_num_extra_tokens > 0:
            # MTPHead's own SmallLMHead runs on hidden_size//2 with lm_head_factor*2 (see
            # mtp_head construction below) -- same chunking constraint, different dims/factor.
            mtp_lm_head_factor = lm_head_factor * 2
            assert vocab_size % mtp_lm_head_factor == 0, (
                f"vocab_size ({vocab_size}) must be divisible by lm_head_factor*2 ({mtp_lm_head_factor}) for MTP"
            )
            assert (hidden_size // 2) % mtp_lm_head_factor == 0, (
                f"hidden_size//2 ({hidden_size // 2}) must be divisible by lm_head_factor*2 ({mtp_lm_head_factor}) for MTP"
            )
        # uint16 fit for Step 8's train.bin dtype -- not a hard architectural limit, just the
        # data pipeline's contract.
        assert vocab_size <= 65536, f"vocab_size ({vocab_size}) exceeds 65536 (Step 8's train.bin is uint16)"

        # routed + shared MoE experts only -- Gemma4TextModel below keeps plain intermediate_size
        moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size

        self.gemma_decoder = Gemma4TextModel(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads // 4, # for GQA
            intermediate_size=intermediate_size,
            dropout=dropout,
            per_layer_embeddings_size=ple_embeddings_size,
        )
        
        import transformer_engine.pytorch as te
        self.moe_embeddings = nn.Embedding(vocab_size, ple_embeddings_size) if ple_embeddings_size is not None else None
        self.moe_embed_proj = te.Linear(ple_embeddings_size, hidden_size, bias=False) if ple_embeddings_size is not None else None
        self.moe = LoopMixtureOfExperts(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            num_mlp_experts=num_mlp_experts,
            num_attn_experts=num_attn_experts,
            num_ir_experts=num_ir_experts,
            num_ir_entries=num_ir_entries,
            ir_dim=ir_dim,
            ir_num_clusters=ir_num_clusters,
            ir_probe_clusters=ir_probe_clusters,
            ir_read_top_k=ir_read_top_k,
            dropout=dropout,
            top_k=top_k,
            n_loops=n_loops,
            max_seq_len=max_seq_len,
            loop_inject=loop_inject,
            evidence_port=evidence_port,
            ir_direct_read=ir_direct_read,
            evidence_reader_rotary=evidence_reader_rotary,
        )

        self.norm = RMSNorm(hidden_size)
        self.lm_head = SmallLMHead(hidden_size, vocab_size, factor=lm_head_factor)

        # reads the evidence reader's own output, never the residual stream (see GroundednessHead).
        # Inferred from the state dict like the port itself, so a checkpoint that predates it keeps
        # loading and a migration is what adds it. Zero-init output, and nothing in forward() reads
        # this head -- a trainer picks the positions -- so attaching it changes no logit.
        self.groundedness_head = (
            GroundednessHead(hidden_size, dropout=dropout)
            if (groundedness_head and evidence_port) else None
        )

        self.mtp_head = MTPHead(
            hidden_size, 
            vocab_size,
            num_extra_tokens=mtp_num_extra_tokens, 
            dropout=dropout,
            lm_head_factor=lm_head_factor * 2, # reduce overhead
        ) if mtp_num_extra_tokens > 0 else None
        
        self.use_checkpointing = True
        self.use_sub_checkpointing = True

        self._token_tracker = TokenTracker()

        # param/FLOP accounting (PLAN.md Step 5) -- printed at construction so the budget math in
        # PLAN.md's Step 11 has a live number to check against instead of going stale silently.
        # "active" excludes the routed MLP experts' unused capacity: parallel_experts holds
        # num_mlp_experts worth of weights but only top_k/num_mlp_experts of them run per token
        # (every other expert in the pool -- attention/IR/shared -- runs densely every loop
        # regardless of routing, so it's already fully "active"). "excl. emb" further drops the
        # embedding-table lookups (embed_tokens, the dense decoder's PLE table, this model's own
        # PLE projection table) since they're memory lookups, not matmuls.
        total_params = sum(p.numel() for p in self.parameters())
        moe_params = sum(p.numel() for p in self.moe.parameters())
        mlp_expert_params = sum(p.numel() for p in self.moe.parallel_experts.parameters())
        active_frac = top_k / num_mlp_experts
        moe_active_params = moe_params - mlp_expert_params + int(mlp_expert_params * active_frac)

        # the IR key/value table is the one place "2 x params" is not even approximately the
        # compute: 33.5M parameters that a two stage read scores ~1.5% of. Bill it from the module's
        # own arithmetic instead, or every MFU number in the run is off by more than a whole dense
        # decoder. (On the exact path this comes back out to the same 2 x params it replaces.)
        ir_table_params = sum(
            m.z_keys.numel() + m.y_values.numel() for m in self.moe.ir_modules
        )
        ir_flops_per_token = sum(m.flops_per_token for m in self.moe.ir_modules)
        # separate from moe_active_params on purpose: the table's weights ARE resident and read
        # every loop, so they stay in the reported active param count. This is only the FLOP proxy.
        moe_flop_params = moe_active_params - ir_table_params
        embed_params = self.gemma_decoder.embed_tokens.weight.numel()
        if self.gemma_decoder.ple is not None:
            embed_params += self.gemma_decoder.ple.weight.numel()
        if self.moe_embeddings is not None:
            embed_params += self.moe_embeddings.weight.numel()
        non_moe_params = total_params - moe_params - embed_params
        active_excl_emb = non_moe_params + moe_active_params
        active_params = active_excl_emb + embed_params

        # FLOP accounting, read by scripts/pretrain.py's MFU logging. Split into three pieces
        # because they do NOT scale together, and folding them into one per-token number is what
        # made the pre-fix estimate understate real compute by roughly 2x:
        #
        #  1. body -- dense decoder + MoE block matmuls. The MoE portion multiplies by n_loops
        #     (its weights are one shared module reused every loop, so the param count appears
        #     once but the compute happens n_loops times); the decoder runs once. Standard "2N"
        #     approximation, embeddings excluded (lookups, not matmuls).
        #  2. heads -- lm_head runs once PER LOOP (per-loop CE, PLAN.md Step 4a), not once, and
        #     the MTP head's own lm_head runs once per extra token. compute_mtp_loss chunk-
        #     checkpoints all of them, so they cost fwd + recompute + bwd (4x) while the body
        #     (checkpointing off) costs 3x. Exposed per-application so the trainer can weight
        #     lm_head by the actual number of supervised loops (loop_ce_subsample).
        #  3. attention -- scales with sum(segment_len^2), not with token count, so it cannot be a
        #     per-token constant at all under document packing. Exposed as a coefficient the
        #     trainer multiplies by the packing structure it actually saw. Per attention layer and
        #     causal segment of length L the two matmuls cost 2 * hidden_size * L^2 (H heads x
        #     head_dim = hidden_size, L^2/2 attended pairs, 2 FLOPs per MAC, twice for QK^T + AV).
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        if self.mtp_head is not None:
            mtp_lm_head_params = sum(p.numel() for p in self.mtp_head.lm_head.parameters())
            mtp_body_params = sum(p.numel() for p in self.mtp_head.parameters()) - mtp_lm_head_params
        else:
            mtp_lm_head_params, mtp_body_params = 0, 0
        body_params = non_moe_params - lm_head_params - mtp_lm_head_params - mtp_body_params

        # 1 shared_attn + (self + cross) per attn expert + 1 per IR expert, every loop
        moe_attn_per_loop = 1 + 2 * num_attn_experts + num_ir_experts
        n_attn_layers = num_layers + n_loops * moe_attn_per_loop

        # split per-loop from run-once so the trainer can bill the loop count it ACTUALLY ran --
        # loop-count sampling (a step may run fewer than n_loops) would otherwise silently inflate
        # the reported MFU by charging every step for the full depth.
        self.dense_flops_per_token = 2 * body_params                 # decoder + heads' trunk, once
        # per MoE loop: dense matmuls at 2N, plus the IR table's own measured retrieval cost
        self.loop_flops_per_token = 2 * moe_flop_params + ir_flops_per_token
        self.dense_attn_flops_per_seqsq = 2 * hidden_size * num_layers
        self.loop_attn_flops_per_seqsq = 2 * hidden_size * moe_attn_per_loop
        self.lm_head_flops_per_token = 2 * lm_head_params            # per application (once per loop)
        self.mtp_flops_per_token = 2 * (mtp_body_params + mtp_num_extra_tokens * mtp_lm_head_params)

        # evidence encoding: linear in the number of EVIDENCE tokens, which is a property of the
        # batch (a buffer that can hold zero chunks or thousands), not of max_seq_len -- exactly the
        # reason the external IR memory read above has no fixed per-token constant either. Folding
        # it into flops_per_token_fwd below would misreport every step that doesn't happen to run at
        # the corpus's average evidence count, so it is exposed as its own per-evidence-token
        # coefficient instead, mirroring attn_flops_per_seqsq's contract: whoever has the batch's
        # real evidence token count multiplies by it to get the true cost. Standard "2N" per layer,
        # embeddings excluded (a lookup, not a matmul); num_layers assumed uniform in shape, true for
        # every config this model runs.
        if num_layers > 0:
            decoder_layer_params = sum(p.numel() for p in self.gemma_decoder.layers[0].parameters())
        else:
            decoder_layer_params = 0
        # meaningless (and never run) without the port at all -- build_evidence short-circuits to
        # None the moment self.moe.shared_evidence is None, so _encode_evidence never executes
        if self.moe.shared_evidence is None or self.evidence_encoder_layers is None:
            encoder_layers_run = 0
        elif self.evidence_encoder_layers == 0:
            encoder_layers_run = num_layers
        else:
            encoder_layers_run = min(self.evidence_encoder_layers, num_layers)
        self.evidence_encoder_flops_per_token = 2 * decoder_layer_params * encoder_layers_run

        # aggregates at the configured loop count, for the log line / anything not tracking depth
        self.body_flops_per_token = self.dense_flops_per_token + n_loops * self.loop_flops_per_token
        self.attn_flops_per_seqsq = self.dense_attn_flops_per_seqsq + n_loops * self.loop_attn_flops_per_seqsq

        # single representative number for the log line: one forward, every loop's lm_head, and
        # attention at a fully-packed max_seq_len (a single max_seq_len document per row, the
        # worst case -- real packing splits it into shorter segments and costs less).
        flops_per_token = (
            self.body_flops_per_token
            + n_loops * self.lm_head_flops_per_token
            + self.mtp_flops_per_token
            + self.attn_flops_per_seqsq * max_seq_len   # sum(L^2)/tokens == max_seq_len when L == max_seq_len
        )
        self.flops_per_token_fwd = flops_per_token
        logger.info(
            f"params: total={total_params/1e6:.1f}M active={active_params/1e6:.1f}M "
            f"(excl. emb={active_excl_emb/1e6:.1f}M) | forward FLOP/token ~= {flops_per_token/1e6:.0f}M "
            f"(body {self.body_flops_per_token/1e6:.0f}M + heads "
            f"{(n_loops * self.lm_head_flops_per_token + self.mtp_flops_per_token)/1e6:.0f}M + attn "
            f"{self.attn_flops_per_seqsq * max_seq_len/1e6:.0f}M @ seq_len={max_seq_len})"
        )
        if self.evidence_encoder_flops_per_token > 0:
            # NOT part of the figure above on purpose -- see the comment where this is computed
            logger.info(
                f"evidence encoder: ~= {self.evidence_encoder_flops_per_token/1e6:.1f}M/evidence-token "
                f"over {encoder_layers_run} decoder layer(s), not counted in forward FLOP/token above "
                f"(scales with the batch's real evidence token count, not max_seq_len)"
            )
    
    @property
    def token_count(self):
        return self._token_tracker.get_count()
    
    def _mtp_forward(self, hidden_state: torch.Tensor, use_checkpointing: bool = False):
        if self.mtp_head is None:
            return None
        if use_checkpointing:
            extra_token_outputs = checkpoint(self.mtp_head, hidden_state, use_reentrant=False)
        else:
            extra_token_outputs = self.mtp_head(hidden_state)
        return extra_token_outputs
    
    def _moe_ple(self, input_ids: torch.Tensor):
        if self.moe_embeddings is None or self.moe_embed_proj is None:
            return None
        moe_embeds = self.moe_embeddings(input_ids)
        moe_embeds = self.moe_embed_proj(moe_embeds)
        return moe_embeds

    def _encode_evidence(self, evidence_ids: torch.Tensor, evidence_chunk_ids: torch.Tensor,
                         position_ids: torch.Tensor) -> torch.Tensor:
        """Embed evidence tokens and, unless disabled, run them through the dense decoder.

        A bag of per-token PLE embeddings can copy a token it already expects but cannot find "the
        token after *born in*" -- no decoder layer had touched it, so no key carries any context.
        Reusing the SAME embedding table and decoder layers that contextualize the real token stream
        fixes that with no new parameters: this is a forward-path choice, not a learned addition, so
        a checkpoint that predates it needs nothing migrated to use it.

        Attention here must never cross a retrieved chunk's own boundary, which is a FINER grouping
        than ``evidence_cu_seqlens`` builds for the reader (that one groups by query segment, so a
        document's several chunks share one segment). So this derives its own chunk-level
        ``cu_seqlens`` from ``evidence_chunk_ids`` via ``cu_seqlens_from_doc_ids`` -- the identical
        construction already used for the main token stream's document packing, and it fits here for
        the same reason: ``modules/data/evidence_dataset.py`` restarts its chunk-id counter at 0 for
        every row, so two different rows' chunks can share a numeric id, and forcing a boundary at
        every row start (which that function already does) is exactly what keeps them from merging.

        Positions restart at 0 per chunk (``chunk_position_ids``, computed by the caller so it is
        not derived twice) rather than counting globally over the packed evidence axis. RoPE's dot
        product depends only on the relative offset between two positions, so this is exactly
        equivalent to a global count AS LONG AS a chunk never attends past its own boundary -- which
        the cu_seqlens above guarantees -- and it is what keeps a long evidence axis (many short
        chunks concatenated end to end) from running past the rotary cache sized for one document's
        worth of positions.

        Args:
            evidence_ids: ``[B, S_ev]`` token ids of the retrieved chunks, packed end to end.
            evidence_chunk_ids: ``[B, S_ev]`` which chunk each token came from.
            position_ids: ``[B, S_ev]`` per chunk positions, from ``chunk_position_ids``.

        Returns:
            ``[B, S_ev, H]`` evidence states in the block's hidden space.
        """
        if self.evidence_encoder_layers is None:
            # legacy path: kept for the encoded-vs-raw comparison this flag exists to make, not
            # because uncontextualized embeddings are a design worth keeping on their own
            states = self._moe_ple(evidence_ids)
            assert states is not None, "the evidence reader needs the MoE embedding path to embed with"
            return states

        decoder = self.gemma_decoder
        B, S_ev = evidence_ids.shape
        hidden_states = decoder.embed_tokens(evidence_ids) * (decoder.hidden_size ** 0.5)
        hidden_states = decoder.dropout(hidden_states)

        chunk_cu_seqlens, chunk_max_seqlen = cu_seqlens_from_doc_ids(evidence_chunk_ids)
        position_embeddings = decoder.rotary_emb.gather(position_ids, hidden_states.dtype)

        if decoder.ple is not None:
            ple_emb = decoder.ple(evidence_ids)
            ple_emb = ple_emb.view(
                B, S_ev, -1, ple_emb.shape[-1] // len(decoder.layers)
            ).transpose(1, 2)
        else:
            ple_emb = None

        num_layers = len(decoder.layers)
        n_run = num_layers if self.evidence_encoder_layers == 0 else min(self.evidence_encoder_layers, num_layers)
        for i in range(n_run):
            hidden_states = decoder.layers[i](
                hidden_states,
                chunk_cu_seqlens,
                chunk_max_seqlen,
                position_embeddings,
                per_layer_embeddings=ple_emb[:, i] if ple_emb is not None else None,
                kv_cache=None,
            )
        # same final norm the full decoder applies -- a truncated depth still reads out through it,
        # matching how the MoE block's own readout reuses self.norm regardless of loops actually run
        return decoder.norm(hidden_states)

    def build_evidence(self, evidence_ids: torch.Tensor, evidence_chunk_ids: torch.Tensor,
                       evidence_segment_ids: torch.Tensor, num_segments: int,
                       chunk_keys: torch.Tensor = None, chunk_segments: torch.Tensor = None,
                       chunk_gold: torch.Tensor = None):
        """Pack retrieved evidence into the form the MoE block reads.

        Evidence tokens are embedded and contextualized through **this model's own** dense decoder
        (see ``_encode_evidence``), the same weights that contextualize the real token stream. That
        is deliberate: the reader has to be able to find "the token after *born in*", and a key only
        carries that if something has already looked at the chunk's own neighbouring tokens -- a raw
        per-token embedding cannot. An external embedder's chunk vector cannot substitute either: it
        is a 384-d summary of a whole passage with no span to copy, which is why the external
        embedder sits on the *selector* side (``chunk_keys``) and never on this one.

        Args:
            evidence_ids: ``[B, S_ev]`` token ids of the retrieved chunks, packed end to end.
            evidence_chunk_ids: ``[B, S_ev]`` which chunk each token came from. Drives the per chunk
                position restart, so two chunks that happen to be adjacent in the packing do not read
                as one continuous passage. Only adjacency matters, so evidence padding may carry -1.
            evidence_segment_ids: ``[B, S_ev]`` which *query* segment each evidence token serves, in
                the query side's own segment numbering, sorted over the flattened axis. Flash pairs
                the two sides by position, so a segment omitted here points every later document at
                the wrong evidence silently rather than raising.
            num_segments: how many segments the query side has (``len(cu_seqlens) - 1``). Passed in
                rather than inferred: a document that retrieved nothing contributes no evidence
                token to infer from, and that is the case the pairing must not lose.
            chunk_keys: ``[num_chunks, embed_dim]`` external embedder vectors, one per chunk.
                Optional -- the reader works without them, and without them the selector simply never
                sees an external store.
            chunk_segments: ``[num_chunks]`` which query segment each chunk serves. Required
                whenever ``chunk_keys`` is given.
            chunk_gold: ``[num_chunks]`` bool/uint8 gold flag, aligned with ``chunk_keys``/
                ``chunk_segments`` exactly. Purely a passenger -- nothing in the forward reads it,
                it only rides along so a trainer computing
                ``information_retrieval.evidence_selection_loss`` finds it on the same object as
                the weights and the ``visible`` mask that loss needs. Optional even when
                ``chunk_keys`` is given: a corpus built before the gold flag existed has none.

        Returns:
            An ``EvidenceBatch``, or None when this model has no evidence port.
        """
        if self.moe.shared_evidence is None:
            return None
        assert chunk_keys is None or chunk_segments is not None, (
            "chunk_keys without chunk_segments -- the selector would have no way to tell which "
            "document owns a chunk, and would score every document against every chunk"
        )
        # computed once and handed to the encoder too, rather than derived twice: the reader's own
        # RoPE (below) and the encoder's internal RoPE (inside _encode_evidence) are two different
        # attention modules with two different head dims, but they rotate the SAME per chunk
        # positions
        position_ids = chunk_position_ids(evidence_chunk_ids)
        states = self._encode_evidence(evidence_ids, evidence_chunk_ids, position_ids)
        cu_seqlens, max_seqlen = evidence_cu_seqlens(evidence_segment_ids, num_segments)
        cos, sin = self.moe.rotary_emb.gather(position_ids, states.dtype)
        return EvidenceBatch(
            states=states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=(cos, sin),
            chunk_ids=evidence_chunk_ids,
            chunk_keys=chunk_keys,
            chunk_segments=chunk_segments,
            chunk_gold=chunk_gold,
        )


    def groundedness_term(self, chunk_gold: torch.Tensor, answerable: torch.Tensor,
                          positions: torch.Tensor):
        """BCE on the reader's own output at ``positions``, against the corpus label. Or None.

        The label is "a gold chunk was in front of the model AND the row actually has an answer"
        (see ``groundedness_loss``), and both halves are facts about the corpus rather than about
        this model's output -- which is what keeps this from collapsing onto ``p_max`` the way the
        deleted correctness head did. They are also genuinely two axes: a natively unanswerable
        SQuAD row is built under the ``gold`` condition and still takes an abstention target, so
        ``answerable`` cannot be recovered from the evidence buffer's contents.

        ``gold_present`` is derived from the selector's own visibility mask rather than from the
        segment arithmetic a second time: ``last_memory_visible`` already says which chunks each
        token's document owns, so one matmul against the gold flag answers "does this token's
        document hold a gold chunk" on exactly the axis the reader ran on.

        Reads what the forward stashed (``moe.last_reader_output``), so it must be called right
        after that forward and only from one that ran WITHOUT gradient checkpointing -- the same
        constraint, for the same reason, as ``LoopMixtureOfExperts.evidence_selection_term``.

        Args:
            chunk_gold: ``[M]`` bool/float per chunk gold flag, on ``chunk_keys``' own axis.
            answerable: ``[B, S]`` 1 where the token's conversation has a real answer as its
                target, 0 where it has a refusal. Negative entries (row padding) are never read --
                ``positions`` excludes them.
            positions: ``[B, S]`` mask of the positions to score, chosen by the trainer (the last
                prompt token of each answer span, typically).

        Returns:
            Scalar loss, or None when this model has no head, this batch carried no evidence, or
            the corpus has no labels -- every one of which is "nothing to supervise", not zero.
        """
        reader = self.moe.last_reader_output
        visible = self.moe.last_memory_visible
        if (self.groundedness_head is None or reader is None or visible is None
                or chunk_gold is None or answerable is None):
            return None
        logits = self.groundedness_head(reader).reshape(-1)                      # [B * S]
        gold_present = visible.to(logits.dtype) @ chunk_gold.to(logits.dtype)    # [B * S]
        return groundedness_loss(
            logits,
            (gold_present > 0).to(logits.dtype),
            answerable.reshape(-1).clamp_min(0).to(logits.dtype),
            mask=positions.reshape(-1).to(logits.dtype),
        )

    def _convergence_exit(self, tol: float, min_loops: int):
        """Build an ``exit_check`` that stops looping once the READOUT stops moving.

        The parameter-free replacement for the deleted halt head. Two things make it work where a
        learned gate did not: it has no parameters to saturate, and it is measured in the *readout*
        rather than in ``||dh||``. That distinction is load-bearing -- a migrated checkpoint's
        ``loop_scale`` still injects a sizeable hidden-state delta on the last loop while the
        predicted distribution is already stationary, so a hidden-state criterion would never fire.

        Evaluated at the **last position only**: one token x vocab, which is free next to a loop of
        the MoE block, and the only position generation actually reads.

        Args:
            tol: stop when the top-1's log-probability moves less than this between consecutive
                loops AND the top-1 token itself is unchanged. ``scripts/eval_calibration.py``
                prints both quantities per transition, which is how this gets picked.
            min_loops: never exit before this many loops have run (1-indexed count).

        Returns:
            A callable suitable for ``LoopMixtureOfExperts.forward``'s ``exit_check``.
        """
        state = {"token": None, "logprob": None}

        def check(loop_idx: int, hidden_states: torch.Tensor) -> bool:
            # [B, 1, H] -> [B, vocab]; self.norm because lm_head only ever reads normed states
            logits = self.lm_head(self.norm(hidden_states[:, -1:, :])).squeeze(1).float()
            logprobs = logits.log_softmax(-1)
            token = logprobs.argmax(-1)
            logprob = logprobs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
            prev_token, prev_logprob = state["token"], state["logprob"]
            state["token"], state["logprob"] = token, logprob
            if prev_token is None or (loop_idx + 1) < min_loops:
                return False
            same = bool((token == prev_token).all())
            settled = bool(((logprob - prev_logprob).abs() < tol).all())
            return same and settled

        return check

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        return_aux_loss=False,
        return_hidden=False,
        n_loops: int = None,
        kv_cache=None,
        converge_tol: float = None,
        min_loops: int = 1,
        skip_mtp: bool = False,
        evidence: EvidenceBatch = None,
        token_mask: torch.Tensor = None,
        reader_sites_kept: int = None,
    ):
        """forward pass of the model

        Args:
            input_ids (torch.Tensor): input token ids, shape [batch_size, seq_len]. When
                ``kv_cache`` is given, this must be only the newly-appended tokens, not the full
                sequence -- everything before them is already reflected in the cache.
            cu_seqlens (torch.Tensor, optional): int32 cumulative segment boundaries over the
                flattened [B*S] token axis for document-packed varlen attention. Defaults to None (normal causal attention).
            max_seqlen (int, optional): longest packed segment length. Defaults to None.
            return_aux_loss (bool, optional): whether to return the MoE routing auxiliary loss. Defaults to False.
            n_loops (int, optional): run the MoE block a different number of times than it was
                configured with. Both the per-loop router bias and loop_scale are indexed by
                absolute loop index, so this needs no weight reshaping -- see
                LoopMixtureOfExperts.forward. Training should leave this None (loop_ce_weights is
                length-checked against the configured n_loops). Defaults to None.
            kv_cache (modules.model.kv_cache.KVCache, optional): incremental decode cache built by
                ``KVCache.for_model(model, n_loops=n_loops)``. When given, ``cu_seqlens`` must be
                None (single unpacked sequence) and ``n_loops`` must match the value the cache was
                built with. Inference/generation only -- never set during training. Defaults to
                None.
            converge_tol (float, optional): enable the parameter-free convergence exit at this
                tolerance (see ``_convergence_exit``). Inference only, and mutually exclusive with
                ``kv_cache``. Defaults to None (always run the full depth).
            min_loops (int, optional): floor on the loop count when ``converge_tol`` is set.
                Defaults to 1.
            skip_mtp (bool, optional): don't run the MTP head at all, and drop
                ``extra_token_outputs`` from the return. The head is a pure function of the final
                loop's normed hidden state, so not running it cannot change the logits -- but it is
                paid over the whole prefix on every generated token, and every caller that only
                wants logits (greedy decode, the log-likelihood scorers, the calibration probes)
                was throwing the result away. Defaults to False.
            evidence (EvidenceBatch, optional): retrieved evidence for this batch, built by
                ``build_evidence``. Read at every loop by the always-on evidence port. Defaults to
                None, which reproduces the forward this model ran before the port existed, bit for
                bit -- that property is what lets one checkpoint serve both modes and is asserted in
                ``tests/test_evidence_port.py``.
            token_mask (torch.Tensor, optional): [batch_size, seq_len], True/1 for a real (non pad)
                token, forwarded to the MoE's load balancing loss (see ``compute_aux_loss``). The
                aux loss is a mean over positions, so without this it is read over padding too and
                its value moves with how full the batch's rows happen to be -- which the evidence
                corpus's rows, closing early on their evidence budget, vary a lot more than
                document-packed pretraining ones do. Defaults to None, which is the aux loss this
                model trained under, bit for bit.
            reader_sites_kept (int, optional): read ablation, see ``LoopMixtureOfExperts.forward``:
                every evidence read site numbered above this value reads nothing. 0 is the residual
                stream of ``evidence=None``; None (the default) is today's forward, bit for bit.
                Inference only. Defaults to None.

        Returns:
            torch.Tensor: output logits, shape [batch_size, seq_len, vocab_size]. If return_hidden
                is True, returns the post-norm hidden states for every loop instead, shape
                [loops_run, batch_size, seq_len, hidden_size] -- index [-1] is the final loop, and
                ``loops_run`` is below ``n_loops`` only when ``converge_tol`` fired.

            float (optional): auxiliary loss from MoE routing, returned if return_aux_loss is True

            extra_token_outputs (optional): if MTP is enabled returns either the hidden states for the extra tokens (if delayed_mtp_loss is True) or the logits for the extra tokens (if delayed_mtp_loss is False)

            If delayed_mtp_loss is True, the shape of extra_token_outputs is [batch_size, seq_len, num_extra_tokens, hidden_size // 2]

            If delayed_mtp_loss is False, the shape of each element in extra_token_outputs is [batch_size, seq_len, vocab_size]
        """
        assert reader_sites_kept is None or not self.training, "reader_sites_kept is inference-only"
        self._token_tracker.count_tokens(input_ids)
        if self.training and self.use_checkpointing:
            assert converge_tol is None, "converge_tol is inference-only"
            x = checkpoint(self.gemma_decoder, input_ids, cu_seqlens, max_seqlen, use_reentrant=False)
            _, aux_loss, hidden_states_all = checkpoint(self.moe, x.last_hidden_state, self._moe_ple(input_ids), True, cu_seqlens, max_seqlen, self.use_sub_checkpointing, n_loops, None, 0, None, evidence, token_mask, use_reentrant=False)
            # final RMSNorm applied at every loop, not just the last -- lm_head reads self.norm(x),
            # never the raw residual stream, so per-loop CE needs this too.
            x_all = self.norm(hidden_states_all)
            x = x_all[-1]
            extra_token_outputs = None if skip_mtp else self._mtp_forward(x, use_checkpointing=self.use_sub_checkpointing)
            x = x_all if return_hidden else self.lm_head(x)
        else:
            assert kv_cache is None or cu_seqlens is None, "kv_cache decoding is single-sequence only, cu_seqlens must be None"
            position_offset = kv_cache.length if kv_cache is not None else 0
            decoder_cache = kv_cache.decoder if kv_cache is not None else None
            moe_cache = kv_cache.moe if kv_cache is not None else None
            exit_check = None if converge_tol is None else self._convergence_exit(converge_tol, min_loops)
            x = self.gemma_decoder(input_ids, cu_seqlens, max_seqlen, kv_cache=decoder_cache, position_offset=position_offset).last_hidden_state
            _, aux_loss, hidden_states_all = self.moe(x, other=self._moe_ple(input_ids), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_loss=True, n_loops=n_loops, kv_cache=moe_cache, position_offset=position_offset, exit_check=exit_check, evidence=evidence, token_mask=token_mask, reader_sites_kept=reader_sites_kept)
            x_all = self.norm(hidden_states_all)
            x = x_all[-1]
            extra_token_outputs = None if skip_mtp else self._mtp_forward(x, use_checkpointing=False)
            x = x_all if return_hidden else self.lm_head(x)

        if extra_token_outputs is not None:
            return (x, aux_loss, extra_token_outputs) if return_aux_loss else (x, extra_token_outputs)
        return (x, aux_loss) if return_aux_loss else x


    def set_checkpointing(self, use_checkpointing: bool, use_sub_checkpointing: bool = None):
        """set gradient checkpointing for the model. If use_sub_checkpointing is None, it will be set to the same value as use_checkpointing.

        Args:
            use_checkpointing (bool): whether to use gradient checkpointing for the model stages (Gemma decoder and MoE)
            use_sub_checkpointing (bool, optional): whether to use gradient checkpointing for the substages within the MoE. Defaults to None.
        """
        self.use_checkpointing = use_checkpointing
        if use_sub_checkpointing is not None:
            self.use_sub_checkpointing = use_sub_checkpointing
    
    def delayed_mtp_loss(self, set_to_true: bool = None):
        """whether to delay MTP loss computation until after the main loss backward pass to save VRAM"""
        if (set_to_true is not None) and self.has_mtp:
            self.mtp_head.late_token_loss = set_to_true
        return self.mtp_head is not None and self.mtp_head.late_token_loss

    @property
    def has_mtp(self):
        return self.mtp_head is not None
