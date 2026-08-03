import torch
from torch import nn
import torch.nn.functional as F

from modules.model.moe import LoopMixtureOfExperts
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextModel
from modules.model.modules import SmallLMHead
from modules.model.mtp import MTPHead
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
        dropout: float = 0.1,
        ple_embeddings_size: int = None,
        mtp_num_extra_tokens: int = 0,
        lm_head_factor: int = 8,
        moe_intermediate_size: int = None,
    ):
        super().__init__()

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
            dropout=dropout,
            top_k=top_k,
            n_loops=n_loops,
            max_seq_len=max_seq_len,
        )
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = SmallLMHead(hidden_size, vocab_size, factor=lm_head_factor)

        # correctness head (PLAN.md Step 4b): separate from the MoE's p_halt on purpose -- p_halt
        # asks "is more compute useful", this asks "is this prediction correct"; they come apart
        # on confident hallucinations. Applied externally (like lm_head/mtp_head) to the final
        # loop's post-norm hidden state inside compute_mtp_loss, not inside forward().
        self.correct_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.correct_proj.weight)
        nn.init.constant_(self.correct_proj.bias, 0.0)

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

        self.body_flops_per_token = 2 * (body_params + n_loops * moe_active_params)
        self.lm_head_flops_per_token = 2 * lm_head_params           # per application (once per loop)
        self.mtp_flops_per_token = 2 * (mtp_body_params + mtp_num_extra_tokens * mtp_lm_head_params)
        self.attn_flops_per_seqsq = 2 * hidden_size * n_attn_layers  # multiply by sum(segment_len^2)

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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        return_aux_loss=False,
        return_hidden=False,
        n_loops: int = None,
    ):
        """forward pass of the model

        Args:
            input_ids (torch.Tensor): input token ids, shape [batch_size, seq_len]
            cu_seqlens (torch.Tensor, optional): int32 cumulative segment boundaries over the
                flattened [B*S] token axis for document-packed varlen attention. Defaults to None (normal causal attention).
            max_seqlen (int, optional): longest packed segment length. Defaults to None.
            return_aux_loss (bool, optional): whether to return auxiliary loss (and p_halt). Defaults to False.
            n_loops (int, optional): run the MoE block a different number of times than it was
                configured with. Both the per-loop router bias and loop_scale are indexed by
                absolute loop index, so this needs no weight reshaping -- see
                LoopMixtureOfExperts.forward. Training should leave this None (loop_ce_weights is
                length-checked against the configured n_loops). Defaults to None.

        Returns:
            torch.Tensor: output logits, shape [batch_size, seq_len, vocab_size]. If return_hidden
                is True, returns the post-norm hidden states for every loop instead, shape
                [n_loops, batch_size, seq_len, hidden_size] (PLAN.md Step 4a) -- index [-1] is the
                final loop, matching the pre-Step-4a single-loop shape.

            float (optional): auxiliary loss from MoE routing, returned if return_aux_loss is True

            p_halt (optional): [n_loops, batch_size, seq_len] per-loop halt probability from the
                MoE halt head, returned if return_aux_loss is True

            extra_token_outputs (optional): if MTP is enabled returns either the hidden states for the extra tokens (if delayed_mtp_loss is True) or the logits for the extra tokens (if delayed_mtp_loss is False)

            If delayed_mtp_loss is True, the shape of extra_token_outputs is [batch_size, seq_len, num_extra_tokens, hidden_size // 2]

            If delayed_mtp_loss is False, the shape of each element in extra_token_outputs is [batch_size, seq_len, vocab_size]
        """
        self._token_tracker.count_tokens(input_ids)
        if self.training and self.use_checkpointing:
            x = checkpoint(self.gemma_decoder, input_ids, cu_seqlens, max_seqlen, use_reentrant=False)
            _, aux_loss, p_halt, hidden_states_all = checkpoint(self.moe, x.last_hidden_state, self._moe_ple(input_ids), True, cu_seqlens, max_seqlen, self.use_sub_checkpointing, n_loops, use_reentrant=False)
            # final RMSNorm applied at every loop, not just the last (PLAN.md Step 4a) -- lm_head
            # reads self.norm(x), never the raw residual stream, so per-loop CE needs this too.
            x_all = self.norm(hidden_states_all)
            x = x_all[-1]
            extra_token_outputs = self._mtp_forward(x, use_checkpointing=self.use_sub_checkpointing)
            x = x_all if return_hidden else self.lm_head(x)
        else:
            x = self.gemma_decoder(input_ids, cu_seqlens, max_seqlen).last_hidden_state
            _, aux_loss, p_halt, hidden_states_all = self.moe(x, other=self._moe_ple(input_ids), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_loss=True, n_loops=n_loops)
            x_all = self.norm(hidden_states_all)
            x = x_all[-1]
            extra_token_outputs = self._mtp_forward(x, use_checkpointing=False)
            x = x_all if return_hidden else self.lm_head(x)

        if extra_token_outputs is not None:
            return (x, aux_loss, p_halt, extra_token_outputs) if return_aux_loss else (x, extra_token_outputs)
        return (x, aux_loss, p_halt) if return_aux_loss else x


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
