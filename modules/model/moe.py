import math

import torch
from torch import nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
# TE checkpoint required for quantized (FP8/NVFP4) layers, see transformer.py note
from transformer_engine.pytorch import checkpoint

from modules.model.router import Router, compute_aux_loss
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm
from modules.model.experts import CrossAttention, InformationRetrievalExpert, SelfAttention
from modules.model.information_retrieval import RetrievalEntropyTracking, evidence_selection_loss
from modules.model.embeddings import RotaryPositionEmbeddingsFrequency
from modules.model.evidence import EvidenceBatch, evidence_memory, chunk_mean_mass, apply_chunk_gate



class ParallelSparseMoELayer(nn.Module):
    """a sparse MoE layer that dispatches each token only to its routed experts via a grouped GEMM (Transformer Engine ``GroupedLinear`` does the Grouped MatMul).

    Flow:
    1. sort the (token, slot) assignments by expert
    2. run one variable sized GEMM per expert group
    3. scatter the weighted results back
    => FF experts only do work for the tokens actually routed to them
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size

        # fuse gate + up into a single grouped GEMM
        # one weight per expert (Linear layout [out, in])
        self.gate_up = te.GroupedLinear(num_experts, hidden_size, 2 * intermediate_size, bias=False)
        self.down = te.GroupedLinear(num_experts, intermediate_size, hidden_size, bias=False)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        # topk_weights, topk_indices: [batch_size, seq_len, top_k]
        # non MLP (null) slots arrive as index 0 with weight 0 => contribute nothing after scaling
        B, S, H = x.shape
        top_k = topk_indices.shape[-1]
        x_flat = x.reshape(-1, H)                       # [T, H]
        T = x_flat.shape[0]

        idx = topk_indices.reshape(-1)                  # [N] expert id per (token, slot), N = T*top_k
        wgt = topk_weights.reshape(-1)                  # [N] routing weight per slot
        tok = torch.arange(T, device=x.device).repeat_interleave(top_k)  # [N] owning token per slot

        # sort assignments so each experts rows are contiguous (required by GroupedLinear)
        # stable=True for determinism across checkpoint recompute pass
        order = torch.argsort(idx, stable=True)         # [N]
        idx_sorted = idx[order]
        tok_sorted = tok[order]
        wgt_sorted = wgt[order]
        x_sorted = x_flat.index_select(0, tok_sorted)   # [N, H] each slot sees its (unscaled) token

        # per expert group sizes
        m_splits = torch.bincount(idx_sorted, minlength=self.num_experts).tolist() # host sync :(

        # run the experts in BF16: NVFP4 requires each GEMMs row count (a dynamic per expert group size here) to be divisible by 16, which cant be guaranteed
        # => sparsity more important
        with te.autocast(enabled=False):
            gate_up = self.gate_up(x_sorted, m_splits)  # [N, 2*intermediate]
            gate, up = gate_up.chunk(2, dim=-1)
            act = self.activation(gate) * up            # [N, intermediate]
            out_sorted = self.down(act, m_splits)       # [N, H]

        # scale each expert output by its routing weight (null slots => *0), then scatter back
        out_sorted = out_sorted * wgt_sorted.unsqueeze(-1)
        combined = torch.zeros(T, H, device=x.device, dtype=out_sorted.dtype)
        combined.index_add_(0, tok_sorted, out_sorted)

        return combined.view(B, S, H)


class SharedMLP(nn.Module):
    """dense SwiGLU MLP applied to every token unconditionally (PLAN.md Step 2) -- gives the loop
    a guaranteed dense transform independent of routing, unlike ``ParallelSparseMoELayer`` which
    only touches the tokens routed to each expert. Static row count (``B*S``), so it may run
    inside ``te.autocast`` rather than being forced to BF16 like the routed grouped GEMM.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up = te.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = te.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.activation(gate) * up)


class _ExpertTracking():
    def __init__(self, num_experts: int):
        self._num_experts = num_experts
        self.prob_dist = None      # lazily placed on device on first update
        self.post_skew_dist = None
        self.choices = None
        self.sliding_window_size = 256
        # recompute guard: under activation checkpointing route() runs again during backward, which would double count every update
        self._expected_updates = None
        self._seen_updates = 0
        self.sample_interval = 8
        self._forward_counter = 0
        self._active = True

    def begin_forward(self, expected_updates: int):
        self._expected_updates = expected_updates
        self._seen_updates = 0
        self._active = (self._forward_counter % self.sample_interval) == 0
        self._forward_counter += 1

    def update(self, topk_indices: torch.Tensor, topk_scores: torch.Tensor, expert_scores: torch.Tensor):
        if not self._active:
            return  # throttled forward, skip the stats gather
        if self._expected_updates is not None:
            if self._seen_updates >= self._expected_updates:
                return  # checkpoint recompute pass, already counted
            self._seen_updates += 1
        n = self._num_experts
        device = topk_indices.device
        if self.prob_dist is None or self.prob_dist.device != device:
            self.prob_dist = torch.zeros(n, device=device)
            self.post_skew_dist = torch.zeros(n, device=device)
            self.choices = torch.zeros(n, device=device)

        topk_indices_flat = topk_indices.detach().view(-1, topk_indices.size(-1))  # [T, k]
        topk_scores_flat = topk_scores.detach().view(-1, topk_scores.size(-1))     # [T, k]
        expert_scores_flat = expert_scores.detach().view(-1, n)                    # [T, n]
        num_tokens = topk_indices_flat.size(0)

        # compute once (not inside a per expert loop)
        one_hot_mask = F.one_hot(topk_indices_flat, num_classes=n).any(dim=1)      # [T, n] bool

        prob_updates = torch.zeros(n, device=device, dtype=topk_scores.dtype)
        prob_updates.scatter_add_(0, topk_indices_flat.reshape(-1), topk_scores_flat.reshape(-1))

        post_skew_updates = (expert_scores_flat * one_hot_mask.to(expert_scores.dtype)).sum(dim=0)
        choices_updates = one_hot_mask.sum(dim=0).float()

        # normalize to per token quantities so the EMAs have a meaning
        decay = 1.0 - 1.0 / self.sliding_window_size
        inv_w = (1.0 / self.sliding_window_size) / max(num_tokens, 1)
        self.prob_dist = self.prob_dist * decay + prob_updates * inv_w
        self.post_skew_dist = self.post_skew_dist * decay + post_skew_updates * inv_w
        self.choices = self.choices * decay + choices_updates * inv_w

    def get_stats(self):
        return {
            "prob_dist": self.prob_dist.cpu().tolist() if self.prob_dist is not None else [0.0] * self._num_experts,
            "post_skew_dist": self.post_skew_dist.cpu().tolist() if self.post_skew_dist is not None else [0.0] * self._num_experts,
            "choices": self.choices.cpu().tolist() if self.choices is not None else [0.0] * self._num_experts,
        }

    def reset_stats(self):
        if self.prob_dist is not None:
            self.prob_dist.zero_()
            self.post_skew_dist.zero_()
            self.choices.zero_()

def is_fresh_loop_param(name: str) -> bool:
    """True for a block tensor a migration adds fresh, which needs the from-scratch rate.

    The input injection and the whole evidence reader. Neither carries any training, so at the
    trunk's rate they would sit at their init for the entire run and the run would measure nothing
    -- the same reason ``is_rebuilt_ir_param`` exists, kept separate because that one is about the
    IR table's reshape and these two lists must be free to disagree.

    The evidence reader is matched as a whole subtree rather than tensor by tensor: it is one module
    that arrives complete, and listing its four projections individually is a list that goes stale
    the first time the module gains a norm.

    The selector's three tensors are matched by name instead, and deliberately NOT as the whole
    ``ir_module`` subtree: the key/value tables in that same module carry a full sharpening run and
    would be wrecked by a from-scratch rate, while the adapters and the source scale have never seen
    a gradient. This is the one place in the block where fresh and trained tensors share a module.

    Three more tensors, all born the same way: the loop-conditioned query biases on the IR module
    and the evidence reader, and the reader's own per-loop gain. None of them existed before any
    checkpoint currently on disk, all are zero- or one-init neutral, and all would sit at that init
    for a whole run at the trunk's rate for the identical reason the tensors above do.

    Two more, added with the direct IR read path and the reader/selector coupling: each IR expert's
    ``direct_gate`` (zero-init, matched by name rather than folded into the ``ir_module`` subtree
    check above -- it lives on the EXPERT, like ``down_proj``/``up_proj``, not inside the table
    module) and ``evidence_gate_scale`` (the reader's per chunk gate, also zero-init).
    """
    return (
        name.endswith("moe.inject.weight")
        or name == "inject.weight"
        or "shared_evidence." in name
        or name.startswith("shared_evidence.")
        or ".key_adapter." in name
        or ".value_adapter." in name
        or name.endswith("log_memory_scale")
        or name.endswith("loop_query_bias.weight")
        or name.endswith("evidence_query_bias.weight")
        or name.endswith("evidence_loop_scale")
        or name.endswith("direct_gate.weight")
        or name.endswith("evidence_gate_scale")
        # the groundedness readout, the one entry here that lives on the model rather than in the
        # block: same story, a whole module arriving at its zero-init from a migration
        or name.startswith("groundedness_head.")
        or ".groundedness_head." in name
    )


class LoopMixtureOfExperts(nn.Module):
    """a Mixture of Experts module that routes tokens to a mixture of attention and feedforward experts in multiple loops"""
    def __init__(
        self,
        hidden_size: int, 
        intermediate_size: int, 
        num_mlp_experts: int, 
        num_attn_experts: int = 4,
        num_ir_experts: int = 0,
        top_k: int = 2,
        n_loops: int = 8, 
        dropout: float = 0.0,
        temperature: float = 1.0,
        num_ir_entries: int = 1024,
        ir_dim: int = 128,
        ir_residual: bool = False,
        ir_num_clusters: int = 0,
        ir_probe_clusters: int = 4,
        ir_read_top_k: int = 32,
        max_seq_len: int = 4096,
        rope_theta: float = 100000.0,
        loop_scale_init: float = None,
        loop_enc_dim: int = 32,
        max_enc_loops: int = 64,
        loop_inject: bool = False,
        evidence_port: bool = False,
        ir_direct_read: bool = True,
    ):
        """Mixture of Experts module with multiple loops of routing to a mixture of attention and feedforward experts

        Args:
            hidden_size (int): hidden size of the input and output representations
            intermediate_size (int): intermediate size of the feedforward layers (MLP)
            num_mlp_experts (int): number of MLP experts in the mixture
            num_attn_experts (int, optional): number of attention experts. Defaults to 4.
            num_ir_experts (int, optional): number of intermediate representation experts. Defaults to 0.
            top_k (int, optional): number of top experts to route each token to. Defaults to 2.
            n_loops (int, optional): number of routing loops. Defaults to 8.
            dropout (float, optional): dropout probability. Defaults to 0.0.
            temperature (float, optional): temperature for the router. Defaults to 1.0.
            num_ir_entries (int, optional): number of entries in the information retrieval expert. Defaults to 1024.
            ir_dim (int, optional): dimension of the information retrieval experts latent space. Defaults to 128.
            ir_residual (bool, optional): whether the information retrieval expert should have a residual connection. Defaults to False.
            ir_num_clusters (int, optional): centroids for the IR table's two stage scoring. 0
                keeps the exact full-table softmax, which is what every checkpoint before the
                65536-entry reshape was trained under. Defaults to 0.
            ir_probe_clusters (int, optional): how many centroids a token opens for exact scoring.
                Defaults to 4.
            ir_read_top_k (int, optional): how many entries the read softmax spans. Defaults to 32.
            loop_scale_init (float, optional): init value for every entry of the per-loop
                ``loop_scale`` gate. Defaults to ``1 / sqrt(n_loops)``, which makes the whole loop
                stack contribute roughly as much variance as the dense decoder's output at init
                instead of the ~1.5% a 0.1 init gave.
            loop_enc_dim (int, optional): width of the sinusoidal loop-index encoding that biases
                the router per loop. Defaults to 32.
            max_enc_loops (int, optional): size of the precomputed loop-encoding table. Loop
                indices past it reuse the last row, so running more loops at inference than were
                trained still works. Defaults to 64.
            loop_inject (bool, optional): re-present the block's input to every loop, as
                ``hidden_states + inject(e)`` where ``e`` is what the dense decoder handed over.
                Without it the only thing a later loop knows about the input is whatever survives
                in the residual stream, which is also carrying every earlier loop's update. The
                projection is zero-init, so turning it on is exactly neutral until it learns
                something. Defaults to False.
            evidence_port (bool, optional): add the always-on evidence reader -- a CrossAttention
                that reads retrieved evidence tokens and seeds the accumulator alongside
                ``shared_mlp``/``shared_attn``. **Deliberately not in the router pool**: the router
                never specialized in the real run (the aux loss balanced it from step 0), so making
                "does this token need a fact?" depend on the weakest measured component is a bad
                bet, and the content is gated by retrieval scores instead. Its output projection is
                zero-init, so attaching a corpus is neutral at step 0 the same way ``g_proj`` and
                ``loop_router_bias`` are. With no evidence attached it does not run at all, which is
                what keeps the no-corpus forward bit-identical. Also adds ``evidence_query_bias``
                (a zero-init per-loop bias on the reader's query, reusing ``loop_enc``) and
                ``evidence_loop_scale`` (a one-init per-loop gain on the read alone, independent of
                ``loop_scale``'s own shrinkage) -- without them the reader reads the same query at
                every loop and the read reaches the residual weaker on later loops regardless of
                what it learns. Defaults to False.
            ir_direct_read (bool, optional): give every IR expert a direct, per token output stage
                instead of the inner attention that averages a document's reads over its whole
                prefix (see ``InformationRetrievalExpert``'s docstring for why the averaged path
                measurably degenerates to a per document constant). Selectable so the averaged path
                stays runnable for comparison; a checkpoint keeps BOTH sets of weights regardless of
                this flag, since it only changes which one ``forward_step`` calls. Defaults to True.
        """
        super().__init__()
        self._num_mlp_experts = num_mlp_experts
        self._num_attn_experts = num_attn_experts * 2 # account for both self and cross attention experts
        self._num_ir_experts = num_ir_experts
        self.top_k = top_k
        self.n_loops = n_loops
        self.temperature = temperature
        
        # num expert heads
        n_heads = 16
        n_kv_heads = 4

        # rotary embeddings for the attention experts. the experts run with their own head count
        # (n_heads above), so their head_dim differs from the Gemma decoders
        # they need a separately sized RoPE cache :( 
        # Positions are global over the packed sequence (0..S-1),
        # matching the decoders convention so cross document packing stays consistent
        self.rotary_emb = RotaryPositionEmbeddingsFrequency(
            dim=hidden_size // n_heads,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
        )

        # experts
        experts = []
        experts.extend([
            SelfAttention(input_size=hidden_size, dropout=dropout, num_heads=n_heads, num_kv_heads=n_kv_heads)
            for _ in range(num_attn_experts)
        ])
        experts.extend([
            CrossAttention(input_size=hidden_size, dropout=dropout, num_heads=n_heads, num_kv_heads=n_kv_heads)
            for _ in range(num_attn_experts)
        ])
        experts.extend([
            InformationRetrievalExpert(
                input_size=hidden_size,
                num_entries=num_ir_entries,
                ir_dim=ir_dim,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                dropout=dropout,
                residual=ir_residual,
                num_clusters=ir_num_clusters,
                probe_clusters=ir_probe_clusters,
                read_top_k=ir_read_top_k,
                # the external store arrives at the embedder's native width, which is what ir_dim
                # was widened to -- so the adapters are square rotations rather than compressions,
                # and no information is thrown away relocating one space into the other
                memory_dim=ir_dim if evidence_port else 0,
                direct_read=ir_direct_read,
            ) for _ in range(num_ir_experts)
        ])
        self.experts = nn.ModuleList(experts)

        # always-on dense transform + full-sequence attention, seeded into forward_step's
        # accumulator every loop. neither is in the router pool (not in Router's output dim,
        # not in compute_aux_loss) -- stabilises early training and lets routed experts
        # specialise instead of all re-learning the same generic function (PLAN.md Step 2).
        self.shared_mlp = SharedMLP(hidden_size, intermediate_size)
        self.shared_attn = SelfAttention(input_size=hidden_size, dropout=dropout, num_heads=n_heads, num_kv_heads=n_kv_heads)

        self.parallel_experts = ParallelSparseMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_mlp_experts
        )
        
        # router
        self.router = Router(hidden_size=hidden_size, num_experts=self.num_experts)
        
        self.post_norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # LayerScale/ReZero-style gate, ONE ENTRY PER LOOP: each loop learns independently how much
        # refinement it contributes, instead of all n_loops sharing a single scalar.
        #
        # This is now the ONLY per-loop gain. It used to be multiplied by a learned per-token
        # (1 - p_halt) as well; that head saturated and was deleted, with its measured per-loop mean
        # folded into these values by scripts/migrate_phase0.py -- so a migrated checkpoint's
        # loop_scale is much smaller than a freshly initialized one and that is correct, not a bug.
        #
        # init 1/sqrt(n_loops), NOT 0 and not 0.1: post_norm makes every delta unit-RMS, so with the
        # old 0.1 the n_loops deltas summed to ~sqrt(n_loops)*0.1 ~ 0.17 against a unit-RMS decoder
        # output -- the entire MoE block (most of the model's parameters) was a ~1.5% perturbation,
        # and a lone scalar at lr=4e-4 cannot climb out of that inside a short run. 1/sqrt(n_loops)
        # puts the loop stack at ~1.0, i.e. on par with the decoder, from step 0.
        # not the same mechanism as gemma4's layer_scalar (init-1 gain on a whole layer output) --
        # that precedent doesn't transfer to this init value.
        # excluded from weight decay by scripts/pretrain.py's param grouping (ndim <= 1): decaying
        # this toward 0 is decaying the loop toward "off".
        loop_scale_init = loop_scale_init if loop_scale_init is not None else 1.0 / math.sqrt(n_loops)
        self.loop_scale = nn.Parameter(torch.full((n_loops,), float(loop_scale_init)))

        # per-loop router conditioning: without it the router sees near-identical inputs on every
        # loop (consecutive loop inputs differ only by loop_scale * unit-RMS delta), picks the same
        # top_k experts all n_loops times, and the recurrence degenerates into running one expert
        # pair n_loops times over.
        #
        # encoded as a SINUSOIDAL function of the absolute loop index rather than a learned
        # [n_loops, num_experts] table, so the loop count is not baked into the weights: any loop
        # index has a defined encoding, and n_loops can be changed at inference (see forward's
        # n_loops override). Precomputed as a table so per-loop lookup is a device index, never a
        # host->device copy in the step path.
        loop_enc_dim = int(loop_enc_dim) // 2 * 2  # sin/cos halves
        inv_freq = 1.0 / (100.0 ** (torch.arange(0, loop_enc_dim, 2).float() / loop_enc_dim))
        angles = torch.arange(max_enc_loops).float().unsqueeze(1) * inv_freq.unsqueeze(0)
        self.register_buffer(
            "loop_enc", torch.cat([angles.sin(), angles.cos()], dim=-1), persistent=False
        )  # [max_enc_loops, loop_enc_dim]
        # zero-init: the loop bias starts as an exact no-op, so routing at step 0 is identical to
        # the un-conditioned router and this cannot perturb the load-balance loss at init.
        self.loop_router_bias = nn.Linear(loop_enc_dim, self.num_experts, bias=False)
        nn.init.zeros_(self.loop_router_bias.weight)

        # re-presents the block's input to every loop (see loop_inject in the docstring). Same
        # zero-init neutrality as loop_router_bias above: it starts as an exact no-op, so a
        # checkpoint migrated to carry it scores identically to the one it came from. Attribute
        # always exists so the forward can branch on `is None` rather than hasattr.
        self.inject = None
        if loop_inject:
            self.inject = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.inject.weight)

        # the evidence reader (see evidence_port in the docstring). Always-on rather than routed,
        # and re-read at every loop, so a later loop can return to the evidence with a query the
        # earlier loops' updates have moved -- which is the one thing the diagnostics said later
        # loops lack. Zero-init o_proj rather than leaving it at its default: a corpus attached to a
        # freshly built port would otherwise inject noise into a converged trunk on step 0, and the
        # whole point of the neutrality pattern is that migration is a no-op you can measure.
        self.shared_evidence = None
        self.evidence_query_bias = None
        self.evidence_loop_scale = None
        if evidence_port:
            self.shared_evidence = CrossAttention(
                input_size=hidden_size, dropout=dropout, num_heads=n_heads, num_kv_heads=n_kv_heads
            )
            nn.init.zeros_(self.shared_evidence.attn.o_proj.weight)

            # per-loop query conditioning for the reader, same problem and same fix as the IR
            # module's loop_query_bias: without it shared_evidence reads step_input directly at
            # every loop, so its three reads are near-copies of each other (Stage 0 measured the
            # identical symptom on the IR query). Reuses this module's own loop_enc buffer rather
            # than building a second one -- zero-init, so a checkpoint that gains this tensor reads
            # identically until it learns something.
            self.evidence_query_bias = nn.Linear(loop_enc_dim, hidden_size, bias=False)
            nn.init.zeros_(self.evidence_query_bias.weight)

            # a gain on the evidence read ALONE, separate from loop_scale: loop_scale already
            # shrinks the WHOLE accumulator every loop (measured [0.63, 0.32, 0.11] on a migrated
            # checkpoint), so without this a later loop's re-read would reach the residual at a
            # fraction of loop 1's gain no matter what the reader learns. Init 1.0, not
            # 1/sqrt(n_loops): shared_evidence's o_proj is zero-init, so the value being scaled is
            # exactly zero at step 0 and ANY finite init is neutral there -- 1.0 is simply the
            # natural starting point once o_proj moves off zero, rather than stacking a second
            # shrinkage on top of loop_scale's own. One entry per loop, indices past n_loops - 1
            # reuse the last one, same convention as loop_scale. ndim 1, so build_param_groups'
            # existing decay rule already excludes it from weight decay without any change there --
            # decaying a gate toward 0 is decaying the read toward "off", same argument as loop_scale.
            self.evidence_loop_scale = nn.Parameter(torch.ones(n_loops))

            # couples the reader to the selector: a per chunk multiplicative gate on the evidence
            # STATES, from that same loop's own selector chunk scores (see forward_step and
            # evidence.apply_chunk_gate). `1 + evidence_gate_scale * sigmoid(chunk_mass)` rather than
            # a bare `sigmoid(...)`, so that a ZERO-INIT scale alone makes the gate exactly 1.0 for
            # every chunk regardless of the sigmoid term -- a plain sigmoid can only approach 1
            # asymptotically, never reach it exactly, and this module's whole neutrality convention
            # is an EXACT no-op at step 0, not an approximate one. Gives the selector a dense,
            # always-on gradient through the reader (complementary to, not a replacement for, the
            # supervised selection loss over the mass split -- see information_retrieval.py).
            self.evidence_gate_scale = nn.Parameter(torch.zeros(()))
        else:
            self.evidence_gate_scale = None

        # the selector's token-to-chunk visibility mask for the most recent forward, the second half
        # of what a supervised selection loss needs (the first half being each IR module's
        # last_memory_weights). Kept here rather than on the IR modules because it is built once per
        # recurrence from the query side's segmentation and is the same object every IR expert and
        # every loop scored against. Plain attribute, cleared on a forward without evidence, same
        # lifetime rule as last_memory_mass.
        self.last_memory_visible = None
        # the evidence reader's own output from the most recent loop, [B, S, H], for a groundedness
        # readout to attach to. See forward_step for why it is the ungated, unscaled read.
        self.last_reader_output = None

        self.expert_tracker = _ExpertTracking(num_experts=self.num_experts)

        # one retrieval entropy tracker shared by every IR expert, so the trainer reads a single
        # per loop vector regardless of how many IR slots the pool has. None when there are no IR
        # experts at all -- the log line then just omits the field
        if num_ir_experts > 0:
            # the tracker normalizes by the width of the softmax the module actually takes: the
            # whole table on the exact path, only the read top-k once two stage scoring is on.
            # Normalizing a top-32 read by ln 65536 would report 0.31 for a read that is perfectly
            # uniform over everything it looked at, i.e. exactly the failure G1 measured, as a pass.
            tracker_width = ir_read_top_k if ir_num_clusters > 0 else num_ir_entries
            self.ir_tracker = RetrievalEntropyTracking(num_entries=tracker_width, n_loops=n_loops)
            for expert in self.experts:
                if isinstance(expert, InformationRetrievalExpert):
                    expert.ir_module.tracker = self.ir_tracker
        else:
            self.ir_tracker = None

    @property
    def num_experts(self):
        return self._num_attn_experts + self._num_ir_experts + self._num_mlp_experts

    @property
    def first_mlp_index(self):
        """index of the first MLP expert in the flat router pool: [self-attn x A | cross-attn x A | IR x I | MLP x M]."""
        return self._num_attn_experts + self._num_ir_experts
    
    def _loop_enc_row(self, loop_idx: int, dtype: torch.dtype) -> torch.Tensor:
        """the raw sinusoidal loop encoding row, shared by every per-loop bias in this block.

        Indices past the precomputed table reuse its last row, so ``n_loops`` stays a runtime
        choice rather than something baked into a weight shape.
        """
        row = min(int(loop_idx), self.loop_enc.size(0) - 1)
        return self.loop_enc[row].to(dtype)

    def loop_bias(self, loop_idx: int) -> torch.Tensor:
        """per-loop additive bias on the router logits, shape [num_experts].

        Derived from a sinusoidal encoding of the *absolute* loop index, so it is defined for any
        index -- ``n_loops`` is a runtime choice, not something baked into the weight shapes.
        Indices past the precomputed table reuse its last row.
        """
        return self.loop_router_bias(self._loop_enc_row(loop_idx, self.loop_router_bias.weight.dtype))

    def route(self, hidden_states: torch.Tensor, temperature: float = 1.0, loop_idx: int = 0, token_mask: torch.Tensor = None):
        """routes tokens to experts and computes the load balancing loss

        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            temperature (float, optional): temperature for the router. Defaults to 1.0.
            loop_idx (int, optional): which loop this call belongs to, used for the per-loop router
                bias so consecutive loops do not all select the same experts. Defaults to 0.
            token_mask (torch.Tensor, optional): [batch_size, seq_len], True/1 for a real (non pad)
                token. Forwarded to ``compute_aux_loss`` unchanged -- see its docstring. None (the
                default) reproduces today's aux loss exactly, pad rows included.

        Returns:
            topk_scores (torch.Tensor): [batch_size, seq_len, top_k] normalized scores for the selected experts

            topk_indices (torch.Tensor): [batch_size, seq_len, top_k] indices of the selected experts

            load_balancing_loss (torch.Tensor): auxiliary loss to encourage balanced routing
        """
        expert_logits = self.router(hidden_states, temperature=temperature)       # [batch_size, seq_len, num_experts] raw logits
        # broadcasts over [B, S, num_experts]; zero-init, so a no-op until it learns something
        expert_logits = expert_logits + self.loop_bias(loop_idx)

        # single softmax over the logits gives both the load balancing signal and the selection distribution
        expert_scores = F.softmax(expert_logits / temperature, dim=-1)

        # one topk, reused for both the aux loss and the selection (the aux loss only needs the
        # indices, which the score normalization below does not change)
        topk_scores, topk_indices = torch.topk(expert_scores, self.top_k, dim=-1) # [batch_size, seq_len, top_k], [batch_size, seq_len, top_k]
        load_balancing_loss = compute_aux_loss(topk_indices, expert_scores, self.num_experts, token_mask=token_mask)

        # normalize the topk scores
        topk_scores = topk_scores / torch.sum(topk_scores, dim=-1, keepdim=True)

        # track expert selection stats
        self.expert_tracker.update(topk_indices, topk_scores, expert_scores)

        return topk_scores, topk_indices, load_balancing_loss

    def _selector_chunk_mass(self, memory) -> torch.Tensor:
        """this loop's selector chunk scores, ``[M]``, averaged over however many IR experts exist.

        Reads ``InformationRetrievalModule.last_memory_weights``, which the non-MLP expert loop in
        ``forward_step`` just populated for every IR expert in the pool (this method is only called
        after that loop runs). Returns None when nothing retrieved this loop -- there is no IR
        expert in the pool at all -- which the caller treats identically to no selector attached.
        Averaging rather than picking one expert also means this does not have to change if
        ``num_ir_experts`` ever grows past 1.
        """
        weights = [m.last_memory_weights for m in self.ir_modules if m.last_memory_weights is not None]
        if not weights:
            return None
        visible = memory[1]
        return torch.stack([chunk_mean_mass(w, visible) for w in weights], dim=0).mean(dim=0)

    def forward_step(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, other: torch.Tensor = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, loop_idx: int = 0, kv_cache=None, inject_bias: torch.Tensor = None, evidence: EvidenceBatch = None, memory=None, token_mask: torch.Tensor = None):
        # what the router and the experts READ. The residual update at the bottom still adds to
        # `hidden_states` itself, so the injection reaches a readout only through what the experts
        # compute from it -- it re-presents the input, it does not write to the stream lm_head sees.
        step_input = hidden_states if inject_bias is None else hidden_states + inject_bias

        topk_scores, topk_indices, load_balancing_loss = self.route(
            step_input, temperature=self.temperature, loop_idx=loop_idx, token_mask=token_mask
        )

        # index placement in the scores would be:
        # [attn_experts..., num_ir_experts..., ff_experts...]
        _other = other if other is not None else step_input

        # compute each non-MLP expert exactly once per forward_step, then cache across k slots
        # (attention runs over the full sequence regardless of routing, so recomputing per k-slot
        # wastes compute). Run BEFORE the always-on shared experts below -- deliberately, not just
        # incidentally: the evidence reader's per chunk gate (further down) reads THIS loop's own
        # selector output, and the selector is one of these experts. Moving WHEN they are computed
        # does not move the ORDER `output`'s summation happens in further below, so this changes
        # nothing about a step without evidence, or without an IR expert at all.
        expert_cache = []
        for i in range(self.first_mlp_index):
            slot_cache = kv_cache.slots[i] if kv_cache is not None else None
            if isinstance(self.experts[i], InformationRetrievalExpert):
                # loop_idx is passed for the entropy instrumentation only (see forward's docstring)
                expert_cache.append(self.experts[i](step_input, cu_seqlens, max_seqlen, position_embeddings, kv_cache=slot_cache, loop_idx=loop_idx, memory=memory))
            elif isinstance(self.experts[i], SelfAttention):
                expert_cache.append(self.experts[i](step_input, cu_seqlens, max_seqlen, position_embeddings, kv_cache=slot_cache))
            elif isinstance(self.experts[i], CrossAttention):
                expert_cache.append(self.experts[i](step_input, _other, cu_seqlens, max_seqlen, position_embeddings, kv_cache=slot_cache))

        # seed with the always-on shared experts (Step 2) before accumulating routed outputs
        shared_attn_cache = kv_cache.shared_attn if kv_cache is not None else None
        output = self.shared_mlp(step_input) + self.shared_attn(step_input, cu_seqlens, max_seqlen, position_embeddings, kv_cache=shared_attn_cache)
        # the evidence read joins the always-on seed when a corpus is attached. Both conditions are
        # structural, not a flag: a checkpoint without the port has no module, and a batch without
        # evidence never enters this branch, so the no-corpus forward is the one that already ran.
        if self.shared_evidence is not None and evidence is not None:
            reader_evidence = evidence
            # gate the reader's evidence states by THIS loop's own selector chunk scores (the IR
            # experts computed above), segment level -- the mean over the document's own tokens of
            # that chunk's external read mass -- mapped through a learned scalar and a sigmoid. See
            # evidence_gate_scale's docstring in __init__ for why a zero-init scale makes this exactly
            # 1.0 for every chunk, not merely close to it, until the scale learns something. This
            # gives the selector a dense, always-on gradient through the reader, complementary to
            # (not a replacement for) a supervised loss on the mass split itself.
            #
            # None when there is nothing to gate by: no IR expert in the pool, or this batch's
            # memory carried no external store at all -- both leave the reader untouched, which is
            # every configuration this repo has trained before this gate existed.
            chunk_mass = self._selector_chunk_mass(memory) if memory is not None else None
            if chunk_mass is not None:
                gate = 1.0 + self.evidence_gate_scale * torch.sigmoid(chunk_mass)
                reader_evidence = apply_chunk_gate(evidence, gate)
            # per-loop conditioned query (see evidence_query_bias's docstring in __init__) and a
            # separate per-loop gain on the read alone, so loop_scale's own shrinkage of the whole
            # accumulator does not also stunt the evidence re-read on later loops. Both zero/one-
            # init neutral, so this is a no-op until either learns something.
            evidence_query = step_input + self.evidence_query_bias(self._loop_enc_row(loop_idx, step_input.dtype))
            evidence_gain = self.evidence_loop_scale[min(int(loop_idx), self.evidence_loop_scale.numel() - 1)]
            reader_output = self.shared_evidence(
                evidence_query, None, cu_seqlens, max_seqlen, position_embeddings, evidence=reader_evidence
            )
            # kept for a groundedness readout, which asks about the READ rather than about what the
            # loop's accumulator did with it -- so it wants this, before the per-loop gain and
            # before the shared/routed sum. Same lifetime rule as the IR module's mass: overwritten
            # every loop (the last one is the most informed read) and cleared by a forward that
            # attaches no evidence, so nothing can pick up a stale batch's activations.
            self.last_reader_output = reader_output
            output = output + evidence_gain * reader_output

        # accumulate the non-MLP experts' weighted outputs. still a mask multiply (never
        # mask.sum()/boolean indexing -- that's a per-expert device sync), but the per-(slot, expert)
        # masks are folded into ONE [B, S, first_mlp_index] gate tensor first: the old
        # top_k * first_mlp_index nested loop built that many [B, S, H] intermediates and kept every
        # one of them alive for backward.
        if self.first_mlp_index > 0:
            dense_slot = topk_indices < self.first_mlp_index
            dense_gate = torch.zeros(
                *topk_indices.shape[:-1], self.first_mlp_index,
                device=topk_scores.device, dtype=topk_scores.dtype,
            )
            dense_gate.scatter_add_(
                -1,
                torch.where(dense_slot, topk_indices, torch.zeros_like(topk_indices)),
                topk_scores * dense_slot.to(topk_scores.dtype),
            )
            for i, cached in enumerate(expert_cache):
                output = output + dense_gate[..., i].unsqueeze(-1) * cached

        # routed tokens to parallel experts. non-MLP slots collapse to (index 0, weight 0) via
        # mask multiply -- never mask.sum()/boolean indexing (per-expert device sync)
        mlp_mask = topk_indices >= self.first_mlp_index
        mlp_indices = torch.where(mlp_mask, topk_indices - self.first_mlp_index, torch.zeros_like(topk_indices))
        mlp_scores = torch.where(mlp_mask, topk_scores, torch.zeros_like(topk_scores))

        parallel_output = self.parallel_experts(
            step_input,
            mlp_scores,
            mlp_indices
        )
        output = output + parallel_output

        # residual update, not a replacement: a gradient path across loop boundaries that doesn't
        # depend on which expert got routed to. loop_scale is per-loop (see __init__); loop indices
        # past the trained count reuse the last entry so extra inference-time loops keep the final
        # loop's learned gain rather than falling off a table.
        loop_scale = self.loop_scale[min(int(loop_idx), self.loop_scale.numel() - 1)]
        delta = self.dropout(self.post_norm(output))
        hidden_states = hidden_states + loop_scale * delta

        return hidden_states, load_balancing_loss


    def forward(
        self,
        hidden_states: torch.Tensor,
        other: torch.Tensor = None,
        return_loss: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        use_checkpointing: bool = False,
        n_loops: int = None,
        kv_cache: list = None,
        position_offset: int = 0,
        exit_check=None,
        evidence: EvidenceBatch = None,
        token_mask: torch.Tensor = None,
    ):
        """
        Args:
            evidence (EvidenceBatch, optional): retrieved evidence for this batch, read by the
                always-on evidence port at **every** loop rather than consumed once -- a later loop
                returns to the same evidence with a query the earlier loops have moved. Defaults to
                None, which is the forward this model ran before the port existed, bit for bit.
            n_loops (int, optional): overrides the configured loop count for this call. Both the
                per-loop router bias and ``loop_scale`` are indexed by absolute loop index (with
                out-of-range indices reusing the last trained entry), so a checkpoint trained at
                ``n_loops=3`` can be run with more or fewer loops without reshaping any weight.
                Defaults to None (use the configured ``self.n_loops``).
            kv_cache (list, optional): one ``_MoELoopCache`` per loop (``KVCache.moe``, see
                ``modules/model/kv_cache.py``), for incremental decoding. Its length must equal
                ``n_loops``. ``hidden_states`` covers only the newly-appended tokens in this mode.
                Defaults to None (the normal packed/varlen training and full-recompute path).
            position_offset (int, optional): absolute position of ``hidden_states[:, 0]``, only
                used to slice RoPE when ``kv_cache`` is given. Defaults to 0.
            exit_check (callable, optional): ``(loop_idx, hidden_states) -> bool``, consulted after
                each loop; a True return stops the recurrence early. This is the parameter-free
                depth policy that replaced the deleted halt head -- the caller owns the criterion
                (``TinyMoETransformer.forward`` supplies a readout-convergence one), so nothing
                here is learned and nothing can saturate. Never set during training: the returned
                ``hidden_states_all`` would then be shorter than ``loop_ce_weights``. Defaults to
                None (always run the full depth).
            token_mask (torch.Tensor, optional): [batch_size, seq_len], True/1 for a real (non pad)
                token. Forwarded to ``route`` (hence ``compute_aux_loss``) at every loop unchanged.
                None (the default) reproduces today's aux loss exactly, over every position
                including padding -- see ``compute_aux_loss``'s own docstring for why that is a
                real, measured distortion and not just a theoretical one.
        """
        n_loops = self.n_loops if n_loops is None else int(n_loops)
        if kv_cache is not None:
            assert len(kv_cache) == n_loops, (
                f"kv_cache has {len(kv_cache)} loop slots but n_loops={n_loops}"
            )
        assert not (exit_check is not None and self.training), (
            "exit_check is inference-only: a short hidden_states_all breaks per-loop CE"
        )
        # exiting early would leave loops L+1..n-1 with no K/V entry for this token, so the NEXT
        # decode step -- which may well run to full depth -- would attend over a cache with a hole
        # in it and silently produce garbage. Filling those caches cheaply (K/V projections only,
        # skipping each skipped loop's experts) is real plumbing through every attention expert and
        # is deliberately not part of this change; until then the two features are exclusive.
        assert not (exit_check is not None and kv_cache is not None), (
            "convergence exit and the KV cache are mutually exclusive: an exited loop appends no "
            "K/V for this token, which corrupts every later step. Run with use_kv_cache=False."
        )
        total_load_balancing_loss = 0.0
        hidden_states_all = []

        self.expert_tracker.begin_forward(n_loops)
        if self.ir_tracker is not None:
            # one update per (loop, IR expert) pair, which is the cap the recompute guard needs
            self.ir_tracker.begin_forward(n_loops * self._num_ir_experts)
        # clear last recurrence's per-loop memory-mass log, not just its final entry
        for module in self.ir_modules:
            module.begin_forward()

        # rotary cos/sin for the expert attention, computed once and reused across loops/experts
        if kv_cache is not None:
            position_embeddings = self.rotary_emb.slice(position_offset, hidden_states.shape[1], hidden_states.dtype)
        else:
            position_embeddings = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])

        # the block's input, projected once and re-presented to every loop. Computed here rather
        # than inside forward_step because `hidden_states` on entry IS that input and it does not
        # change with the loop index -- one GEMM for the whole recurrence, not one per loop.
        inject_bias = self.inject(hidden_states) if self.inject is not None else None

        # the selector's view of the same evidence, built once for the whole recurrence: the chunk
        # set and the token-to-document mask do not change with the loop index, only the query that
        # scores them does. None whenever there is no external store to read, which is what keeps
        # every IR expert on the exact read path it had before the port existed.
        memory = evidence_memory(
            evidence, cu_seqlens, hidden_states.shape[0], hidden_states.shape[1], hidden_states.device
        )
        # cleared rather than left alone on a batch with no external store, so a trainer reading it
        # after the forward cannot pair this batch's read weights with the last batch's mask
        self.last_memory_visible = None if memory is None else memory[1]
        # same rule for the reader's output: a batch that attaches no evidence never enters the
        # reader branch below, so without this a groundedness readout would score the last
        # evidence-bearing batch's activations against this batch's labels
        if evidence is None:
            self.last_reader_output = None

        loops_run = 0
        for loop in range(n_loops):
            loop_cache = kv_cache[loop] if kv_cache is not None else None
            if self.training and use_checkpointing:
                hidden_states, load_balancing_loss = checkpoint(self.forward_step, hidden_states, cu_seqlens, max_seqlen, other, position_embeddings, loop, loop_cache, inject_bias, evidence, memory, token_mask, use_reentrant=False)
            else:
                hidden_states, load_balancing_loss = self.forward_step(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, other=other, position_embeddings=position_embeddings, loop_idx=loop, kv_cache=loop_cache, inject_bias=inject_bias, evidence=evidence, memory=memory, token_mask=token_mask)
            total_load_balancing_loss += load_balancing_loss
            hidden_states_all.append(hidden_states)
            loops_run += 1
            if exit_check is not None and exit_check(loop, hidden_states):
                break

        # [loops_run, B, S, H] -- per-loop hidden states, so lm_head can be applied at every loop
        # instead of only the last one. hidden_states_all[-1] is hidden_states.
        hidden_states_all = torch.stack(hidden_states_all, dim=0)

        if return_loss:
            return hidden_states, total_load_balancing_loss / loops_run, hidden_states_all
        else:
            return hidden_states, hidden_states_all

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_router_noise(self, noise_factor: float):
        """set the global multiplier on the router's exploration noise (annealed 1 -> 0 by the trainer over training). 0 disables the noise entirely."""
        self.router.noise_factor = noise_factor

    @property
    def ir_modules(self):
        """every ``InformationRetrievalModule`` in the expert pool, in router index order."""
        return [e.ir_module for e in self.experts if isinstance(e, InformationRetrievalExpert)]

    def set_ir_temperature_scale(self, scale: float):
        """set the anneal multiplier on every IR table's learned temperature.

        Driven by the trainer from the live token count, the same way the router noise anneal is.
        Lowering the temperature only at inference is not equivalent: values that have only ever
        been read as a near-uniform mixture are not individually meaningful, so a sharp read of
        them is a loss spike rather than a sharper retrieval.
        """
        for module in self.ir_modules:
            module.set_temperature_scale(scale)

    def refresh_ir_clusters(self, recycle: bool = True, dead_quantile: float = 0.0):
        """re-cluster every IR table and return one stats dict per table.

        Called on a token cadence by the trainer, never inside the step path: it is a full k-means
        over the key table plus an exact-scoring recall measurement, and it mutates the partition
        the forward pass indexes.
        """
        return [
            m.refresh_clusters(recycle=recycle, dead_quantile=dead_quantile)
            for m in self.ir_modules if m.num_clusters > 0
        ]

    def evidence_selection_term(self, chunk_gold: torch.Tensor, supervised: torch.Tensor = None):
        """Supervised selection loss over the read the forward pass just did. Scalar, or None.

        The selector's only other gradient toward RELEVANCE is the reader's per chunk gate, which
        reaches it through the reader's own zero-init output projection and only for the ~6% of
        tokens the router opened an MLP slot for -- so on its own it trains the split's magnitude
        long before it trains its ranking. This is the direct term: ``evidence_selection_loss``
        (see information_retrieval.py) against the corpus's per chunk gold flag.

        Averaged over every (loop, IR expert) read of this recurrence, not only the last one: the
        module is re-applied at every loop with a query the earlier loops moved, each application
        produces its own mass split, and supervising only the final one leaves the earlier reads --
        which is where the table demonstrably sharpens -- unsupervised. Uniform weights rather than
        ``loop_ce_weights``' ascending ones, because a relevant chunk is relevant at every depth;
        nothing here is a readout whose later loops should dominate.

        Must be called right after the forward that produced the reads, and only from a forward
        that ran WITHOUT gradient checkpointing: the stashed weights of a checkpointed segment were
        computed under ``no_grad`` and carry no graph, so the loss would be a silent no-op.

        Args:
            chunk_gold: ``[M]`` bool/float, the corpus's gold flag on the same chunk axis as
                ``EvidenceBatch.chunk_keys``/``chunk_segments``.
            supervised: optional ``[B * S]`` mask restricting the loss to supervised query
                positions (``(labels != -100).reshape(-1)``).

        Returns:
            Scalar loss, or None when this forward carried no external store to select over (no
            evidence, no IR expert, or a checkpoint without the port) -- which every caller before
            the evidence corpus existed is.
        """
        visible = self.last_memory_visible
        if visible is None or chunk_gold is None:
            return None
        terms = [
            evidence_selection_loss(weights, visible, chunk_gold, supervised=supervised)
            for module in self.ir_modules
            for weights in module.memory_weights_by_loop.values()
            if weights is not None
        ]
        if not terms:
            return None
        return torch.stack(terms).mean()



