import torch
from torch import nn
import transformer_engine.pytorch as te
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextAttention as GroupedQueryAttention
from modules.model.information_retrieval import InformationRetrievalModule, scale_gradient
from modules.model.evidence import EvidenceBatch

   
class SelfAttention(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.1, num_heads: int = 8, num_kv_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        self.attn = GroupedQueryAttention(
            hidden_size=input_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=input_size // num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None, evidence: EvidenceBatch = None, evidence_token_gate: torch.Tensor = None, evidence_value_scale: torch.Tensor = None) -> torch.Tensor:
        """``evidence`` adds the evidence tokens as leading keys of each query's own segment.

        The evidence states go through this expert's own norm, then the optional per token gate
        ``evidence_token_gate`` ``[B, S_ev]`` (after the norm, which would otherwise divide it back
        out), then its own ``k_proj``/``v_proj``, placed by ``evidence.prefix_position_embeddings``.
        ``evidence_value_scale`` scales their values. None is the forward this expert always ran.
        """
        x_norm = self.norm(x)
        if evidence is not None:
            states = self.norm(evidence.states)
            if evidence_token_gate is not None:
                states = states * evidence_token_gate.unsqueeze(-1).to(states.dtype)
            attn_output = self.attn(
                hidden_states=x_norm,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
                prefix_states=states,
                prefix_cu_seqlens=evidence.cu_seqlens,
                prefix_max_seqlen=evidence.max_seqlen,
                prefix_position_embeddings=evidence.prefix_position_embeddings,
                prefix_value_scale=evidence_value_scale,
            )
            return self.dropout(attn_output)
        attn_output = self.attn(
            hidden_states=x_norm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
        )
        return self.dropout(attn_output)


class CrossAttention(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.1, num_heads: int = 8, num_kv_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        self.attn = GroupedQueryAttention(
            hidden_size=input_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=input_size // num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, other: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None, evidence: EvidenceBatch = None) -> torch.Tensor:
        """``evidence`` replaces ``other`` as the key/value side when a corpus is attached.

        The two differ in more than content. ``other`` is one per-token tensor aligned with ``x``, so
        it inherits the queries' segmentation, their position basis and causal masking. Evidence is a
        different set of tokens entirely: its own segments (one evidence set per query segment), its
        own per chunk position basis, and no causal order relative to the queries -- a retrieved
        passage is not "before" or "after" the token reading it.
        """
        x_norm = self.norm(x)
        if evidence is None:
            attn_output = self.attn(
                hidden_states=x_norm,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                other_states=other,
                kv_cache=kv_cache,
            )
        else:
            attn_output = self.attn(
                hidden_states=x_norm,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                other_states=evidence.states,
                kv_cache=None,
                cu_seqlens_k=evidence.cu_seqlens,
                max_seqlen_k=evidence.max_seqlen,
                other_position_embeddings=evidence.position_embeddings,
                causal=False,
            )
        return self.dropout(attn_output)

class InformationRetrievalExpert(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_entries: int,
        ir_dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = False,
        num_clusters: int = 0,
        probe_clusters: int = 4,
        read_top_k: int = 32,
        memory_dim: int = 0,
        direct_read: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        # the inner attention exists only for the averaged output stage (direct_read off, the A/B
        # path). Under the direct read it never runs, so building it left four weights per IR expert
        # with no gradient (DDP refuses that without find_unused_parameters) and billed their FLOPs.
        # utils.load_model_state drops the tensors an older checkpoint carries for it.
        self.attn = None
        if not direct_read:
            self.attn = GroupedQueryAttention(
                hidden_size=input_size,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=input_size // num_heads,
                dropout=dropout,
            )

        # operate on the per-head dimension for more efficient retrieval
        self.ir_module = InformationRetrievalModule(
            num_entries=num_entries,
            latent_dim=ir_dim,
            output_dim=ir_dim,
            temperature=1.0,
            use_min_dist=False,
            residual=residual,
            num_clusters=num_clusters,
            probe_clusters=probe_clusters,
            read_top_k=read_top_k,
            memory_dim=memory_dim,
        )
        self.down_proj = te.Linear(input_size, ir_dim, bias=False)
        self.up_proj = te.Linear(ir_dim, input_size, bias=False)

        # the direct read path (see forward()'s docstring for the failure it fixes). Zero-init, for
        # the same reason g_proj is zeroed by the reshape migration: turning this on for a checkpoint
        # that has never trained it must be measurable as the "read zeroed" ablation this file
        # already reports at 0.0002 nats, not as an untrained transform injecting noise into a
        # converged trunk. A full projection rather than a bare scalar gate, so it can learn WHICH
        # output channels of a token's own read matter, not just how loud the read is overall.
        self.direct_read = bool(direct_read)
        self.direct_gate = None
        if self.direct_read:
            self.direct_gate = te.Linear(input_size, input_size, bias=False)
            nn.init.zeros_(self.direct_gate.weight)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None, loop_idx: int = 0, memory=None) -> torch.Tensor:
        """``memory`` is the external store the table is read alongside, or None for the table alone.

        This expert is the SELECTOR half of the evidence port: it decides how much of a token's read
        comes from outside the weights, and that decision is what the groundedness signal is. It
        does not carry the evidence's text -- an external chunk vector is a summary of a whole
        passage and no span can be copied out of it, which is why the reader is a separate module
        over the evidence TOKENS.

        Note what routing does and does not reach. This expert is in the router pool, so a token the
        router did not select it for has its retrieved value multiplied by a zero gate and the read
        never reaches the residual stream. The mass split is unaffected: every non-MLP expert runs
        unconditionally once per step, so the split is computed for every token whether or not its
        gate is open. So the groundedness readout covers the whole batch, while the gradient that
        trains the adapters only arrives through the routed fraction -- which is the argument for
        watching the split's AUROC rather than the adapters' norms when judging whether it is
        learning.

        Two output stages, selected by ``direct_read`` at construction:

        - ``direct_read=True`` (default): each token's own retrieved vector (``information`` below)
          reaches the output through ``direct_gate`` alone -- no attention over any other token's
          read.
        - ``direct_read=False``: the ORIGINAL path (the inner attention is built only here), an inner ``GroupedQueryAttention`` with queries
          from the residual and keys/values from every position's own retrieved vector, causal over
          the document. This measurably does not do what it looks like it does: with ``information``
          at RMS 0.004-0.008 and a default-init ``k_proj``, the attention logits' std is on the order
          of 1e-3, so the softmax is uniform to a fraction of a percent and the expert emits
          ``v_proj`` of the running MEAN of the prefix's reads -- a per document constant, not a per
          token one. Kept only so the two paths stay A/B comparable on the same checkpoint family.
        """
        x_norm = self.norm(x)

        down = self.down_proj(x_norm)
        # loop_idx only buckets the retrieval entropy instrumentation (see RetrievalEntropyTracking);
        # the retrieval itself is loop independent, which is exactly what the Stage 0 query drift
        # measurement found and what NEXT.md's loop conditioned query is meant to change
        ir_output = self.ir_module(down, loop_idx=loop_idx, memory=memory)
        if (
            self.ir_module.stash_query_input
            and torch.is_grad_enabled()
            and int(loop_idx) in self.ir_module.memory_query_by_loop
        ):
            self.ir_module.query_input_by_loop[int(loop_idx)] = x
        information = self.up_proj(ir_output)

        if self.direct_read:
            # each position's own read, gated straight to the output -- no mixing across the
            # document (see the class docstring for why the alternative measurably does not do what
            # it looks like it does). kv_cache is accepted but unused here: there is no attention to
            # cache against, so a generation loop simply never populates this expert's slot.
            return self.dropout(self.direct_gate(information))

        attn_output = self.attn(
            hidden_states=x_norm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            other_states=information,
            kv_cache=kv_cache,
        )
        return self.dropout(attn_output)

    def contrastive_query(self, rows: torch.Tensor, loop_idx: int, trunk_grad: float) -> torch.Tensor:
        """The selector's unit query recomputed from expert input rows under a scaled trunk gradient.

        Same arithmetic as the forward's query (``norm``, ``down_proj``, the module's per-loop bias,
        normalize), so the value matches the stashed ``memory_query_by_loop`` row; only the gradient
        that flows back into ``rows`` is scaled, which leaves the query head's own tensors (``norm``,
        ``down_proj``, ``loop_query_bias``) on their full gradient and the IR read itself untouched.

        Args:
            rows: ``[P, input_size]`` expert input rows at the query positions.
            loop_idx: the loop whose query is rebuilt.
            trunk_grad: the share of the gradient that reaches ``rows``, in [0, 1].

        Returns:
            ``[P, ir_dim]`` unit query rows.
        """
        rows = scale_gradient(rows, trunk_grad)
        return self.ir_module.unit_query(self.down_proj(self.norm(rows)), loop_idx)


def grouped_mlp_forward(layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                        null_mask: torch.Tensor = None) -> torch.Tensor:
    """Sparse MLP dispatch of ``ParallelSparseMoELayer`` that never runs the null picks.

    Picks of a dense (attention or IR) slot arrive as ``(index 0, weight 0)``. They sort after every
    real expert, are counted in the same ``bincount`` host sync, and are sliced off before the
    grouped GEMM, so MLP expert 0 only sees its own rows. Their output is exactly zero, as the weight
    multiply made it before.

    Args:
        layer: the ``ParallelSparseMoELayer`` owning ``gate_up``, ``down``, ``activation`` and
            ``num_experts``.
        x: ``[B, S, H]`` layer input.
        topk_weights: ``[B, S, top_k]`` routing weights, zero on dense slots.
        topk_indices: ``[B, S, top_k]`` local MLP expert ids, zero on dense slots.
        null_mask: optional ``[B, S, top_k]`` bool, True on dense slots. Defaults to
            ``topk_weights == 0``, which also drops a real pick whose weight underflowed to zero.

    Returns:
        ``[B, S, H]`` weighted expert output.
    """
    B, S, H = x.shape
    top_k = topk_indices.shape[-1]
    x_flat = x.reshape(-1, H)
    T = x_flat.shape[0]
    num_experts = layer.num_experts

    idx = topk_indices.reshape(-1)
    wgt = topk_weights.reshape(-1)
    null = (wgt == 0) if null_mask is None else null_mask.reshape(-1)
    key = torch.where(null, torch.full_like(idx, num_experts), idx)
    tok = torch.arange(T, device=x.device).repeat_interleave(top_k)

    order = torch.argsort(key, stable=True)
    key_sorted = key[order]
    counts = torch.bincount(key_sorted, minlength=num_experts + 1).tolist()  # the one host sync
    m_splits = counts[:num_experts]
    n_real = sum(m_splits)
    if n_real == 0:
        return torch.zeros(B, S, H, device=x.device, dtype=x.dtype)

    order_real = order[:n_real]
    tok_sorted = tok[order_real]
    wgt_sorted = wgt[order_real]
    x_sorted = x_flat.index_select(0, tok_sorted)

    with te.autocast(enabled=False):
        gate, up = layer.gate_up(x_sorted, m_splits).chunk(2, dim=-1)
        out_sorted = layer.down(layer.activation(gate) * up, m_splits)

    out_sorted = out_sorted * wgt_sorted.unsqueeze(-1)
    combined = torch.zeros(T, H, device=x.device, dtype=out_sorted.dtype)
    combined.index_add_(0, tok_sorted, out_sorted)
    return combined.view(B, S, H)
