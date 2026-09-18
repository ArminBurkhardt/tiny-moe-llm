import torch
from torch import nn
import transformer_engine.pytorch as te
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextAttention as GroupedQueryAttention
from modules.model.information_retrieval import InformationRetrievalModule
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

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None) -> torch.Tensor:
        x_norm = self.norm(x)
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
    ):
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
        """
        x_norm = self.norm(x)

        down = self.down_proj(x_norm)
        # loop_idx only buckets the retrieval entropy instrumentation (see RetrievalEntropyTracking);
        # the retrieval itself is loop independent, which is exactly what the Stage 0 query drift
        # measurement found and what NEXT.md's loop conditioned query is meant to change
        ir_output = self.ir_module(down, loop_idx=loop_idx, memory=memory)
        information = self.up_proj(ir_output)

        attn_output = self.attn(
            hidden_states=x_norm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            other_states=information,
            kv_cache=kv_cache,
        )
        return self.dropout(attn_output)

