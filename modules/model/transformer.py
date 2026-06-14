import torch
from torch import nn
import torch.nn.functional as F

from modules.model.moe import LoopMixtureOfExperts
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextModel
from modules.model.modules import SmallLMHead
from modules.model.mtp import MTPHead

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
    ):
        super().__init__()
        
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
            intermediate_size=intermediate_size,
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
        identity_skew: float = 0.0,
        return_hidden=False,
    ):
        """forward pass of the model

        Args:
            input_ids (torch.Tensor): input token ids, shape [batch_size, seq_len]
            cu_seqlens (torch.Tensor, optional): int32 cumulative segment boundaries over the
                flattened [B*S] token axis for document-packed varlen attention. Defaults to None (normal causal attention).
            max_seqlen (int, optional): longest packed segment length. Defaults to None.
            return_aux_loss (bool, optional): whether to return auxiliary loss. Defaults to False.
            identity_skew (float, optional): skew for identity routing. Defaults to 0.0.

        Returns:
            torch.Tensor: output logits, shape [batch_size, seq_len, vocab_size]
            
            float (optional): auxiliary loss from MoE routing, returned if return_aux_loss is True
            
            extra_token_outputs (optional): if MTP is enabled returns either the hidden states for the extra tokens (if delayed_mtp_loss is True) or the logits for the extra tokens (if delayed_mtp_loss is False)
            
            If delayed_mtp_loss is True, the shape of extra_token_outputs is [batch_size, seq_len, num_extra_tokens, hidden_size // 2]
            
            If delayed_mtp_loss is False, the shape of each element in extra_token_outputs is [batch_size, seq_len, vocab_size]
        """
        self._token_tracker.count_tokens(input_ids)
        if self.training and self.use_checkpointing:
            x = checkpoint(self.gemma_decoder, input_ids, cu_seqlens, max_seqlen, use_reentrant=False)
            x, aux_loss = checkpoint(self.moe, x.last_hidden_state, self._moe_ple(input_ids), True, cu_seqlens, max_seqlen, identity_skew, self.use_sub_checkpointing, use_reentrant=False)
            x = self.norm(x)
            extra_token_outputs = self._mtp_forward(x, use_checkpointing=self.use_sub_checkpointing)
            if not return_hidden:
                x = self.lm_head(x)
        else:
            x = self.gemma_decoder(input_ids, cu_seqlens, max_seqlen).last_hidden_state
            x, aux_loss = self.moe(x, other=self._moe_ple(input_ids), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_loss=True, identity_skew=identity_skew)
            x = self.norm(x)
            extra_token_outputs = self._mtp_forward(x, use_checkpointing=False)
            if not return_hidden:
                x = self.lm_head(x)
        
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
