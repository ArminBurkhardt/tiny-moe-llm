import torch
from torch import nn
import torch.nn.functional as F

from modules.model.moe import LoopMixtureOfExperts
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextModel

from torch.utils.checkpoint import checkpoint

class SmallLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        factor = 8
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_heads = nn.ModuleList([nn.Linear(hidden_size // factor, vocab_size // factor, bias=False) for _ in range(factor)])

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        slices = torch.chunk(x, len(self.lm_heads), dim=-1)
        head_outputs = [lm_head(slice) for lm_head, slice in zip(self.lm_heads, slices)]
        return torch.cat(head_outputs, dim=-1)

class TokenTracker():
    def __init__(self):
        self.num_tokens = 0
    
    def count_tokens(self, input_ids: torch.Tensor):
        self.num_tokens += input_ids.numel()
    
    def reset(self):
        self.num_tokens = 0
        
    def get_count(self):
        return self.num_tokens

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
        )
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = SmallLMHead(hidden_size, vocab_size)
        
        self.use_checkpointing = True
        self.use_sub_checkpointing = True
        
        self._token_tracker = TokenTracker()
    
    @property
    def token_count(self):
        return self._token_tracker.get_count()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None, 
        return_aux_loss=False,
        identity_skew: float = 0.0,
    ):
        """forward pass of the model

        Args:
            input_ids (torch.Tensor): input token ids, shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): attention mask. Defaults to None.
            return_aux_loss (bool, optional): whether to return auxiliary loss. Defaults to False.
            identity_skew (float, optional): skew for identity routing. Defaults to 0.0.

        Returns:
            torch.Tensor: output logits, shape [batch_size, seq_len, vocab_size]
            
            float (optional): auxiliary loss from MoE routing, returned if return_aux_loss is True
        """
        self._token_tracker.count_tokens(input_ids)
        if self.training and self.use_checkpointing:
            x = checkpoint(self.gemma_decoder, input_ids, attention_mask, use_reentrant=False)
            x = checkpoint(self.moe, x.last_hidden_state, return_aux_loss, attention_mask, identity_skew, self.use_sub_checkpointing, use_reentrant=False)
            x = self.norm(x)
            x = self.lm_head(x)
        else:
            x = self.gemma_decoder(input_ids, attention_mask=attention_mask).last_hidden_state
            x, aux_loss = self.moe(x, attention_mask=attention_mask, return_loss=True, identity_skew=identity_skew)
            x = self.norm(x)
            x = self.lm_head(x)
        return x, aux_loss if return_aux_loss else x


    def set_checkpointing(self, use_checkpointing: bool, use_sub_checkpointing: bool = None):
        """set gradient checkpointing for the model. If use_sub_checkpointing is None, it will be set to the same value as use_checkpointing.

        Args:
            use_checkpointing (bool): whether to use gradient checkpointing for the model stages (Gemma decoder and MoE)
            use_sub_checkpointing (bool, optional): whether to use gradient checkpointing for the substages within the MoE. Defaults to None.
        """
        self.use_checkpointing = use_checkpointing
        if use_sub_checkpointing is not None:
            self.use_sub_checkpointing = use_sub_checkpointing
    

