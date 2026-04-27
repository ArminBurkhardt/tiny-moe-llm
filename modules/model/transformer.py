import torch 
from torch import nn
from modules.model import Gemma4Model, Gemma4Attention, Decoder, MixtureOfExperts, \
    ExpertModuleWithSkip, LatentRouter, InvertibleLinear, SolvableLinear, \
    GroupedQueryAttention, RoPE, SelfAttentionExpert
from modules.model.information_retrieval import InformationRetrievalModule
from utils import FP64

class FinalTransformer(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 1408,
        vocab_size: int = 262144,
        intermediate_size: int = 704,
        num_gemma_layers: int = 8,
        num_initial_experts: int = 4,
        num_attention_experts: int = 1,
        ir_num_entries: int = 256,
        steps_per_expert_add: int = 2,
        prune_step_interval: int = 1000,
        max_recurrence: int = 10,
        expert_template: nn.Module = None,
        dropout: float = 0.1,
        *args, **kwargs,
    ):
        super().__init__()
        
        if args or kwargs:
            print(f"Warning: Unused arguments passed to FinalTransformer: {args} {kwargs}")
        
        config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_gemma_layers,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "max_position_embeddings": 32000,
            "sliding_window": 512,
            "num_experts": 32,
            "num_active_experts": 4,
            "num_shared_experts": 1
        }
        
        self.encoder = Gemma4Model(config).train()
        
        self.expert_embedding = nn.Embedding(vocab_size, hidden_size)
        self.expert_attn = Gemma4Attention(
            hidden_size=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            sliding_window=config["sliding_window"],
        )

        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_dropout = nn.Dropout(dropout)
        
        if expert_template is None:
            expert_template = ExpertModuleWithSkip(
                input_size=hidden_size*2, 
                output_size=hidden_size, 
                dropout=dropout,
            ) 
        
        special_experts_list = []
        for _ in range(num_attention_experts):
            special_experts_list.append(SelfAttentionExpert(hidden_size, hidden_size, dropout=dropout))
        
        special_experts_list.append(InformationRetrievalModule(
            num_entries=ir_num_entries, 
            latent_dim=hidden_size, 
            output_dim=hidden_size, 
            residual=True
        ))
        
        special_experts = nn.ModuleList(special_experts_list)
        num_special_experts = len(special_experts)
            
        router = LatentRouter(
            input_size=hidden_size*2, 
            num_experts=num_initial_experts + num_special_experts, 
            hidden_size=hidden_size
        )
        
        self.moe = MixtureOfExperts(
            router=router, 
            expert=expert_template,
            special_experts=special_experts,
            steps_per_expert=steps_per_expert_add,
            dtype=torch.float32,
            hidden_size=hidden_size,
        )
        
        # pre populate experts using deepcopy to ensure they are distinct instances
        import copy
        for _ in range(num_initial_experts):
            new_expert = copy.deepcopy(expert_template)
            new_expert.reset()
            self.moe.experts.append(new_expert)
            self.moe.usage_counts = torch.cat([self.moe.usage_counts, torch.zeros(1, device=self.moe.usage_counts.device)])

        # invertible decoder
        self.decoder = Decoder(hidden_size=hidden_size, output_size=vocab_size)

        self.prune_step_interval = prune_step_interval
        self.max_recurrence = max_recurrence
        self.global_step = 0
        self.skew_factor = 0.5  # Adjustable hyperparameter for inference skewing (OUTPUT prob increase per expert call)

    def forward(self, input_ids: torch.Tensor, target_vectors: torch.Tensor = None, attention_mask: torch.Tensor | None = None):
        """Run a forward pass in pretraining mode or inference mode.

        Args:
            input_ids: Token indices of shape ``[Batch, Seq]``.
            target_vectors: Supervision signal in vocabulary space of shape
                ``[Batch, Seq, VocabSize]``.  Required during training so that the
                target latent ``z_target`` can be computed for solving new experts.
            attention_mask: Optional boolean attention mask of shape
                ``[Batch, Seq]`` passed through to the encoder.

        Returns:
            Training: ``(logits, router_loss)`` where ``logits`` is
            ``[Batch, Seq, VocabSize]`` and ``router_loss`` is a scalar tensor.

            Inference: ``logits`` of shape ``[Batch, Seq, VocabSize]``.
        """
        context = self.encoder(input_ids, attention_mask=attention_mask, return_hidden_states=True).last_hidden_state
        
        context = self.encoder_norm(context)
        context = self.encoder_dropout(context)

        z = context.clone()
        embeds = self.expert_embedding(input_ids)
        embeds = self.expert_attn(hidden_states=embeds, attention_mask=attention_mask)
        
        if self.training:
            assert target_vectors is not None, "Target vectors required for training"
            
            # Pruning Logic
            if self.global_step > 0 and self.global_step % self.prune_step_interval == 0:
                self.moe.prune_least_used()
            
            self.global_step += 1
            
            # Compute Target Z for the new expert solving
            # z_target = Decoder^(-1)(y_target, context)
            # We want the final output to match target_vectors.
            # So the input to the decoder (which is the output of MoE loop) should be z_target.
            with torch.no_grad():
                z_target = self.decoder.inverse(target_vectors.to(FP64), context.to(FP64))
            
            # MoE Handling
            # The MoE module handles the "Cycle" internally based on its own `current_step`.
            # If we want to align "First two expert calls" with recurrent steps 0 and 1,
            # and "Final call" with step 2 (Add Expert), we need to ensure MoE sees that.
            
            router_loss = 0
            
            # Recurrent Loop Logic matching User Curriculum
            loop_count = 0
            curr_z = z
            
            while True:
                # add embeddings to the latent for routing
                curr_z = torch.cat([curr_z, embeds], dim=-1)
                # Check what MoE will do
                cycle_len = self.moe.steps_per_expert + 2
                cycle_pos = self.moe.current_step % cycle_len
                
                # Check if we are about to Add Expert (final call logic)
                is_adding_expert = (cycle_pos == self.moe.steps_per_expert)
                
                if is_adding_expert:
                    # "Final call has only the new added expert that is being solved for"
                    # We need to provide the TARGET for this solving.
                    # The target is z_target.
                    # So we call MoE with target.
                    curr_z, probs, target_idx = self.moe(curr_z, target=z_target)
                    
                    # Compute router loss: Force router to pick new expert
                    # CrossEntropy(probs, target_idx)
                    loss = nn.CrossEntropyLoss()(probs.reshape(-1, probs.size(-1)), 
                                                torch.full(probs.shape[:-1], target_idx, device=probs.device, dtype=torch.long).reshape(-1))
                    router_loss += loss
                    
                    # After adding expert, the output of MoE is the output of the new expert.
                    break
                
                elif cycle_pos == self.moe.steps_per_expert + 1:
                    # Output Expert Mode
                    # "Route to OUTPUT expert only"
                    curr_z, probs, target_idx = self.moe(curr_z, input_ids=input_ids)
                    
                    # Router loss to force Output
                    loss = nn.CrossEntropyLoss()(probs.reshape(-1, probs.size(-1)), 
                                                torch.full(probs.shape[:-1], target_idx, device=probs.device, dtype=torch.long).reshape(-1))
                    router_loss += loss
                    break
                    
                else:
                    # Normal routing (Old Experts)
                    # "Recurrent call"
                    curr_z = self.moe(curr_z, input_ids=input_ids)
                    
                    loop_count += 1
                    
                    # Safety break if configured wrong or just training length
                    if loop_count > self.max_recurrence:
                        break
            
            # Final Decoder Pass
            # The output of the experts core goes to decoder
            final_output = self.decoder(curr_z, context)
            
            return final_output, router_loss
            
        else:
            # Inference
            # "router decides how often to route... skew function... adds to probability of OUTPUT"
            
            curr_z = z
            loop_count = 0
            
            while loop_count < self.max_recurrence:
                # Skew function: "takes the number of expert calls and adds them to the probability of the OUTPUT expert"
                # "adds them" -> Linear additive skew.
                skew = float(loop_count) * self.skew_factor  # Adjustable hyperparameter
                
                # add embeddings to the latent for routing
                curr_z = torch.cat([curr_z, embeds], dim=-1)
                
                output, probs = self.moe(curr_z, output_skew=skew, input_ids=input_ids)
                
                # Using argmax for "Decision" on per-sample basis
                # For simplicity, we check if average Output probability is dominant
                # This could be improved to per-sample dynamic exit
                output_prob = probs[..., self.moe.router.output_index].mean()
                if output_prob > 0.5: # Threshold
                    curr_z = output
                    break
                
                curr_z = output
                loop_count += 1
            
            final_output = self.decoder(curr_z, context)
            return final_output


    def sft_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for supervised fine-tuning (SFT).

        Replicates the recurrent routing loop from the inference branch of
        :meth:`forward` while keeping gradients enabled for all parameters
        (encoder, router, experts, decoder).  No expert-addition, solving, or
        pruning takes place; the MoE lifecycle is intentionally bypassed.

        The recurrence logic is identical to inference:

        1. At each iteration the latent is routed through all experts (including
           the OUTPUT expert) with a skew that grows linearly with ``loop_count``
           to nudge the router towards the OUTPUT expert over time.
        2. The loop exits early when the mean OUTPUT-expert probability exceeds
           ``0.5``, otherwise it runs for at most ``max_recurrence`` steps.

        The MoE module's ``training`` flag (and the router's) is temporarily set
        to ``False`` inside the loop so that the inference path of
        :class:`~modules.model.moe.MixtureOfExperts` is used (no expert masking).
        Setting the flag directly—rather than calling ``.eval()``—ensures that
        dropout in the encoder and decoder backbones remains active.

        Args:
            input_ids: Token indices of shape ``[Batch, Seq]``.
            attention_mask: Optional boolean attention mask of shape
                ``[Batch, Seq]`` passed through to the encoder.

        Returns:
            Logits tensor of shape ``[Batch, Seq, VocabSize]``.
        """
        # Encoder (gradients enabled; training flag controls dropout)
        context = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

        z = context.clone()

        # Temporarily override the MoE and router training flags so that the
        # inference-path routing (no masking) is used while gradients still flow.
        _moe_training = self.moe.training
        _router_training = self.moe.router.training
        self.moe.training = False
        self.moe.router.training = False

        curr_z = z
        loop_count = 0
        
        embeds = self.expert_embedding(input_ids)
        embeds = self.expert_attn(hidden_states=embeds, attention_mask=attention_mask)

        while loop_count < self.max_recurrence:
            # add embeddings to the latent for routing
            curr_z = torch.cat([curr_z, embeds], dim=-1)
                
            skew = float(loop_count) * self.skew_factor
            output, probs = self.moe(curr_z, output_skew=skew, input_ids=input_ids)

            curr_z = output
            loop_count += 1

            output_prob = probs[..., self.moe.router.output_index].mean()
            if output_prob > 0.5:
                break

        # Restore training flags before the decoder forward pass
        self.moe.training = _moe_training
        self.moe.router.training = _router_training

        logits = self.decoder(curr_z, context)
        return logits






