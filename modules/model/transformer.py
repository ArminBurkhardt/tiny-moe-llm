import torch 
from torch import nn
from modules.model import Gemma3Encoder, Decoder, MixtureOfExperts, \
    ExpertModuleWithSkip, LatentRouter, InvertibleLinear, SolvableLinear, \
    GroupedQueryAttention, RoPE, ExpertModuleWithSkipAndEmbedding
from utils import FP64

class FinalTransformer(nn.Module):
    def __init__(
        self, 
        model_dir: str, 
        latent_dim: int, 
        vocab_size: int,
        num_initial_experts: int = 4,
        steps_per_expert_add: int = 2, # "First two expert calls..." implying 2 normal calls
        prune_step_interval: int = 1000,
        max_recurrence: int = 10,
        expert_template: nn.Module = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoder: Gemma 3 1B
        # "up the layer determined to be the best for latent encodings"
        # Since we don't know the exact layer, we'll default to a mid-layer or expose it.
        # Assuming layer 12 is "best" for example, or letting user specify.
        self.encoder = Gemma3Encoder(model_dir=model_dir, target_layer=12, torch_dtype=torch.float32)
        # TODO: use Gemma4Encoder with correct downloaded model
        #       NOTE: Gemma4 already uses RoPE and GQA, so no further changes needed.
        
        # Encoder is only finetuned.
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Normalise and regularise the encoder output before it enters the MoE loop.
        # LayerNorm stabilises the latent distribution across the batch; dropout prevents
        # over-reliance on any single encoder feature.
        self.encoder_norm = nn.LayerNorm(latent_dim)
        self.encoder_dropout = nn.Dropout(dropout)
            
        # Decoder: Invertible
        self.decoder = Decoder(hidden_size=latent_dim, output_size=vocab_size)
        # Decoder trained normally
        
        # Experts Core
        # ExpertModuleWithSkipAndEmbedding is the chosen expert for the final model.
        if expert_template is None:
            expert_template = ExpertModuleWithSkipAndEmbedding(
                input_size=latent_dim, 
                output_size=latent_dim, 
                dropout=dropout, 
                num_embeddings=vocab_size
            ) 
            
        router = LatentRouter(input_size=latent_dim, num_experts=num_initial_experts, hidden_size=latent_dim)
        
        self.moe = MixtureOfExperts(
            router=router, 
            expert=expert_template,
            steps_per_expert=steps_per_expert_add,
            dtype=torch.float32,
            hidden_size=latent_dim,
        )
        # Pre-populate experts using deepcopy to ensure they are distinct instances
        import copy
        for _ in range(num_initial_experts):
            self.moe.experts.append(copy.deepcopy(expert_template))
            self.moe.usage_counts = torch.cat([self.moe.usage_counts, torch.zeros(1)])

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
        # Encoder
        context = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Normalise and regularise encoder output before feeding into the MoE loop.
        context = self.encoder_norm(context)
        context = self.encoder_dropout(context)

        # Initial latent z. 
        # Using context as z? Or separate? 
        # Usually MoE transforms z -> z'.
        z = context.clone()
        
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
            
            outputs = []
            router_loss = 0
            
            # Recurrent Loop Logic matching User Curriculum
            loop_count = 0
            curr_z = z
            
            while True:
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
                    curr_z, probs, target_idx = self.moe(curr_z, target=z_target, input_ids=input_ids)
                    
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

        while loop_count < self.max_recurrence:
            skew = float(loop_count) * self.skew_factor
            output, probs = self.moe(curr_z, output_skew=skew)

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






