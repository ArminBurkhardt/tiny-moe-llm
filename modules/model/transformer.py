import torch 
from torch import nn
from modules.model import Gemma3Encoder, Decoder, MixtureOfExperts
from modules.model.router import LatentRouter
from modules.model.linear import InvertibleLinear, SolvableLinear
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
        expert_template: nn.Module = None
    ):
        super().__init__()
        
        # Encoder: Gemma 3 1B
        # "up the layer determined to be the best for latent encodings"
        # Since we don't know the exact layer, we'll default to a mid-layer or expose it.
        # Assuming layer 12 is "best" for example, or letting user specify.
        self.encoder = Gemma3Encoder(model_dir=model_dir, target_layer=12, torch_dtype=torch.float32)
        
        # Encoder is only finetuned.
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        # Decoder: Invertible
        self.decoder = Decoder(hidden_size=latent_dim, output_size=vocab_size)
        # Decoder trained normally
        
        # Experts Core
        # "solvable experts". We need an expert template.
        if expert_template is None:
            # Simple Invertible Linear as expert? or MLP?
            # "Solvable" implies InvertibleLinear usually in this context
            expert_template = SolvableLinear(latent_dim, latent_dim)
            
        router = LatentRouter(input_size=latent_dim, num_experts=num_initial_experts, hidden_size=latent_dim)
        
        self.moe = MixtureOfExperts(
            router=router, 
            expert=expert_template,
            steps_per_expert=steps_per_expert_add,
            dtype=torch.float32
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

    def forward(self, input_ids: torch.Tensor, target_vectors: torch.Tensor = None):
        """
        Args:
            input_ids: [Batch, Seq]
            target_vectors: [Batch, Seq, VocabSize] - Required for training (solving experts)
        """
        # Encoder
        # Context is used for the decoder and potentially as initial latent?
        # Gemma3Encoder returns hidden states.
        context = self.encoder(input_ids).last_hidden_state
        
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
                    curr_z, probs, target_idx = self.moe(curr_z)
                    
                    # Router loss to force Output
                    loss = nn.CrossEntropyLoss()(probs.reshape(-1, probs.size(-1)), 
                                                torch.full(probs.shape[:-1], target_idx, device=probs.device, dtype=torch.long).reshape(-1))
                    router_loss += loss
                    break
                    
                else:
                    # Normal routing (Old Experts)
                    # "Recurrent call"
                    curr_z = self.moe(curr_z)
                    
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
                
                output, probs = self.moe(curr_z, output_skew=skew)
                
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






