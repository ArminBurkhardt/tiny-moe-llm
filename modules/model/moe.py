import torch 
from torch import nn
from modules.model.router import LatentRouter
from utils import FP64
import copy


# contains the LatentRouter and all expert modules
# Implements MoE:
# - Training:
#   - routes latents to all old experts (except OUTPUT) based on router probabilities
#   - after a given number of steps n, adds new expert and only routes to this expert and computes loss for the router to choose the new expert here
#   - after only the router was used, at step n+1, routes to OUTPUT expert only and computes loss for the router to choose OUTPUT here
# - Inference:
#   - routes latents to all experts except OUTPUT based on router probabilities
#   - after a given number of steps n, routes to all experts including OUTPUT based on router probabilities
# 
# NOTE: Once a new expert has been added during training, its desired output will be computed and the params for the expert will be solved for
#       using the expert's invertible nature (.solve_for_batch). This avoids backprop through the expert and speeds up training.

class MixtureOfExperts(nn.Module):
    def __init__(self, router: LatentRouter, experts: nn.ModuleList = None, dtype=FP64, expert: nn.Module = None, steps_per_expert: int = 100):
        super().__init__()
        self.router = router
        self.experts = experts if experts is not None else nn.ModuleList()
        self.expert_template = expert
        self.dtype = dtype
        self.steps_per_expert = steps_per_expert
        self.current_step = 0

    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        cycle_len = self.steps_per_expert + 2
        cycle_pos = self.current_step % cycle_len
        
        if self.training:
            if cycle_pos < self.steps_per_expert:
                # Normal routing to old experts
                probs = self.router(x, is_final=False)
                
                output = torch.zeros_like(x)
                for i, expert in enumerate(self.experts):
                    output += probs[..., i].unsqueeze(-1) * expert(x)
                
                self.current_step += 1
                return output
                
            elif cycle_pos == self.steps_per_expert:
                # Add new expert
                if self.expert_template is None:
                    raise ValueError("expert_template must be provided to add new experts")
                if target is None:
                    raise ValueError("Target required for solving new expert")
                
                new_expert = copy.deepcopy(self.expert_template)
                new_expert.solve_for_batch(x, target)
                
                self.experts.append(new_expert)
                self.router.add_experts(1)
                
                # Route only to new expert
                output = new_expert(x)
                
                # Get probs for loss computation
                # We want router to predict the new expert (index: len(experts)-1)
                probs = self.router(x, is_final=False)
                target_idx = len(self.experts) - 1
                
                self.current_step += 1
                return output, probs, target_idx
                
            elif cycle_pos == self.steps_per_expert + 1:
                # Route to OUTPUT expert only
                probs = self.router(x, is_final=True)
                
                # Assuming OUTPUT expert is identity
                output = x
                
                target_idx = self.router.output_index
                
                self.current_step += 1
                return output, probs, target_idx
        
        else:
            # Inference
            probs = self.router(x, is_final=None)
            
            if self.current_step < self.steps_per_expert:
                # Mask OUTPUT
                output_idx = self.router.output_index
                probs = probs.clone()
                probs[..., output_idx] = 0.0
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
            
            output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                output += probs[..., i].unsqueeze(-1) * expert(x)
            
            # Add OUTPUT contribution (identity)
            output += probs[..., self.router.output_index].unsqueeze(-1) * x
            
            self.current_step += 1
            return output

    def reset_step(self):
        self.current_step = 0






        
        

