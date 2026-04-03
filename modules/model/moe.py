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
        self.usage_counts = torch.zeros(0)  # Will be resized when experts are added

    def prune_least_used(self):
        """Removes the expert with the lowest usage count."""
        if len(self.experts) == 0:
            return
        
        # Find least used expert
        # usage_counts should match len(experts)
        if len(self.usage_counts) != len(self.experts):
            # Resize if out of sync (e.g. initial state)
            old_counts = self.usage_counts
            self.usage_counts = torch.zeros(len(self.experts), device=old_counts.device)
            min_len = min(len(old_counts), len(self.experts))
            self.usage_counts[:min_len] = old_counts[:min_len]

        min_idx = torch.argmin(self.usage_counts).item()
        
        # Remove from experts
        del self.experts[min_idx]
        
        # Remove from usage counts
        self.usage_counts = torch.cat([self.usage_counts[:min_idx], self.usage_counts[min_idx+1:]])
        
        # Remove from router
        # Router has 'head' which is Linear(hidden, num_experts + 1)
        # We need to slice the weights of the head to remove the column corresponding to min_idx
        with torch.no_grad():
            old_head = self.router.head
            new_head = nn.Linear(self.router.hidden_size, self.router.num_experts) # num_experts is already -1 conceptually after removal, but head includes OUTPUT
            
            # Construct new weights
            # Indices: [0...min_idx-1] U [min_idx+1...num_experts-1] U [OUTPUT]
            keep_indices = [i for i in range(self.router.num_experts) if i != min_idx]
            # OUTPUT index is self.router.num_experts (original). 
            # But the new head needs size (N-1) + 1.
            # The indices in old_head are 0..N-1 (experts) and N (Output).
            # We want to keep everything except min_idx.
            # output_index in old_head is 'num_experts'.
            
            indices_to_keep = keep_indices + [self.router.num_experts]
            indices_tensor = torch.tensor(indices_to_keep, device=old_head.weight.device)
            
            new_head.weight.data = old_head.weight.data[indices_tensor]
            new_head.bias.data = old_head.bias.data[indices_tensor]
            
            self.router.head = new_head
            self.router.num_experts -= 1

    def forward(self, x: torch.Tensor, target: torch.Tensor = None, output_skew: float = 0.0):
        cycle_len = self.steps_per_expert + 2
        cycle_pos = self.current_step % cycle_len
        
        # Ensure usage counts size
        if len(self.usage_counts) != len(self.experts):
             self.usage_counts = torch.cat([
                 self.usage_counts.to(x.device), 
                 torch.zeros(len(self.experts) - len(self.usage_counts), device=x.device)
             ])
        
        if self.training:
            if cycle_pos < self.steps_per_expert:
                # Normal routing to old experts
                probs = self.router(x, is_final=False, output_skew=output_skew)
                
                # Update usage counts (soft approximation or hard choice?)
                # User said "least used expert". Summing probs is a good proxy.
                with torch.no_grad():
                    # probs shape: [Batch, ..., NumExperts+1]
                    # ignore output expert (last one) for usage stats of experts
                    expert_probs = probs[..., :-1]
                    # Sum over batch/spatial dims
                    usage = expert_probs.sum(dim=list(range(expert_probs.ndim - 1)))
                    self.usage_counts += usage.detach()

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
                
                # Prune if needed BEFORE adding new one? Or after?
                # User: "after a given amount of training steps, the least used expert is discarded"
                # Doing it here effectively keeps population constant if we prune every cycle.
                # Let's assume external control or do it here? 
                # I'll leave it to external control via prune_least_used().
                
                new_expert = copy.deepcopy(self.expert_template)
                
                # Flatten batch and sequence dimensions for solving if necessary
                x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                target_flat = target.reshape(-1, target.shape[-1]) if target is not None and target.ndim > 2 else target
                
                new_expert.solve_from_batch(x_flat, target_flat)
                
                # Enable gradients so the expert trains as a regular layer from this
                # point on.  ``consolidate`` (ExpertModule) converts SolvableLinear to a
                # plain nn.Linear and enables grad; ``enable_grad`` (SolvableLinear) just
                # flips the flag.
                if hasattr(new_expert, 'consolidate'):
                    new_expert.consolidate(disable_grad=False, dtype=torch.float32)
                elif hasattr(new_expert, 'enable_grad'):
                    new_expert.enable_grad(True)
                
                self.experts.append(new_expert)
                self.usage_counts = torch.cat([self.usage_counts, torch.zeros(1, device=self.usage_counts.device)])

                self.router.add_experts(1)
                
                # Route only to new expert
                output = new_expert(x)
                
                # Get probs for loss computation
                # We want router to predict the new expert (index: len(experts)-1)
                probs = self.router(x, is_final=False, output_skew=output_skew)
                target_idx = len(self.experts) - 1
                
                self.current_step += 1
                return output, probs, target_idx
                
            elif cycle_pos == self.steps_per_expert + 1:
                # Route to OUTPUT expert only
                probs = self.router(x, is_final=True, output_skew=output_skew)
                
                # Assuming OUTPUT expert is identity
                output = x
                
                target_idx = self.router.output_index
                
                self.current_step += 1
                return output, probs, target_idx
        
        else:
            # Inference
            probs = self.router(x, is_final=None, output_skew=output_skew)
            
            # In inference we don't care about cycle_pos masking usually, 
            # unless we want to simulate the "training only on old experts" phase?
            # User request: "router decides how often to route... skew function... adds to probability of OUTPUT"
            # So we respect the router's decision on OUTPUT vs Experts.
            
            output = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                output += probs[..., i].unsqueeze(-1) * expert(x)
            
            # Add OUTPUT contribution (identity)
            output += probs[..., self.router.output_index].unsqueeze(-1) * x
            
            return output, probs

    def reset_step(self):
        self.current_step = 0






        
        

