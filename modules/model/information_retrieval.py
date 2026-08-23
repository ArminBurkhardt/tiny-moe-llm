import math

import torch
from torch import nn
import torch.nn.functional as F


class RetrievalEntropyTracking():
    """per loop EMA of the retrieval softmax entropy, normalized as ``E / ln(num_entries)``.

    Same quantity ``scripts/eval_stage0.py`` reports as its diagnostic 1, deliberately in the same
    units, so a training log line and a Stage 0 report can be compared directly. 1.0 means a token's
    read touches the whole table uniformly, i.e. the table stores nothing (which is what both 16B
    checkpoints measured at, 0.995); Gate G2 in docs/plans/NEXT.md wants this well below 1.

    Two guards, for the same reasons ``_ExpertTracking`` in moe.py has them: the reduction only runs
    on every ``sample_interval``-th forward, because it is a full pass over the [T, num_entries]
    weight matrix and is pure instrumentation; and it is capped at the number of updates one real
    forward makes, so an activation checkpoint recompute cannot count the same loop twice. Values
    stay on device -- ``get_stats()`` is the only host sync, and the trainer calls it at its log
    cadence.
    """

    def __init__(self, num_entries: int, n_loops: int):
        self._n_loops = n_loops
        self._ln_entries = math.log(num_entries)
        self.entropy = None        # [n_loops], lazily placed on device on the first update
        self._warm = False
        # shorter window than _ExpertTracking's 256: sampling every 8th forward already smooths
        # this, and an anneal that sharpens the table is something you want to see move.
        self.window = 64
        self.sample_interval = 8
        # rows per fp32 chunk. the bf16 weights are already [B*S, num_entries]; upcasting that whole
        # tensor at once is another ~0.5GB at B*S=16384/num_entries=8192, which is real peak memory.
        # 1024 measured fastest at that shape (1.35ms vs 1.83 at 2048 and 1.90 at 512): small enough
        # that both live tensors stay in cache-friendly territory, large enough not to be launch bound
        self.chunk_rows = 1024
        self._expected_updates = None
        self._seen_updates = 0
        self._forward_counter = 0
        self._active = True

    def begin_forward(self, expected_updates: int):
        self._expected_updates = expected_updates
        self._seen_updates = 0
        self._active = (self._forward_counter % self.sample_interval) == 0
        self._forward_counter += 1

    @torch.no_grad()
    def update(self, weights: torch.Tensor, loop_idx: int = 0):
        """fold one loop's retrieval weights [.., num_entries] into that loop's EMA."""
        if not self._active:
            return  # throttled forward, skip the reduction
        if self._expected_updates is not None:
            if self._seen_updates >= self._expected_updates:
                return  # checkpoint recompute pass, already counted
            self._seen_updates += 1

        w = weights.detach().reshape(-1, weights.shape[-1])
        total = torch.zeros((), device=w.device, dtype=torch.float32)
        for start in range(0, w.shape[0], self.chunk_rows):
            # torch.special.entr is -x*log(x) in ONE kernel (and defines entr(0) = 0, so no clamp
            # is needed). The written-out -(c * c.clamp_min().log()).sum() form costs three extra
            # full-size intermediates and measured 3x slower at the real [16384, 8192] shape.
            total += torch.special.entr(w[start:start + self.chunk_rows].float()).sum()
        frac = total / max(w.shape[0], 1) / self._ln_entries

        if self.entropy is None or self.entropy.device != w.device:
            self.entropy = torch.zeros(self._n_loops, device=w.device)
            self._warm = False
        # loop indices past the trained count fold into the last slot, matching how loop_scale and
        # the router bias treat an inference time depth override
        idx = min(int(loop_idx), self._n_loops - 1)
        if not self._warm:
            # seed every slot, so the first log line reads a real entropy rather than an EMA that
            # is still climbing out of zero
            self.entropy.fill_(0.0)
            self.entropy += frac
            self._warm = True
        else:
            decay = 1.0 - 1.0 / self.window
            self.entropy[idx] = self.entropy[idx] * decay + frac * (1.0 - decay)

    def get_stats(self):
        """the per loop ``E / ln(num_entries)`` fractions, as a plain list (one host sync)."""
        if self.entropy is None:
            return []
        return self.entropy.cpu().tolist()

    def reset_stats(self):
        if self.entropy is not None:
            self.entropy.zero_()
            self._warm = False


class InformationRetrievalModule(nn.Module):
    """
    Information Retrieval Module:
    1. Normalization: x_hat = x / ||x||, z_hat = z / ||z||
    2. Similarity: s = (x_hat @ z_hat^T) / temperature
    3. Routing: w = softmax(s)
    4. Retrieval: y_ret = w @ Y_values
    5. Projection: output = y_ret @ W_g
    """
    def __init__(self, num_entries, latent_dim, output_dim, temperature=1.0, use_min_dist=False, residual=False):
        """
        Args:
            num_entries: Number of trainable z and y pairs.
            latent_dim: Dimension of the input vector `x`.
            output_dim: Dimension of the returned vector y.
            temperature: Controls the sharpness of the retrieval (lower = harder selection).
            use_min_dist: If True, retrieves the vector with the **minimum** dot product.
            residual: If True, considers `x` during output projection (output = g(y_ret, x)).
        """
        super().__init__()
        self.num_entries = num_entries
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.use_min_dist = use_min_dist
        self.residual = residual

        # retrieval keys z
        self.z_keys = nn.Parameter(torch.empty(num_entries, latent_dim))

        # optional instrumentation, attached from outside (LoopMixtureOfExperts shares one tracker
        # across every IR expert). None means the module is a plain retrieval module again
        self.tracker: RetrievalEntropyTracking = None

        # y vectors: trainable information vectors (the 'values')
        self.y_values = nn.Parameter(torch.randn(num_entries, output_dim) * 0.02)
        
        # g(x) = final transformation layer
        # project the retrieved information back into the latent space
        import transformer_engine.pytorch as te
        self.g_proj = te.Linear(
            in_features=output_dim + latent_dim if residual else output_dim, 
            out_features=output_dim, 
            bias=False
        )
        
        self.reset_keys()

    def reset_keys(self):
        """Ensures all z vectors are correctly initialized."""
        nn.init.normal_(self.z_keys, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_weights=False, loop_idx: int = 0, **kwargs) -> (torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        """
        x: input tensor of shape (batch, seq_len, latent_dim) or (batch, latent_dim)
        return_weights: if True, also returns the retrieval weights (similarity scores)
        loop_idx: which loop of the MoE recurrence this call belongs to. Only used to bucket the
            entropy instrumentation per loop; the retrieval itself does not depend on it
        """
        # handle 2D or 3D inputs
        original_shape = x.shape
        if x.dim() == 3:
            x_flat = x.view(-1, self.latent_dim)
        else:
            x_flat = x
        
        # 1. normalization
        # for Cosine Similarity (via dot product)
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        z_norm = F.normalize(self.z_keys, p=2, dim=-1)

        # 2. similarity <x, z>
        # [batch, latent_dim] @ [latent_dim, num_entries] -> [batch, num_entries]
        logits = torch.matmul(x_norm, z_norm.t())
        
        if self.use_min_dist:
            logits = -logits # flip to find the minimum dot product
            
        logits = logits / self.temperature
        
        # 3. retrieval process
        weights = F.softmax(logits, dim=-1) # [batch, num_entries]

        # instrumentation only: no grad, throttled, and reading the weights that already exist here
        # rather than recomputing them (which is what eval_stage0.py has to do from outside)
        if self.tracker is not None:
            self.tracker.update(weights, loop_idx)

        # 4. retrieve the information vector y
        # [batch, num_entries] @ [num_entries, output_dim] -> [batch, output_dim]
        retrieved_y = torch.matmul(weights, self.y_values)
        
        # restore original shape (batch, seq_len, output_dim)
        if len(original_shape) == 3:
            retrieved_y = retrieved_y.view(original_shape[0], original_shape[1], -1)
        
        # 5. final projection g
        if self.residual:
            out = self.g_proj(torch.cat([retrieved_y, x], dim=-1))
        else:
            out = self.g_proj(retrieved_y)
        
        if return_weights:
            return out, weights
        return out


