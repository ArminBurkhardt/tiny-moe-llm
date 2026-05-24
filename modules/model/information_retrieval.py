import torch
from torch import nn
import torch.nn.functional as F

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
        
        # y vectors: trainable information vectors (the 'values')
        self.y_values = nn.Parameter(torch.randn(num_entries, output_dim) * 0.02)
        
        # g(x) = final transformation layer
        # project the retrieved information back into the latent space
        self.g_proj = nn.Linear(
            in_features=output_dim + latent_dim if residual else output_dim, 
            out_features=output_dim, 
            bias=False
        )
        
        self.reset_keys()

    def reset_keys(self):
        """Ensures all z vectors are correctly initialized."""
        nn.init.normal_(self.z_keys, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_weights=False, **kwargs) -> (torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        """
        x: input tensor of shape (batch, seq_len, latent_dim) or (batch, latent_dim)
        return_weights: if True, also returns the retrieval weights (similarity scores)
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


