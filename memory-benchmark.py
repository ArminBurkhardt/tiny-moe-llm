import torch
import time
import os
from torch.optim import AdamW
from modules.model.gemma4 import Gemma4ForCausalLM
from modules.model.moe import LoopMixtureOfExperts
from modules.model.utils import create_causal_attention_mask
from modules.model.transformer import TinyMoETransformer
from config import ModelConfig, TrainingConfig

def measure_gemma4_training_memory(
    vocab_size: int = 256000,
    max_position_embeddings: int = 8192,
    hidden_size: int = 2048,
    intermediate_size: int = 8192,
    head_dim: int = 256,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 1,
    num_hidden_layers: int = 12,
    batch_size: int = 1,
    seq_len: int = 1024,
    per_layer_embeddings_size: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    optimizer_type: str = "adamw"
) -> float:
    """
    measure peak GPU memory usage during a training step (forward and backward pass)
    for Gemma4
    
    Returns:
        peak memory usage in GB
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # clear cache and reset peak memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    timer = time.time()

    try:
        model = Gemma4ForCausalLM(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            per_layer_embeddings_size=per_layer_embeddings_size,
        ).to(device="cuda", dtype=dtype)

        # create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        if optimizer_type.lower() == "adamw":
            optimizer = AdamW(model.parameters(), lr=1e-4)
        elif optimizer_type.lower() == "bnb":
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(model.parameters(), lr=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        logits = model(input_ids)
        
        # loss
        # shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1)
        )

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        peak_memory = torch.cuda.max_memory_allocated()
        
        num_parameters = sum(p.numel() for p in model.parameters())
        
        return peak_memory / (1024**3), num_parameters, time.time() - timer

    except Exception as e:
        raise e

    finally:
        # cleanup
        del model
        if 'optimizer' in locals():
            del optimizer
        if 'logits' in locals():
            del logits
        if 'loss' in locals():
            del loss
        torch.cuda.empty_cache()

def measure_moe_training_memory(
    hidden_size: int = 2048,
    intermediate_size: int = 8192,
    num_mlp_experts: int = 8,
    top_k: int = 2,
    n_loops: int = 8,
    num_attn_experts: int = 4,
    batch_size: int = 4,
    seq_len: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
    optimizer_type: str = "adamw"
) -> float:
    """
    measure peak GPU memory usage during a training step (forward and backward pass)
    for LoopMixtureOfExperts
    
    Returns:
        peak memory usage in GB, parameter count, execution time
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # clear cache and reset peak memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    timer = time.time()

    try:
        model = LoopMixtureOfExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size//2,
            top_k=top_k,
            n_loops=n_loops,
            num_mlp_experts=num_mlp_experts,
            num_attn_experts=num_attn_experts,
            num_ir_experts=1,
            num_ir_entries=1024*4,
            ir_dim=256,
            ir_residual=False,
        ).to(device="cuda", dtype=dtype)

        # create dummy input
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, 
            device="cuda", dtype=dtype, requires_grad=True
        )
        targets = torch.randn(
            batch_size, seq_len, hidden_size, 
            device="cuda", dtype=dtype
        )
        attn_mask = create_causal_attention_mask(seq_len, dtype=torch.bool, device="cuda")

        if optimizer_type.lower() == "adamw":
            optimizer = AdamW(model.parameters(), lr=1e-4)
        elif optimizer_type.lower() == "bnb":
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(model.parameters(), lr=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        output, aux_loss = model(
            hidden_states, 
            attention_mask=attn_mask,
            return_loss=True
        )
        
        # dummy loss
        loss = torch.nn.functional.mse_loss(output, targets) + aux_loss

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        peak_memory = torch.cuda.max_memory_allocated()
        num_parameters = sum(p.numel() for p in model.parameters())
        
        return peak_memory / (1024**3), num_parameters, time.time() - timer
    
    except Exception as e:
        raise e

    finally:
        # cleanup
        del model
        if 'optimizer' in locals():
            del optimizer
        if 'output' in locals():
            del output
        if 'loss' in locals():
            del loss
        if 'aux_loss' in locals():
            del aux_loss
        if 'hidden_states' in locals():
            del hidden_states
        torch.cuda.empty_cache()

def save_model_info(model: torch.nn.Module, file_prefix: str):
    dtypes = check_all_dtypes(model)
    file_name = f"{file_prefix}_model_parameter_dtypes.txt"
    with open(os.path.join("", file_name), "w") as f:
        for name, dtype in dtypes.items():
            f.write(f"{name}: {dtype}\n")
    
    # save param distribution
    param_counts = {}
    total_params = 0
    for name, param in model.named_parameters():
        param_counts[name] = param.numel()
        total_params += param.numel()
    file_name = f"{file_prefix}_model_parameter_counts.txt"
    with open(os.path.join("", file_name), "w") as f:
        for name, count in param_counts.items():
            f.write(f"{name}: {count}\n")


def measure_transformer_training_memory(
    mtp=True,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # clear cache and reset peak memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    timer = time.time()
    
    
    try:
        model = TinyMoETransformer(
            **ModelConfig.Params,
            mtp_num_extra_tokens=2 if mtp else 0,
        ).to(device="cuda", dtype=torch.bfloat16)
        
        # create dummy input
        input_ids = torch.randint(0, ModelConfig.Params["vocab_size"], (TrainingConfig.Batch_size, TrainingConfig.Seq_length), device="cuda")
        targets = torch.randint(0, ModelConfig.Params["vocab_size"], (TrainingConfig.Batch_size, TrainingConfig.Seq_length), device="cuda")
        attn_mask = create_causal_attention_mask(TrainingConfig.Seq_length, dtype=torch.bool, device="cuda")

        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        # loss
        if mtp:
            logits, aux_loss, mtp_outputs = model(input_ids, attention_mask=attn_mask, return_aux_loss=True)
            from modules.model.mtp import compute_mtp_loss
            loss = compute_mtp_loss(
                outputs=logits, 
                targets=targets, 
                mtp_outputs=mtp_outputs, 
                lm_head=model.mtp_head.lm_head,
                lambda_mtp=0.1
            ) + aux_loss
        else:
            logits, aux_loss = model(input_ids, attention_mask=attn_mask, return_aux_loss=True)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, ModelConfig.Params["vocab_size"]), 
                shift_labels.view(-1)
            ) + aux_loss

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        peak_memory = torch.cuda.max_memory_allocated()
        
        num_parameters = sum(p.numel() for p in model.parameters())
        
        return peak_memory / (1024**3), num_parameters, time.time() - timer
    
    except Exception as e:
        raise e
    
    finally:
        # cleanup
        del model
        if 'optimizer' in locals():
            del optimizer
        if 'logits' in locals():
            del logits
        if 'loss' in locals():
            del loss
        torch.cuda.empty_cache()
    

def check_all_dtypes(model: torch.nn.Module) -> dict[str, torch.dtype]:
    """Recursively check the dtype of all parameters in the model.

    Returns:
        A dict mapping parameter names to their dtypes.
    """
    dtypes = {}
    for name, param in model.named_parameters():
        dtypes[name] = param.dtype
    # recursively check all modules with .dtype attribute (e.g. experts) and log their dtypes as well
    for module_name, module in model.named_modules():
        if hasattr(module, "dtype"):
            dtypes[module_name] = module.dtype
    return dtypes

if __name__ == "__main__":
    dtypes_to_test = [torch.bfloat16]
    
    print("--------------------------------------")
    print(" Gemma4 Causal LM Benchmark ")
    print("--------------------------------------")
    print(f"{'Dtype':<15} | {'Peak Memory (GB)':<20}")
    print("-" * 38)
    
    for dt in dtypes_to_test:
        try:
            mem, num_params, training_time = measure_gemma4_training_memory(
                dtype=dt, 
                optimizer_type="bnb", 
                batch_size=4, 
                seq_len=1024,
                per_layer_embeddings_size=None,
            )
            print(f"{str(dt):<15} | {mem:<20.4f}")
        except Exception as e:
            print(f"{str(dt):<15} | Error: {e}")
            raise e
    
    print("-" * 38)
    print(f"Model Parameters: {num_params:,}")
    print(f"Training Time: {training_time:.4f} seconds\n")

    print("--------------------------------------")
    print(" LoopMixtureOfExperts Benchmark ")
    print("--------------------------------------")
    print(f"{'Dtype':<15} | {'Peak Memory (GB)':<20}")
    print("-" * 38)
    
    for dt in dtypes_to_test:
        try:
            mem, num_params, training_time = measure_moe_training_memory(
                dtype=dt, 
                optimizer_type="bnb", 
                batch_size=4, 
                seq_len=1024,
            )
            print(f"{str(dt):<15} | {mem:<20.4f}")
        except Exception as e:
            print(f"{str(dt):<15} | Error: {e}")
            raise e
    print("-" * 38)
    print(f"Model Parameters: {num_params:,}")
    print(f"Training Time: {training_time:.4f} seconds\n")
    
    
    print("--------------------------------------")
    print(" TinyMoETransformer Benchmark ")
    print("--------------------------------------")
    print(f"{'Dtype':<15} | {'Peak Memory (GB)':<20}")
    print("-" * 38)
    
    for dt in dtypes_to_test:
        try:
            mem, num_params, training_time = measure_transformer_training_memory()
            print(f"{str(dt):<15} | {mem:<20.4f}")
        except Exception as e:
            print(f"{str(dt):<15} | Error: {e}")
            raise e
    print("-" * 38)
    print(f"Model Parameters: {num_params:,}")
    print(f"Training Time: {training_time:.4f} seconds\n")
    

"""
--------------------------------------
 Gemma4 Causal LM Benchmark 
--------------------------------------
Dtype           | Peak Memory (GB)    
--------------------------------------
torch.bfloat16  | 18.5648             
--------------------------------------
Model Parameters: 1,765,853,196
Training Time: 7.6033 seconds

--------------------------------------
 LoopMixtureOfExperts Benchmark 
--------------------------------------
Dtype           | Peak Memory (GB)    
--------------------------------------
torch.bfloat16  | 20.4216             
--------------------------------------
Model Parameters: 288,495,617
Training Time: 2.2479 seconds

--------------------------------------
 TinyMoETransformer Benchmark 
--------------------------------------
Dtype           | Peak Memory (GB)    
--------------------------------------
torch.bfloat16  | 21.2050             
--------------------------------------
Model Parameters: 578,503,687
Training Time: 4.2684 seconds
"""
