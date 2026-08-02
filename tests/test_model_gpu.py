"""Tiny end-to-end GPU check: forward + packed (cu_seqlens) attention + MoE + MTP + backward.
Mirrors scripts/pretrain.dry_run but with small dims so it runs fast."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss

BF16 = torch.bfloat16
recipe = NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=2, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
)

def main():
    torch.manual_seed(0)
    dev = "cuda"
    model = TinyMoETransformer(**P).to(dev).to(BF16).train()
    model.set_checkpointing(True, True)
    model.delayed_mtp_loss(True)

    B, S = 2, 64
    pad_id = 0
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device=dev)
    input_ids[:, S-4:] = pad_id
    half = S // 2
    row = [0]*half + [1]*(half-4) + [2,3,4,5]   # two docs + 4 trailing length-1 pad segments
    document_ids = torch.tensor([row]*B, dtype=torch.long, device=dev)
    cu, maxlen = cu_seqlens_from_doc_ids(document_ids)
    pad_mask = (input_ids == pad_id)

    tok_before = model.token_count
    with te.autocast(enabled=True, recipe=recipe):
        logits, aux_loss, p_halt, mtp = model(
            input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
            return_aux_loss=True, return_hidden=True,
        )
        loss, _ = compute_mtp_loss(
            logits, input_ids, mtp_outputs=mtp,
            lm_head=model.mtp_head.lm_head, lambda_mtp=0.1,
            main_lm_head=model.lm_head, pad_mask=pad_mask,
            loop_ce_weights=[0.3, 1.0],  # len == P["n_loops"]
        )
        loss = loss + aux_loss
        loss.backward()

    assert torch.isfinite(loss), loss
    # return_hidden=True => model returns per-loop hidden states, not logits (PLAN.md Step 4a)
    assert logits.shape == (P["n_loops"], B, S, P["hidden_size"]), logits.shape
    assert mtp.shape == (B, S, P["mtp_num_extra_tokens"], P["hidden_size"]//2), mtp.shape
    # token tracker incremented by exactly B*S
    assert model.token_count - tok_before == B * S, (model.token_count, tok_before)
    # gradients exist and are finite on a sampled param
    g = next(p.grad for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(g).all()
    print(f"[ok] forward+backward finite loss={loss.item():.4f} aux={float(aux_loss):.4f}")
    print(f"[ok] shapes hidden={tuple(logits.shape)} mtp={tuple(mtp.shape)}")
    print(f"[ok] token_count delta = {model.token_count - tok_before} (== B*S={B*S})")

    # plain causal path (cu_seqlens=None) should also run
    model.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=recipe):
        out2 = model(input_ids=input_ids, cu_seqlens=None, max_seqlen=None,
                     return_aux_loss=True, return_hidden=True)
    assert torch.isfinite(out2[0]).all()
    print("[ok] plain-causal (cu_seqlens=None) path runs")
    print("\nMODEL GPU CHECK PASSED")

if __name__ == "__main__":
    main()
