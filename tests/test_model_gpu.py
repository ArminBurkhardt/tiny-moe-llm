"""Tiny end-to-end GPU check: forward + packed (cu_seqlens) attention + MoE + MTP + backward.
Mirrors scripts/pretrain.dry_run but with small dims so it runs fast."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling, NVFP4BlockScaling

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss

BF16 = torch.bfloat16
# the full-path recipe is DelayedScaling, matching scripts/pretrain.py's chosen_recipe under
# USE_FP8=1 -- that is the only recipe training ever selects, and since SmallLMHead's sub-heads
# became te.Linear the loss path is now genuinely quantized rather than silently opting out.
recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
# NVFP4 cannot cover the LM/MTP heads at all: the MTP head's sub-heads are hidden_size//2 //
# (lm_head_factor*2) wide (48 even in the real config) and cuBLAS has no NVFP4 GEMM that narrow --
# the same "divisibility" caveat already noted at the top of scripts/pretrain.py. So NVFP4 is
# exercised over the model body only (decoder + MoE + MTP trunk), which is where it's relevant.
nvfp4_recipe = NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=2, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    # lm_head_factor=4, not the default 8: SmallLMHead's sub-heads are te.Linear, so every sub-head
    # dim must be divisible by 16 to run under this file's NVFP4 recipe. At factor=8 the MTP head
    # (which uses lm_head_factor*2) would be (256/2)/16 = 8 wide and TE rejects the GEMM. The real
    # config.yaml is already safe (192 and 48); this only bites at toy hidden sizes.
    lm_head_factor=4,
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
        logits, aux_loss, mtp = model(
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

    # NVFP4 over the model body only (see the recipe comment above). With return_hidden=True and
    # delayed_mtp_loss(True) the forward never touches a SmallLMHead, so this covers the decoder,
    # the MoE loop and the MTP trunk -- the parts ParallelSparseMoELayer's te.autocast(enabled=False)
    # row-divisibility workaround actually exists for.
    model.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=nvfp4_recipe):
        hidden, aux3, _, mtp3 = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                                      return_aux_loss=True, return_hidden=True)
        (hidden.float().pow(2).mean() + mtp3.float().pow(2).mean() + aux3).backward()
    assert torch.isfinite(hidden).all() and torch.isfinite(mtp3).all()
    g3 = next(p.grad for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(g3).all()
    print("[ok] NVFP4 body-only forward+backward finite (heads excluded: no NVFP4 GEMM that narrow)")
    print("\nMODEL GPU CHECK PASSED")

if __name__ == "__main__":
    main()
