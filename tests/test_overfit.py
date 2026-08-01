"""End-to-end learning check: the full model + loss + packed attention must be able to
overfit one small batch (loss -> near 0). Catches broken gradient flow, wrong label
alignment, dead routing, etc. GPU required. Runs in BF16 like real training.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss

torch.manual_seed(0)
dev = "cuda"
P = dict(vocab_size=512, max_seq_len=256, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=2, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2)

model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
model.set_checkpointing(False, False)
model.delayed_mtp_loss(True)
opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

B, S = 2, 128
pad_id = 1
input_ids = torch.randint(2, P["vocab_size"], (B, S), device=dev)
half = S // 2
# two docs per row, 2 MTP pads after each (mirrors dataset layout)
input_ids[:, half-2:half] = pad_id
input_ids[:, S-2:] = pad_id
row = [0]*half + [1]*(half-2) + [2, 3]
doc_ids = torch.tensor([row]*B, device=dev)
cu, maxlen = cu_seqlens_from_doc_ids(doc_ids)
pad_mask = input_ids == pad_id

labels = input_ids.clone()
labels[pad_mask] = -100
labels[:, 0] = -100; labels[:, half] = -100  # first token of each doc

first = last = None
for step in range(150):
    hidden, aux, p_halt, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                             return_aux_loss=True, return_hidden=True)
    loss, _ = compute_mtp_loss(hidden, labels, mtp_outputs=mtp, lm_head=model.mtp_head.lm_head,
                            lambda_mtp=0.1, main_lm_head=model.lm_head, pad_mask=pad_mask)
    total = loss + aux
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step == 0: first = loss.item()
    if step % 30 == 0: print(f"step {step}: loss={loss.item():.4f} aux={float(aux):.4f}")
    last = loss.item()

print(f"\nloss {first:.3f} -> {last:.3f}")
assert last < first * 0.2, f"model failed to overfit a single batch: {first} -> {last}"
print("OVERFIT CHECK PASSED (gradient flow + label alignment work end to end)")
