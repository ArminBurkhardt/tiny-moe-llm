"""Acceptance test for PLAN.md Step 4a: per-loop CE supervision.

Trains a tiny overfit-style model with ascending loop_ce_weights, then reads out each loop's own
(unweighted) chunked CE via the same lm_head. If per-loop hidden states aren't actually threaded
through to compute_mtp_loss (e.g. every loop secretly reuses the final hidden state), all loops
read out ~equal CE -- that's the bug this test catches, not just "loss went down". GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss, _chunked_linear_ce

torch.manual_seed(0)
dev = "cuda"
N_LOOPS = 3
LOOP_CE_WEIGHTS = [0.2, 0.5, 1.0]  # ascending, len == n_loops
P = dict(vocab_size=512, max_seq_len=256, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=N_LOOPS, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2)

model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
model.set_checkpointing(False, False)
model.delayed_mtp_loss(True)
opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

B, S = 2, 128
pad_id = 1
input_ids = torch.randint(2, P["vocab_size"], (B, S), device=dev)
half = S // 2
input_ids[:, half-2:half] = pad_id
input_ids[:, S-2:] = pad_id
row = [0]*half + [1]*(half-2) + [2, 3]
doc_ids = torch.tensor([row]*B, device=dev)
cu, maxlen = cu_seqlens_from_doc_ids(doc_ids)
pad_mask = input_ids == pad_id

labels = input_ids.clone()
labels[pad_mask] = -100
labels[:, 0] = -100; labels[:, half] = -100

torch.cuda.reset_peak_memory_stats()
# stop well short of full overfit (unlike test_overfit.py's 150 steps to near-zero loss). Earlier
# loops backprop-receive gradient from every later loop's CE too (that's how backprop through the
# loop recurrence works), not just their own loop_ce_weights entry -- so once training pushes deep
# into the overfit regime, later loops' *smaller* remaining headroom can flip the ordering (loop 1
# reads out lower CE than loop 2) even though nothing is broken. Sampled empirically: ordering is
# clean through ~step 24 on this seed/config and flips by ~step 26, so stop with real margin.
for step in range(18):
    hidden, aux, p_halt, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                             return_aux_loss=True, return_hidden=True)
    loss, loss_ce = compute_mtp_loss(hidden, labels, mtp_outputs=mtp, lm_head=model.mtp_head.lm_head,
                            lambda_mtp=0.1, main_lm_head=model.lm_head, pad_mask=pad_mask,
                            loop_ce_weights=LOOP_CE_WEIGHTS)
    total = loss + aux
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 6 == 0:
        print(f"step {step}: loss={loss.item():.4f} loss_ce(final loop)={loss_ce.item():.4f}")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"peak memory during training: {peak_gb:.3f} GB")

# read out each loop's own raw CE (unweighted, hidden -> chunked lm_head -> CE), same path
# compute_mtp_loss uses internally -- this is the direct signal acceptance
# criterion cares about, not just the scalar training loss.
model.eval()
with torch.no_grad():
    hidden, aux, p_halt, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                             return_aux_loss=True, return_hidden=True)
    main_labels = labels[:, 1:].contiguous().view(-1)
    per_loop_ce = []
    for loop in range(N_LOOPS):
        h = hidden[loop, :, :-1, :].contiguous().view(-1, hidden.size(-1))
        per_loop_ce.append(_chunked_linear_ce(model.lm_head, h, main_labels).item())

print(f"per-loop CE: {[f'{c:.4f}' for c in per_loop_ce]}")

# strictly decreasing loop 0 -> loop 2, with a small margin (not just "<") so this doesn't flake
# on a near-tie -- equal/flat values would mean per-loop hidden states aren't threaded through.
for i in range(1, N_LOOPS):
    assert per_loop_ce[i] < per_loop_ce[i - 1] * 0.9, (
        f"per-loop CE not clearly decreasing: loop {i-1}={per_loop_ce[i-1]:.4f} vs "
        f"loop {i}={per_loop_ce[i]:.4f} -- looks like per-loop hidden states aren't threaded through"
    )

print("\nPER-LOOP CE CHECK PASSED (loop 0 > loop 1 > ... > final loop, hidden states are threaded through)")
