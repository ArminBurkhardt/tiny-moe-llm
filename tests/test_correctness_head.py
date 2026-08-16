"""Acceptance test for PLAN.md Step 4b's correctness head (correct_proj).

1. Gradient-leak check: `main_lm_head`'s own gradient must be bit-identical whether the
   correctness term is included or not -- `is_correct` is a no_grad target derived from the CE
   logits, and correct_proj is a separate module reading the same hidden state, so the two loss
   terms must not interact at the CE path. This is the exact scenario PLAN.md Step 4b's acceptance
   warns about ("no gradient leak through is_correct").
2. Smoke check: after a short overfit-style run, mean sigmoid(correct_proj(hidden)) ("p_correct")
   should track batch top-1 accuracy reasonably closely -- a real calibration check is Gate 5's
   job (scripts/eval_calibration.py, not written yet), this is only a sanity floor. GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss

torch.manual_seed(0)
dev = "cuda"
N_LOOPS = 2
LOOP_CE_WEIGHTS = [0.3, 1.0]
P = dict(vocab_size=512, max_seq_len=256, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=N_LOOPS, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2)

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

# --- 1. gradient-leak check: isolate to just the two heads by detaching the hidden states, so
#        the only thing backward() can populate gradients for is main_lm_head/correct_proj -- if
#        adding the correctness term changed main_lm_head's own gradient, that would mean
#        is_correct (or something else in the conf branch) is leaking into the CE computation.
model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
model.set_checkpointing(False, False)
model.delayed_mtp_loss(True)
with torch.no_grad():
    hidden_all, _, _, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                                    return_aux_loss=True, return_hidden=True)
hidden_all = hidden_all.detach()

model.zero_grad(set_to_none=True)
loss_a, loss_ce_a = compute_mtp_loss(hidden_all, labels, main_lm_head=model.lm_head,
                                      pad_mask=pad_mask, loop_ce_weights=LOOP_CE_WEIGHTS,
                                      correct_proj=None, lambda_conf=0.0)
loss_a.backward()
grads_a = [p.grad.clone() for p in model.lm_head.parameters()]

model.zero_grad(set_to_none=True)
loss_b, loss_ce_b = compute_mtp_loss(hidden_all, labels, main_lm_head=model.lm_head,
                                      pad_mask=pad_mask, loop_ce_weights=LOOP_CE_WEIGHTS,
                                      correct_proj=model.correct_proj, lambda_conf=0.05)
loss_b.backward()
grads_b = [p.grad.clone() for p in model.lm_head.parameters()]

assert torch.equal(loss_ce_a, loss_ce_b), f"loss_ce changed with correct_proj enabled: {loss_ce_a} vs {loss_ce_b}"
for ga, gb in zip(grads_a, grads_b):
    assert torch.equal(ga, gb), "lm_head gradient changed when the correctness term was added -- gradient leak through is_correct"
g = next(p.grad for p in model.correct_proj.parameters() if p.grad is not None)
assert torch.count_nonzero(g).item() > 0, "correct_proj received no gradient from its own loss term"
print("[ok] lm_head gradient (and loss_ce) unaffected by lambda_conf/correct_proj -- no leak through is_correct")
print("[ok] correct_proj itself receives a nonzero gradient from the conf loss")

# --- 2. smoke check: after overfitting, mean p_correct should track batch top-1 accuracy ---
torch.manual_seed(0)
model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
model.set_checkpointing(False, False)
model.delayed_mtp_loss(True)
opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

for step in range(300):
    hidden, aux, p_halt, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                             return_aux_loss=True, return_hidden=True)
    loss, loss_ce = compute_mtp_loss(hidden, labels, mtp_outputs=mtp, lm_head=model.mtp_head.lm_head,
                            lambda_mtp=0.1, main_lm_head=model.lm_head, pad_mask=pad_mask,
                            loop_ce_weights=LOOP_CE_WEIGHTS, correct_proj=model.correct_proj,
                            lambda_conf=0.05)
    total = loss + aux
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 60 == 0:
        print(f"step {step}: loss={loss.item():.4f} loss_ce={loss_ce.item():.4f}")

model.eval()
with torch.no_grad():
    hidden, aux, p_halt, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=maxlen,
                             return_aux_loss=True, return_hidden=True)
    final = hidden[-1, :, :-1, :].contiguous()
    main_labels = labels[:, 1:].contiguous()
    logits = model.lm_head(final)
    valid = main_labels != -100
    top1_acc = ((logits.argmax(-1) == main_labels) & valid).sum().float() / valid.sum()
    p_max = logits.float().softmax(-1).max(-1).values[valid].mean()
    p_correct = torch.sigmoid(model.correct_proj(final).squeeze(-1))[valid].mean()

print(f"batch top-1 accuracy: {top1_acc.item():.4f}")
print(f"mean p_correct:       {p_correct.item():.4f}")
print(f"mean p_max:           {p_max.item():.4f}  (logged alongside p_correct per PLAN.md Step 4b)")

assert abs(p_correct.item() - top1_acc.item()) < 0.15, (
    f"mean p_correct ({p_correct.item():.4f}) too far from batch top-1 accuracy ({top1_acc.item():.4f}); "
    "a smoke-test floor only -- real calibration check is Gate 5"
)

print("\nCORRECTNESS HEAD CHECK PASSED")
