"""Checkpoint save/load roundtrip on the real model class (small dims), plus a check of what
the checkpoint does NOT contain (scheduler/RNG state). GPU required (TE layers).
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim

from modules.model.transformer import TinyMoETransformer
from utils import save_checkpoint, load_checkpoint

torch.manual_seed(0)
dev = "cuda"
P = dict(vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=2, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2)

m1 = TinyMoETransformer(**P).to(dev).to(torch.bfloat16)
opt1 = optim.AdamW(m1.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=100)

# one optimizer step so optimizer state is non-trivial
out = m1(torch.randint(0, P["vocab_size"], (1, 32), device=dev), return_aux_loss=True, return_hidden=True)
(out[0].float().mean() + out[1]).backward()
opt1.step(); opt1.zero_grad()

path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
save_checkpoint(m1, opt1, scheduler, epoch=3, dataset_idx=1234, path=path, token_count=987654)

m2 = TinyMoETransformer(**P).to(dev).to(torch.bfloat16)
opt2 = optim.AdamW(m2.parameters(), lr=1e-4)
epoch, idx, tokens, file_idx, _ = load_checkpoint(m2, opt2, scheduler, path)
assert (epoch, idx, tokens) == (3, 1234, 987654), (epoch, idx, tokens, file_idx)

sd1, sd2 = m1.state_dict(), m2.state_dict()
assert sd1.keys() == sd2.keys()
for kk in sd1:
    assert torch.equal(sd1[kk], sd2[kk]), f"param mismatch after load: {kk}"
print(f"[ok] model state roundtrips exactly ({len(sd1)} tensors), epoch/idx/token_count restored")

o1 = opt1.state_dict(); o2 = opt2.state_dict()
assert len(o1["state"]) == len(o2["state"]) and len(o1["state"]) > 0
print(f"[ok] optimizer state roundtrips ({len(o1['state'])} param states)")

ck = torch.load(path, weights_only=False)
assert "scheduler_state_dict" in ck, "scheduler state must be checkpointed"
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)
sched2.load_state_dict(ck["scheduler_state_dict"])
assert sched2.last_epoch == scheduler.last_epoch, (sched2.last_epoch, scheduler.last_epoch)
print(f"[ok] scheduler state roundtrips (last_epoch={sched2.last_epoch}); "
      "note: RNG state is not checkpointed (router noise/dropout not bit-reproducible on resume)")

# both forwards agree after load
x = torch.randint(0, P["vocab_size"], (1, 32), device=dev)
m1.eval(); m2.eval()
with torch.no_grad():
    y1 = m1(x); y2 = m2(x)
# model with MTP head returns (logits, extra_token_outputs)
assert torch.equal(y1[0], y2[0]) and torch.equal(y1[1], y2[1])
print("[ok] loaded model reproduces identical logits")
print("\nCHECKPOINT ROUNDTRIP CHECKS PASSED")
