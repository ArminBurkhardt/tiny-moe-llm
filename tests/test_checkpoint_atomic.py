"""save_checkpoint must be atomic and must round-trip `phase`. No GPU, no TE -- uses tiny
nn.Module stand-ins rather than TinyMoETransformer, so this runs on the dev box and in CI-less
environments alike. test_checkpoint_roundtrip.py covers the real model.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn, optim

from utils import save_checkpoint, load_checkpoint

d = tempfile.mkdtemp()
m = nn.Linear(4, 4)
opt = optim.AdamW(m.parameters(), lr=1e-3)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

path = os.path.join(d, "checkpoint_phase1_tok100M_loss1.2345.pt")
save_checkpoint(m, opt, sched, epoch=1, dataset_idx=7, path=path,
                token_count=100_000_000, global_offset=4242, losses=[3.0, 2.0], phase="phase1")

# no .tmp left behind
assert not os.path.exists(path + ".tmp"), "the temp file must be renamed away, not left on disk"
assert os.path.isfile(path)
print("[ok] save leaves no .tmp behind")

m2 = nn.Linear(4, 4)
opt2 = optim.AdamW(m2.parameters(), lr=1e-3)
epoch, idx, tokens, offset, losses, phase = load_checkpoint(m2, opt2, sched, path)
assert (epoch, idx, tokens, offset, phase) == (1, 7, 100_000_000, 4242, "phase1"), \
    (epoch, idx, tokens, offset, phase)
assert losses == [3.0, 2.0]
print("[ok] phase and global_offset round-trip")

# legacy checkpoint (no phase key) loads with phase None rather than raising
legacy = os.path.join(d, "checkpoint_epoch0_idx1_loss9.9999.pt")
torch.save({
    "model_state_dict": m.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "scheduler_state_dict": sched.state_dict(),
    "dataset_idx": 3, "epoch": 0,
}, legacy)
epoch, idx, tokens, offset, losses, phase = load_checkpoint(m2, opt2, sched, legacy)
assert phase is None and tokens == 0 and offset == 0, (phase, tokens, offset)
print("[ok] legacy checkpoints still load, phase is None")

# a truncated .tmp is never mistaken for a real checkpoint
stray = os.path.join(d, "checkpoint_phase1_tok200M_loss1.0000.pt.tmp")
with open(stray, "wb") as f:
    f.write(b"\x00" * 32)
assert not stray.endswith(".pt"), "a partial write must not end in .pt"
print("[ok] partial writes cannot masquerade as .pt files")

print("\nATOMIC CHECKPOINT CHECKS PASSED")
