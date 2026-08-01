"""Runs the real Dataset with the real tokenizer over one fineweb parquet shard and checks:
- batch shapes and packing density (non-pad fraction; should be ~99% with doc splitting)
- invariants: labels mirror input_ids where unmasked, segment starts masked, only the EOS
  separator may be supervised among pad-id positions, document_ids segment the row
- BOS handling (prepended by the dataset since the tokenizer adds none)
CPU is fine (no model involved).
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer

from modules.data.dataset import Dataset

TOK = "ckpts/pretrained/DeepSeek-V4-Pro-tokenizer"
tok = AutoTokenizer.from_pretrained(TOK)
print(f"pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id} bos_token_id={tok.bos_token_id}")

# does the tokenizer add BOS itself? (it doesn't -> dataset must prepend it)
probe = tok("hello world", add_special_tokens=True)["input_ids"]
print(f"tokenized 'hello world' -> {probe[:4]} (tokenizer adds BOS: {probe[0] == tok.bos_token_id})")

# point a temp config at a single shard so the test is fast
shard_root = "data/datasets/parquet/fineweb/CC-MAIN-2021-49"
shard = sorted(os.listdir(shard_root))[0]
d = tempfile.mkdtemp()
root = os.path.join(d, "root"); os.makedirs(root)
os.symlink(os.path.abspath(os.path.join(shard_root, shard)), os.path.join(root, shard))
cfg = os.path.join(d, "cfg.json")  # keep cfg OUTSIDE root: FileIterator rglobs data files in root
with open(cfg, "w") as f:
    json.dump({"pretrain": [{"root": root, "column": "content"}]}, f)

B, S = 3, 4096
ds = Dataset(tok, batch_size=B, max_length=S, mode="pretrain", config_path=cfg, num_mtp_tokens=2)

it = iter(ds)
densities, n_batches = [], 4
for bi in range(n_batches):
    batch = next(it)
    ii, did, lab = batch["input_ids"], batch["document_ids"], batch["labels"]
    assert ii.shape == (B, S) and did.shape == (B, S) and lab.shape == (B, S), (ii.shape, did.shape, lab.shape)

    pad = ii == tok.pad_token_id  # NOTE: pad == eos for this tokenizer, includes EOS separators
    density = 1.0 - pad.float().mean().item()
    densities.append(density)

    # labels: unmasked labels must equal the token at that position
    m = lab != -100
    assert torch.equal(lab[m], ii[m]), "unmasked labels must mirror input_ids"
    # pad-id positions are either unsupervised or carry the EOS target (doc terminator)
    pad_labels = lab[pad]
    assert ((pad_labels == -100) | (pad_labels == tok.eos_token_id)).all(), \
        "pad positions may only be -100 or a supervised EOS"
    n_eos_supervised = int((pad_labels == tok.eos_token_id).sum())
    # document_ids must be non-decreasing per row and increment by 0/1
    d_ = did.diff(dim=1)
    assert ((d_ == 0) | (d_ == 1)).all(), "document_ids must increase by 0 or 1 along the row"
    # first token of each segment must be masked (no target for a doc's first token)
    seg_start = torch.ones_like(ii, dtype=torch.bool)
    seg_start[:, 1:] = did[:, 1:] != did[:, :-1]
    assert (lab[seg_start] == -100).all(), "first token of each segment must be -100"
    # non-pad segment starts: doc starts carry the prepended BOS, continuation chunks don't
    starts = seg_start & ~pad
    bos_frac = (ii[starts] == tok.bos_token_id).float().mean().item()
    n_segs = int(starts.sum())
    print(f"batch {bi}: density={density*100:.1f}% segments={n_segs} "
          f"starts-with-BOS={bos_frac*100:.0f}% supervised={m.float().mean().item()*100:.1f}% "
          f"eos-targets={n_eos_supervised}")
    assert n_eos_supervised > 0, "expected at least one supervised EOS separator per batch"

avg = sum(densities)/len(densities)
print(f"\navg non-pad density over {n_batches} batches: {avg*100:.1f}%")
assert avg > 0.95, f"doc splitting should yield >95% non-pad tokens, got {avg*100:.1f}%"
print("DATASET REAL-DATA CHECKS PASSED")
