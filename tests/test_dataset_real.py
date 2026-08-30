"""Runs the real Dataset with the real tokenizer over a handful of real fineweb documents,
pre-tokenized into a synthetic {split}.bin/.idx pair (mirroring what scripts/prepare_data.py,
PLAN.md Step 11, will produce), and checks:
- batch shapes and packing density (non-pad fraction; should be ~99% with doc splitting)
- invariants: labels mirror input_ids where unmasked, segment starts masked, only the EOS
  separator may be supervised among pad-id positions, document_ids segment the row
- BOS handling (prepended by the dataset since train.bin stores raw, BOS-less token ids)
CPU is fine (no model involved).
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from modules.data.dataset import Dataset
from utils import TOKENIZER_DIR

tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
print(f"pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id} bos_token_id={tok.bos_token_id}")

# does the tokenizer add BOS itself? (it doesn't -> dataset must prepend it)
probe = tok("hello world", add_special_tokens=True)["input_ids"]
print(f"tokenized 'hello world' -> {probe[:4]} (tokenizer adds BOS: {probe[0] == tok.bos_token_id})")

# tokenize a real fineweb shard's documents (no special tokens -- train.bin stores raw content
# ids only; BOS-prepending is the dataset's job) into a synthetic phase1.bin/.idx pair
shard_root = "data/datasets/parquet/fineweb/CC-MAIN-2021-49"
shard = sorted(os.listdir(shard_root))[0]
df = pd.read_parquet(os.path.join(shard_root, shard), columns=["content"])
texts = df["content"].dropna().tolist()[:200]

tokens, offsets = [], [0]
for text in texts:
    ids = tok(str(text), truncation=False, add_special_tokens=False)["input_ids"]
    if not ids:
        continue
    tokens.extend(ids)
    offsets.append(len(tokens))

d = tempfile.mkdtemp()
np.array(tokens, dtype=np.uint16).tofile(os.path.join(d, "phase1.bin"))
np.array(offsets, dtype=np.uint64).tofile(os.path.join(d, "phase1.idx"))

B, S = 3, 4096
ds = Dataset(data_dir=d, tokenizer=tok, batch_size=B, max_length=S, num_mtp_tokens=2)

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
