"""Regression guard for the cu_seqlens-through-accelerate trap.

Reproduces the REAL input pipeline (DataLoader workers + accelerate.prepare with split_batches=True)
and asserts the contract the trainer relies on:
  1. document_ids (batch-aligned, dim0 == B) survives accelerate's batch dispatch intact, so
     cu_seqlens derived from it in the training thread is correct.
  2. the dataset must NOT carry a ragged cu_seqlens (dim0 == num_segments+1) in the batch: accelerate
     truncates its dim0 to the batch size, silently corrupting the attention segmentation (which
     poisons gradients -> NaN after the first optimizer step). This test fails if that key reappears.
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids

tok = AutoTokenizer.from_pretrained("ckpts/pretrained/DeepSeek-V4-Pro-tokenizer")
shard_root = "data/datasets/parquet/fineweb/CC-MAIN-2021-49"
shard = sorted(os.listdir(shard_root))[0]
d = tempfile.mkdtemp(); root = os.path.join(d, "root"); os.makedirs(root)
os.symlink(os.path.abspath(os.path.join(shard_root, shard)), os.path.join(root, shard))
cfg = os.path.join(d, "cfg.json")
with open(cfg, "w") as f:
    json.dump({"pretrain": [{"root": root, "column": "content"}]}, f)

B, S = 3, 4096
ds = Dataset(tok, batch_size=B, max_length=S, mode="pretrain", config_path=cfg, num_mtp_tokens=2)
dl = DataLoader(ds, batch_size=None, num_workers=2, prefetch_factor=2)
acc = Accelerator(device_placement=True, split_batches=True, gradient_accumulation_steps=8)
dl = acc.prepare(dl)

for step, batch in enumerate(dl):
    # the dataset must not ship a ragged cu_seqlens through accelerate
    assert "cu_seqlens" not in batch, \
        "dataset must NOT carry cu_seqlens in the batch: accelerate truncates its dim0 to B"

    doc = batch["document_ids"]
    assert doc.shape == (B, S), f"document_ids must survive accelerate as [B, S], got {tuple(doc.shape)}"

    # cu_seqlens derived in-thread from the (intact) document_ids must be well-formed: it covers the
    # whole B*S token axis and is strictly increasing.
    cu, max_seqlen = cu_seqlens_from_doc_ids(doc)
    assert cu[0].item() == 0 and cu[-1].item() == B * S, \
        f"cu_seqlens must span 0..B*S, got [{cu[0].item()}..{cu[-1].item()}] (B*S={B*S})"
    assert (cu.diff() > 0).all(), "cu_seqlens must be strictly increasing"
    assert 0 < max_seqlen <= S, f"max_seqlen out of range: {max_seqlen}"
    print(f"step {step}: document_ids={tuple(doc.shape)} intact | "
          f"cu_seqlens segments={cu.numel() - 1} max_seqlen={max_seqlen} (well-formed)")
    if step >= 2:
        break

print("\nACCELERATE BATCH-CONTRACT CHECKS PASSED "
      "(document_ids survives; cu_seqlens stays out of the batch and is derived in-thread)")
