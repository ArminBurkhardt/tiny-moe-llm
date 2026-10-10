"""Regression guard for the cu_seqlens-through-accelerate trap.

Reproduces the REAL input pipeline (DataLoader workers + accelerate.prepare with split_batches=True)
over a synthetic phase1.bin/.idx corpus and asserts the contract the trainer relies on:
  1. document_ids (batch-aligned, dim0 == B) survives accelerate's batch dispatch intact, so
     cu_seqlens derived from it in the training thread is correct.
  2. the dataset must NOT carry a ragged cu_seqlens (dim0 == num_segments+1) in the batch: accelerate
     truncates its dim0 to the batch size, silently corrupting the attention segmentation (which
     poisons gradients -> NaN after the first optimizer step). This test fails if that key reappears.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator

from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids


class MockTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2


# synthetic corpus: 500 documents of varying length so packing/splitting actually exercises
# the segment boundaries this test cares about
rng = np.random.default_rng(0)
docs = [rng.integers(3, 1000, size=int(n)).tolist() for n in rng.integers(50, 6000, size=500)]
tokens, offsets = [], [0]
for doc in docs:
    tokens.extend(doc)
    offsets.append(len(tokens))

d = tempfile.mkdtemp()
np.array(tokens, dtype=np.uint16).tofile(os.path.join(d, "phase1.bin"))
np.array(offsets, dtype=np.uint64).tofile(os.path.join(d, "phase1.idx"))

B, S = 3, 4096
ds = Dataset(data_dir=d, tokenizer=MockTok(), batch_size=B, max_length=S, num_mtp_tokens=2)
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
