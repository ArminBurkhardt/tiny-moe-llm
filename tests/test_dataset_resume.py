"""PLAN.md Step 9 acceptance: interrupting at step N and resuming from the checkpointed
global_offset must consume the next sequence with no gap or repeat, and a synthetic
train.bin/.idx with known document boundaries must produce the expected document_ids/labels.
Single-process (no DataLoader workers) so the resume point is exact, not the conservative
cross-worker minimum used in scripts/pretrain.py. CPU only.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np

from modules.data.dataset import Dataset


class MockTok:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2


def write_corpus(data_dir, docs, split="phase1"):
    tokens, offsets = [], [0]
    for doc in docs:
        tokens.extend(doc)
        offsets.append(len(tokens))
    np.array(tokens, dtype=np.uint16).tofile(os.path.join(data_dir, f"{split}.bin"))
    np.array(offsets, dtype=np.uint64).tofile(os.path.join(data_dir, f"{split}.idx"))


def batches_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in ("input_ids", "document_ids", "labels"))


def test_synthetic_boundaries_known():
    # 3 fixed-size docs of 10 content tokens each. BOS(1) + 10 content + EOS(1) + 1 pad = 13
    # tokens, so max_length=13 makes each document exactly fill one row (clean, hand-checkable
    # boundary: no cross-document packing to reason about).
    d = tempfile.mkdtemp()
    docs = [[5] * 10, [6] * 10, [7] * 10]
    write_corpus(d, docs)
    ds = Dataset(data_dir=d, tokenizer=MockTok(), batch_size=1, max_length=13, num_mtp_tokens=2)
    batches = list(iter(ds))
    assert len(batches) == 3, len(batches)
    for i, val in enumerate((5, 6, 7)):
        ii = batches[i]["input_ids"][0].tolist()
        did = batches[i]["document_ids"][0].tolist()
        # BOS + 10 content tokens + EOS + 1 pad, exactly filling the row -> single segment
        assert ii[0] == MockTok.bos_token_id, ii
        assert ii[1:11] == [val] * 10, ii
        assert ii[11] == MockTok.eos_token_id, ii  # supervised EOS separator
        assert did == [0] * 13, did                 # doc + its separator pad, one segment, no leftover
        lab = batches[i]["labels"][0].tolist()
        assert lab[0] == -100, lab                  # BOS itself never a target
        assert lab[1:11] == [val] * 10, lab          # content continuation supervised
        assert lab[11] == MockTok.eos_token_id, lab  # EOS supervised
        assert lab[12] == -100, lab                  # trailing separator pad unsupervised
    print("[ok] synthetic train.bin/.idx with known boundaries -> expected document_ids/labels")


def test_resume_no_gap_no_repeat():
    d = tempfile.mkdtemp()
    # 300 docs, each BOS(1) + 10 content + EOS(1) + 1 pad = 13 = max_length, so every document
    # fills exactly one row. This keeps the interrupt point always on a document boundary --
    # resume is document-granular by design (PLAN.md Step 9's "single global sequence offset"
    # simplification doesn't checkpoint a document's in-flight packing state), so a doc that's
    # only partially packed at the freeze point is out of scope for this check.
    docs = [[3 + (i % 50)] * 10 for i in range(300)]
    write_corpus(d, docs)

    B, S = 4, 13
    full = list(iter(Dataset(data_dir=d, tokenizer=MockTok(), batch_size=B, max_length=S, num_mtp_tokens=2)))
    assert len(full) > 10, "need enough batches for a meaningful interrupt/resume split"

    N = 5  # interrupt after N batches (N*B sequences)
    first_run = list(iter(Dataset(data_dir=d, tokenizer=MockTok(), batch_size=B, max_length=S, num_mtp_tokens=2)))[:N]
    for a, b in zip(first_run, full[:N]):
        assert batches_equal(a, b)

    # single-process (num_workers=1): the last doc_idx a fresh iterator reaches after N batches,
    # +1 (num_workers), is the exact next-wanted doc -- mirrors pretrain.py's snapshot_global_offset
    ds_interrupted = Dataset(data_dir=d, tokenizer=MockTok(), batch_size=B, max_length=S, num_mtp_tokens=2)
    it = iter(ds_interrupted)
    last_doc_idx = None
    for _ in range(N):
        b = next(it)
        last_doc_idx = int(b["doc_idx"][0])
    global_offset = last_doc_idx + 1  # num_workers=1

    resumed = list(iter(Dataset(data_dir=d, tokenizer=MockTok(), batch_size=B, max_length=S,
                                 num_mtp_tokens=2, start_doc_idx=global_offset)))

    # "no gap or repeat": interrupted-run + resumed-run must reproduce the reference run exactly,
    # sequence for sequence, starting right at N*batch_size
    stitched = first_run + resumed
    assert len(stitched) == len(full), (len(stitched), len(full))
    for i, (a, b) in enumerate(zip(stitched, full)):
        assert batches_equal(a, b), f"mismatch at batch {i} (sequence {i * B})"
    print(f"[ok] interrupt at step {N} (sequence {N*B}), resume from global_offset={global_offset} "
          "reproduces the reference run with no gap or repeat")


if __name__ == "__main__":
    test_synthetic_boundaries_known()
    test_resume_no_gap_no_repeat()
    print("\nDATASET RESUME + BOUNDARY CHECKS PASSED")
