"""``estimate_packed_rows`` replays each reader's own row rule, and the two rules differ.

The evidence reader packs to ``max_length - 1`` and skips a document whose length plus the MTP pads
exceeds that; the plain SFT reader packs to ``max_length`` and keeps it. An estimator that used the
SFT rule for both put the evidence arms' step count 7% high, so their cosine stopped at 93%.

1. A document of ``usable - num_mtp_tokens`` tokens is kept by both readers.
2. A document of ``max_length - num_mtp_tokens`` tokens is kept by the SFT reader and skipped by the
   evidence reader.
3. A document of one more is skipped by both.
4. The estimator's row count equals the reader's actual row count for each reader, in on-disk order
   and under the reader's shuffled order.

Pure index arithmetic: no model forward. Importing ``scripts.sft`` needs the CUDA env for TE.
"""
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace

import numpy as np

from modules.data.evidence_dataset import EvidenceDataset, EMBED_DIM
from modules.data.sft_dataset import SFTDataset
from scripts.prepare_evidence_data import CONDITIONS, EvidenceWriter
from scripts.sft import estimate_packed_rows

BOS, PAD = 0, 1
MAX_LEN, MTP = 16, 2
USABLE = MAX_LEN - 1
# lengths include the BOS the builders write
LENGTHS = [USABLE - MTP, MAX_LEN - MTP, MAX_LEN - MTP + 1, 5, USABLE - MTP, 3, MAX_LEN - MTP, 7]


def doc_tokens(i, length):
    return [BOS] + [10 + i] * (length - 1)


def write_sft(data_dir, split):
    offsets, ids, mask = [0], [], []
    for i, n in enumerate(LENGTHS * 3):
        ids.extend(doc_tokens(i, n))
        mask.extend([0] + [1] * (n - 1))
        offsets.append(len(ids))
    np.asarray(ids, dtype=np.uint16).tofile(os.path.join(data_dir, f"{split}.bin"))
    np.asarray(offsets, dtype=np.uint64).tofile(os.path.join(data_dir, f"{split}.idx"))
    np.asarray(mask, dtype=np.uint8).tofile(os.path.join(data_dir, f"{split}.mask"))


def write_evidence(data_dir, split):
    writer = EvidenceWriter(data_dir, split, {})
    for i, n in enumerate(LENGTHS * 3):
        ev = [100 + i, 101 + i]
        keys = np.zeros((1, EMBED_DIM), dtype=np.float32)
        writer.write(doc_tokens(i, n), [0] + [1] * (n - 1), ev, [0, 0], keys, [1],
                     CONDITIONS.index("gold"), 1)
    writer.sync()
    writer.close()


def count_rows(dataset):
    return sum(batch["input_ids"].shape[0] for batch in iter(dataset))


def main():
    tmp = tempfile.mkdtemp(prefix="rows_estimate_")
    try:
        write_sft(tmp, "sft")
        write_evidence(tmp, "ev")
        tok = SimpleNamespace(bos_token_id=BOS, pad_token_id=PAD, eos_token_id=PAD)

        kept_sft = [n + MTP <= MAX_LEN for n in LENGTHS]
        kept_ev = [n + MTP <= USABLE for n in LENGTHS]
        assert kept_sft[:3] == [True, True, False] and kept_ev[:3] == [True, False, False]
        print("1-3. fixture lengths straddle both readers' limits       PASS")

        for shuffle in (False, True):
            sft = SFTDataset(tmp, tok, batch_size=1, max_length=MAX_LEN, split="sft",
                             num_mtp_tokens=MTP, shuffle=shuffle)
            ev = EvidenceDataset(tmp, tok, batch_size=1, max_length=MAX_LEN, split="ev",
                                 num_mtp_tokens=MTP, shuffle=shuffle, max_evidence_tokens=64)
            est_sft = estimate_packed_rows(
                sft.idx_path, MAX_LEN, MTP,
                order=sft.document_order(sft.num_docs) if shuffle else None)
            est_ev = estimate_packed_rows(
                ev.idx_path, MAX_LEN, MTP, evidx_path=ev.evidx_path, max_evidence_tokens=64,
                order=ev.document_order(ev.num_docs) if shuffle else None)
            got_sft, got_ev = count_rows(sft), count_rows(ev)
            assert est_sft == got_sft, f"sft estimate {est_sft} against reader {got_sft}"
            assert est_ev == got_ev, f"evidence estimate {est_ev} against reader {got_ev}"
            print(f"4. shuffle={shuffle}: sft {est_sft}={got_sft}, evidence {est_ev}={got_ev}   PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
