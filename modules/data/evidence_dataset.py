"""Packed SFT batches that also carry each conversation's retrieved evidence.

``SFTDataset`` with a second token stream attached. Everything about the query side is inherited
unchanged -- the same packing rule, the same per-epoch permutation, the same position based resume
-- because the evidence is an *addition* to a row, not a different kind of row, and a second copy of
the resume arithmetic is a second thing that can drift from the trainer's ``global_offset``.

Three things are genuinely new, and each is forced by how the port reads evidence.

**One evidence segment per query segment, paired by position.** Flash pairs the two sides by
position, so document *i* of the batch reads evidence set *i*. That makes the evidence side's
segment list a parallel array to the query side's, and a document that retrieved nothing does not
drop out of it -- it contributes a zero length segment. Everything here exists to keep those two
lists the same length in the same order.

**Rows pack to ``max_length - 1``, not ``max_length``.** The evidence rows are padded to a common
width, and those pad tokens have to belong to *some* segment; they are given the row's final query
segment, which is a single trailing pad token whose output nothing reads. Reserving one position
guarantees such a segment exists. A row that happened to fill exactly would otherwise hand its
evidence padding to the last real conversation, which would then attend to it -- a contamination
that depends on packing luck and produces no error.

**Nothing ragged leaves the worker.** ``chunk_keys`` is naturally ``[total chunks, 384]``, whose
first dimension is not the batch size, and accelerate's batch splitting truncates dim 0 to the batch
size -- the same trap that keeps ``cu_seqlens`` out of the batch dict. So chunks are emitted as
``[B, C, 384]`` with a ``[B, C]`` slot map, and the trainer flattens them in-thread
(``evidence_from_batch``) once it has the query side's segmentation to number them against.

Batches carry, on top of the SFT keys:

    evidence_ids        [B, S_ev]      evidence tokens, in conversation order within each row
    evidence_chunk_ids  [B, S_ev]      chunk index within the row (-1 on padding); position restarts
    evidence_doc_slot   [B, S_ev]      which conversation of this row the token serves (-1 = padding)
    chunk_keys          [B, C, 384]    external embedder vectors
    chunk_slot          [B, C]         which conversation of this row each chunk serves (-1 = unused)

A batch in which nothing retrieved anything omits all five, which is how a pure replay batch takes
the bit-identical no-evidence forward rather than an all-padding one.
"""
import os
from typing import Iterator, List

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from modules.data.sft_dataset import SFTDataset
from utils import logger

EMBED_DIM = 384


class EvidenceDataset(SFTDataset):
    """streams packed SFT batches plus the evidence each conversation retrieved."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 4,
        max_length: int = 4096,
        split: str = "evidence_train",
        num_mtp_tokens: int = 1,
        start_doc_idx: int = 0,
        seed: int = 42,
        epoch: int = 0,
        shuffle: bool = True,
        max_evidence_tokens: int = 12288,
    ) -> None:
        """
        Args:
            max_evidence_tokens: cap on one ROW's total evidence. Conversations whose evidence would
                push a row past it start a new row instead, exactly as an over-long conversation
                does -- the evidence axis is a real memory cost (it is embedded and cross attended at
                every loop) and letting it grow with the packing would make peak memory depend on
                which conversations happened to land together. **It must be set against the corpus's
                evidence-to-prompt ratio**, not for memory alone: below that ratio times
                ``max_length`` it becomes the budget that always binds, and rows close with their
                token budget nearly untouched. ``fill`` in the packing log line is what says whether
                it is set high enough.
            (everything else: see ``SFTDataset``.)
        """
        super().__init__(
            data_dir=data_dir, tokenizer=tokenizer, batch_size=batch_size,
            max_length=max_length, split=split, num_mtp_tokens=num_mtp_tokens,
            start_doc_idx=start_doc_idx, seed=seed, epoch=epoch, shuffle=shuffle,
        )
        self.max_evidence_tokens = max_evidence_tokens
        self.ev_path = os.path.join(data_dir, f"{split}.ev")
        self.evidx_path = os.path.join(data_dir, f"{split}.evidx")
        self.evchunk_path = os.path.join(data_dir, f"{split}.evchunk")
        self.evkey_path = os.path.join(data_dir, f"{split}.evkey")
        self.evkeyidx_path = os.path.join(data_dir, f"{split}.evkeyidx")
        for path in (self.ev_path, self.evidx_path, self.evchunk_path,
                     self.evkey_path, self.evkeyidx_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} is missing -- run `python scripts/prepare_evidence_data.py` first"
                )
        # the five evidence files are indexed by the same document number as bin/idx/mask, so a
        # length disagreement means a build was interrupted between two of them and a document's
        # prompt is now paired with another document's evidence. That is not an error any later
        # check catches: both files are still internally consistent.
        for path in (self.evidx_path, self.evkeyidx_path):
            docs = os.path.getsize(path) // 8 - 1
            if docs != self.num_docs:
                raise ValueError(
                    f"{os.path.basename(path)} indexes {docs:,} documents but "
                    f"{split}.idx has {self.num_docs:,} -- the corpus is out of sync, rebuild it"
                )
        if os.path.getsize(self.evchunk_path) != os.path.getsize(self.ev_path):
            raise ValueError(f"{split}.evchunk and {split}.ev disagree -- rebuild the corpus")
        n_chunks = os.path.getsize(self.evkey_path) // (EMBED_DIM * 2)
        logger.info(
            f"EvidenceDataset[{split}]: {os.path.getsize(self.ev_path) // 2:,} evidence tokens, "
            f"{n_chunks:,} chunks"
        )

    def _batch_iterator(self) -> Iterator[dict]:
        bin_mmap = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        idx_mmap = np.memmap(self.idx_path, dtype=np.uint64, mode="r")
        mask_mmap = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        ev_mmap = np.memmap(self.ev_path, dtype=np.uint16, mode="r")
        evidx_mmap = np.memmap(self.evidx_path, dtype=np.uint64, mode="r")
        evchunk_mmap = np.memmap(self.evchunk_path, dtype=np.uint16, mode="r")
        evkey_mmap = np.memmap(self.evkey_path, dtype=np.float16, mode="r").reshape(-1, EMBED_DIM)
        evkeyidx_mmap = np.memmap(self.evkeyidx_path, dtype=np.uint64, mode="r")

        num_docs = idx_mmap.shape[0] - 1
        order = self.document_order(num_docs)

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        first = self.start_doc_idx + ((worker_id - self.start_doc_idx) % num_workers)
        logger.info(
            f"[worker {worker_id}] evidence epoch {self.epoch}: starting at position "
            f"{first}/{num_docs} (global resume point {self.start_doc_idx})"
        )

        # one position short of max_length, so every row ends with at least one pad token and
        # therefore at least one trailing length-1 segment -- see the module docstring
        usable = self.max_length - 1

        rows: List[dict] = []
        current = {"seq": [], "labels": [], "weights": [], "sections": [], "evidence": []}
        committed_position = first - num_workers
        skipped_too_long = 0
        # how full the rows actually come out. A row costs a full width forward whatever fraction of
        # it is real, so an unbalanced evidence cap is invisible in the loss and shows up only as
        # throughput -- which reads like a slow GPU rather than a corpus that is 96% padding. This
        # is the number that names it, logged early enough to kill a run over.
        packed = {"rows": 0, "tokens": 0, "evidence": 0, "closed_by_evidence": 0}

        def push_row():
            pad_len = self.max_length - len(current["seq"])
            padded = current["seq"] + [self._pad_id] * pad_len
            labels = current["labels"] + [-100] * pad_len
            weights = current["weights"] + [0.0] * pad_len

            doc_ids, seg, start = [], 0, 0
            for block_len in current["sections"]:
                block_end = min(start + block_len, self.max_length)
                if block_end > start:
                    doc_ids.extend([seg] * (block_end - start))
                    seg += 1
                start = block_end
            for _ in range(start, self.max_length):
                doc_ids.append(seg)
                seg += 1

            rows.append({
                "input_ids": padded, "document_ids": doc_ids,
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_weights": torch.tensor(weights, dtype=torch.float32),
                "evidence": list(current["evidence"]),
                "num_segments": seg,
            })
            packed["rows"] += 1
            packed["tokens"] += len(current["seq"])
            packed["evidence"] += sum(len(e["ids"]) for e in current["evidence"])
            if packed["rows"] in (16, 256, 4096):
                fill = packed["tokens"] / (packed["rows"] * self.max_length)
                ev_fill = packed["evidence"] / (packed["rows"] * self.max_evidence_tokens)
                logger.info(
                    f"[worker {worker_id}] packing over {packed['rows']} rows: fill {fill:.0%}, "
                    f"evidence {ev_fill:.0%} of its cap, {packed['closed_by_evidence']} row(s) "
                    f"closed by the evidence budget"
                )
            for key in ("seq", "labels", "weights", "sections", "evidence"):
                current[key].clear()

        def yield_batch():
            batch = {
                "input_ids": torch.tensor([r["input_ids"] for r in rows], dtype=torch.long),
                "document_ids": torch.tensor([r["document_ids"] for r in rows], dtype=torch.long),
                "labels": torch.stack([r["labels"] for r in rows]),
                "loss_weights": torch.stack([r["loss_weights"] for r in rows]),
                "doc_idx": torch.full((len(rows),), committed_position, dtype=torch.long),
                "worker_id": torch.full((len(rows),), worker_id, dtype=torch.long),
            }
            batch.update(_pack_evidence(rows))
            rows.clear()
            return batch

        for position in range(first, num_docs, num_workers):
            doc = int(order[position])
            start, end = int(idx_mmap[doc]), int(idx_mmap[doc + 1])
            if end <= start:
                continue
            tokens = bin_mmap[start:end].tolist()
            supervised = mask_mmap[start:end].tolist()
            if self._bos_id is not None and tokens[0] != self._bos_id:
                tokens = [self._bos_id] + tokens
                supervised = [0] + supervised

            ev_start, ev_end = int(evidx_mmap[doc]), int(evidx_mmap[doc + 1])
            key_start, key_end = int(evkeyidx_mmap[doc]), int(evkeyidx_mmap[doc + 1])
            ev_tokens = ev_mmap[ev_start:ev_end].tolist()
            ev_chunks = evchunk_mmap[ev_start:ev_end].tolist()
            ev_keys = np.asarray(evkey_mmap[key_start:key_end], dtype=np.float32)

            block_len = len(tokens) + self.num_mtp_tokens
            if block_len > usable or len(ev_tokens) > self.max_evidence_tokens:
                skipped_too_long += 1
                continue

            row_ev = sum(len(e["ids"]) for e in current["evidence"])
            over_tokens = len(current["seq"]) + block_len > usable
            over_evidence = row_ev + len(ev_tokens) > self.max_evidence_tokens
            if over_tokens or over_evidence:
                # the evidence budget closes a row exactly as the token budget does: the cost of the
                # evidence axis is real and must not depend on which conversations packed together
                packed["closed_by_evidence"] += int(over_evidence and not over_tokens)
                push_row()
                if len(rows) == self.batch_size:
                    yield yield_batch()

            n_supervised = sum(supervised)
            per_token_weight = 1.0 / n_supervised if n_supervised else 0.0
            current["seq"].extend(tokens)
            current["labels"].extend(t if f else -100 for t, f in zip(tokens, supervised))
            current["weights"].extend(per_token_weight if f else 0.0 for f in supervised)
            current["seq"].extend([self._pad_id] * self.num_mtp_tokens)
            current["labels"].extend([-100] * self.num_mtp_tokens)
            current["weights"].extend([0.0] * self.num_mtp_tokens)
            current["sections"].append(block_len)
            current["evidence"].append({"ids": ev_tokens, "chunks": ev_chunks, "keys": ev_keys})
            committed_position = position

            if len(current["seq"]) >= usable:
                push_row()
                if len(rows) == self.batch_size:
                    yield yield_batch()

        if current["seq"]:
            push_row()
        if rows:
            yield yield_batch()

        if skipped_too_long:
            logger.warning(
                f"[worker {worker_id}] skipped {skipped_too_long} conversation(s) over "
                f"max_length={self.max_length} or max_evidence_tokens={self.max_evidence_tokens}"
            )


def _pack_evidence(rows: List[dict]) -> dict:
    """Rows' per-conversation evidence -> the five batch-aligned tensors. ``{}`` when there is none.

    ``doc_slot`` and ``chunk_slot`` say which conversation *of this row* each entry serves, which is
    all a worker can know: the global segment numbering depends on how many segments the earlier
    rows of the batch turned out to have, and that is the trainer's to resolve
    (``evidence_from_batch``). Keeping it row-local here is also what keeps every tensor's first
    dimension equal to the batch size.
    """
    widths = [sum(len(e["ids"]) for e in r["evidence"]) for r in rows]
    chunk_counts = [sum(e["keys"].shape[0] for e in r["evidence"]) for r in rows]
    S_ev, C = max(widths, default=0), max(chunk_counts, default=0)
    if S_ev == 0 or C == 0:
        # nothing retrieved anywhere in this batch: emit no evidence at all, so the trainer runs the
        # forward the model ran before the port existed rather than one over all-padding evidence
        return {}

    B = len(rows)
    ids = np.zeros((B, S_ev), dtype=np.int64)
    chunk_ids = np.full((B, S_ev), -1, dtype=np.int64)
    doc_slot = np.full((B, S_ev), -1, dtype=np.int64)
    keys = np.zeros((B, C, EMBED_DIM), dtype=np.float32)
    chunk_slot = np.full((B, C), -1, dtype=np.int64)

    for r, row in enumerate(rows):
        token_at, chunk_at = 0, 0
        for slot, ev in enumerate(row["evidence"]):
            n = len(ev["ids"])
            if n:
                ids[r, token_at:token_at + n] = ev["ids"]
                # offset by the row's running chunk count so two conversations' chunk 0 do not read
                # as one continuous passage when they land next to each other
                chunk_ids[r, token_at:token_at + n] = np.asarray(ev["chunks"]) + chunk_at
                doc_slot[r, token_at:token_at + n] = slot
                token_at += n
            k = ev["keys"].shape[0]
            if k:
                keys[r, chunk_at:chunk_at + k] = ev["keys"]
                chunk_slot[r, chunk_at:chunk_at + k] = slot
                chunk_at += k

    return {
        "evidence_ids": torch.from_numpy(ids),
        "evidence_chunk_ids": torch.from_numpy(chunk_ids),
        "evidence_doc_slot": torch.from_numpy(doc_slot),
        "chunk_keys": torch.from_numpy(keys),
        "chunk_slot": torch.from_numpy(chunk_slot),
    }


def evidence_from_batch(model, batch: dict, cu_seqlens: torch.Tensor):
    """Build the ``EvidenceBatch`` for one batch, in the trainer's thread.

    The worker emits row-local slots; the global segment numbering is resolved here because it needs
    the query side's ``cu_seqlens``, which is itself built in-thread (it is ragged, and accelerate
    truncates a ragged dim 0 to the batch size).

    The numbering falls straight out of the query segmentation: conversation *j* of row *r* is global
    segment ``seg[r, 0] + j``, because a row's conversations are its first segments in order. Row
    padding on the evidence axis is assigned to ``seg[r, -1]`` -- the row's final trailing pad
    segment, whose single query token's output nothing reads.

    Returns None when the batch carries no evidence, which takes the bit-identical forward.
    """
    from modules.model.attention import _segment_ids

    if "evidence_ids" not in batch:
        return None
    # the device comes from cu_seqlens, not from the batch: the training loop reads an accelerate
    # prepared dataloader whose batches are already on the GPU, but `evaluate` iterates the dataset
    # directly and moves only the tensors it names. Taking the device from the batch would then build
    # the evidence on the host and hand it to a CUDA model, which fails inside the embedding gather
    # with an error naming neither this function nor evidence. cu_seqlens is built next to the
    # forward in both paths, so it always carries the device the forward will run on.
    device = cu_seqlens.device
    B, S = batch["input_ids"].shape
    seg = _segment_ids(cu_seqlens, B, S, device)                      # [B, S], global
    row_first = seg[:, :1]                                            # [B, 1]
    row_last = seg[:, -1:]                                            # [B, 1]

    doc_slot = batch["evidence_doc_slot"].to(device)
    ev_segments = torch.where(doc_slot >= 0, row_first + doc_slot, row_last.expand_as(doc_slot))

    chunk_slot = batch["chunk_slot"].to(device)
    valid = chunk_slot >= 0
    chunk_segments = (row_first + chunk_slot)[valid]                  # [M], row major
    chunk_keys = batch["chunk_keys"].to(device)[valid]                # [M, 384], same order

    return model.build_evidence(
        batch["evidence_ids"].to(device),
        batch["evidence_chunk_ids"].to(device),
        ev_segments,
        int(cu_seqlens.numel() - 1),
        chunk_keys=chunk_keys,
        chunk_segments=chunk_segments,
    )
