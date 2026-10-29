import os
import torch
import numpy as np
from utils import logger
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
from typing import Iterator


class Dataset(IterableDataset):
    """streams packed training batches from a pre-tokenized flat-file corpus.

    reads two files per split: ``{split}.bin`` (a flat uint16 token stream, produced by
    scripts/prepare_data.py, PLAN.md Step 11) and ``{split}.idx`` (uint64 document start
    offsets into that stream, one entry per document plus a trailing entry == len(bin)).
    documents are read in their on-disk order, once, with no shuffling: Step 11 already
    interleaves sources at the target mix ratios while writing, so a straight sequential
    read reproduces that mix -- reshuffling here would just undo it.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 64,
        max_length: int = 1024,
        split: str = "phase1",
        num_mtp_tokens: int = 1,
        start_doc_idx: int = 0,
    ) -> None:
        """
        Args:
            data_dir: directory containing ``{split}.bin`` / ``{split}.idx``.
            tokenizer: only its bos/eos/pad ids are used (tokens already come pre-tokenized).
            batch_size: number of samples per batch.
            max_length: sequence length to pack documents into.
            split: file stem to read, e.g. "phase1" or "phase2".
            num_mtp_tokens: number of separator tokens appended after each document. The first
                is EOS (supervised, so the model learns to terminate documents), the rest are
                pads. Must stay >= the model's number of MTP heads to keep MTP from being
                supervised across document boundaries.
            start_doc_idx: global index into the (unshuffled) document stream to resume from.
                Sharding across DataLoader workers is pure ``doc_idx % num_workers`` arithmetic,
                so this single scalar is enough to resume deterministically -- no per-worker
                bookkeeping needed (PLAN.md Step 9).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_mtp_tokens = num_mtp_tokens
        self.start_doc_idx = start_doc_idx

        self.bin_path = os.path.join(data_dir, f"{split}.bin")
        self.idx_path = os.path.join(data_dir, f"{split}.idx")

        # document framing: prepend BOS if it isn't already the first token (idempotent whether
        # or not Step 11 baked one in) and terminate each document with a supervised EOS
        self._bos_id = getattr(tokenizer, "bos_token_id", None)
        self._eos_id = getattr(tokenizer, "eos_token_id", None)
        self._sep_id = self._eos_id if self._eos_id is not None else tokenizer.pad_token_id
        self._supervise_eos = self._eos_id is not None

        # idx is tiny (8 bytes/doc) -- just stat it for the doc count instead of holding a memmap
        # open outside the worker processes that will actually read it
        idx_bytes = os.path.getsize(self.idx_path)
        self.num_docs = idx_bytes // 8 - 1
        logger.info(
            f"Dataset[{split}]: {self.num_docs:,} documents, "
            f"{os.path.getsize(self.bin_path) / 1e9:.2f}GB token stream"
        )

    def _batch_iterator(self) -> Iterator[dict]:
        # reopen the memmaps fresh per worker/epoch (not held across the object's lifetime) --
        # a long-lived memmap handed across DataLoader worker restarts is a known leak vector
        bin_mmap = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        idx_mmap = np.memmap(self.idx_path, dtype=np.uint64, mode="r")
        num_docs = idx_mmap.shape[0] - 1

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # deterministic w::num_workers sharding over the whole (unshuffled) doc stream, offset to
        # the first index >= start_doc_idx this worker owns
        first = self.start_doc_idx + ((worker_id - self.start_doc_idx) % num_workers)
        logger.info(
            f"[worker {worker_id}] starting at doc {first}/{num_docs} "
            f"(global resume point {self.start_doc_idx})"
        )

        current_batch_input_ids = []
        current_batch_doc_ids = []      # per sample [max_length] segment id list (batch aligned)
        current_batch_labels = []

        current_seq = []
        current_seq_sections = []       # list of tuples (text_len, num_pad)
        current_doc_idx = first

        def push_sequence():
            # pad to max_length
            pad_len = self.max_length - len(current_seq)
            padded_seq = current_seq + [self.tokenizer.pad_token_id] * pad_len

            # per token segment id covering all max_length positions
            # => each document block is one causal segment
            # any trailing padding positions become length-1 (self attention only) segments
            # the model turns these into flash-attn cu_seqlens
            # emitted as a batch aligned [max_length] id list for accelerates batch handling
            doc_ids = []
            seg = 0
            start = 0
            for text_len, num_pad in current_seq_sections:
                block_len = text_len + num_pad
                block_end = min(start + block_len, self.max_length)
                actual_block_len = block_end - start
                if actual_block_len > 0:
                    doc_ids.extend([seg] * actual_block_len)
                    seg += 1
                start = block_end
            for _ in range(start, self.max_length):     # trailing pad => own length-1 segment
                doc_ids.append(seg)
                seg += 1

            label_seq = torch.tensor(padded_seq)
            label_mask = torch.zeros(self.max_length, dtype=torch.bool)
            start = 0
            for text_len, num_pad in current_seq_sections:
                l_end = min(start + text_len, self.max_length)
                if l_end > start + 1:                   # mask to predict all tokens except the first one of each block
                    label_mask[start+1:l_end] = True
                # supervise the EOS separator right after the document so the model learns to terminate documents (the remaining separator pads stay unsupervised)
                if num_pad > 0 and self._supervise_eos and l_end < self.max_length:
                    label_mask[l_end] = True
                start += text_len + num_pad

            label_seq[~label_mask] = -100

            current_batch_input_ids.append(padded_seq)
            current_batch_doc_ids.append(doc_ids)
            current_batch_labels.append(label_seq)

            current_seq.clear()
            current_seq_sections.clear()

        def yield_batch():
            nonlocal current_batch_input_ids, current_batch_doc_ids, current_batch_labels
            batch = {
                "input_ids": torch.tensor(current_batch_input_ids, dtype=torch.long),
                # batch aligned [B, max_length] segment ids -> model builds cu_seqlens from these.
                # NOTE: the trainer must derive cu_seqlens from these in thread; a ragged cu_seqlens
                # cannot be carried in the batch because accelerates split_batches truncates its
                # dim-0 to the batch size (see modules/model/attention.py).
                "document_ids": torch.tensor(current_batch_doc_ids, dtype=torch.long),
                "labels": torch.stack(current_batch_labels),
                # global doc index this worker had reached when this batch was assembled. the
                # trainer takes the min across workers at checkpoint time to get a single, safe
                # resume point (PLAN.md Step 9) -- shape [B] so accelerates batch splitting
                # handles it like any other tensor.
                "doc_idx": torch.full((len(current_batch_input_ids),), current_doc_idx, dtype=torch.long),
                "worker_id": torch.full((len(current_batch_input_ids),), worker_id, dtype=torch.long),
            }
            current_batch_input_ids = []
            current_batch_doc_ids = []
            current_batch_labels = []
            return batch

        for current_doc_idx in range(first, num_docs, num_workers):
            start, end = int(idx_mmap[current_doc_idx]), int(idx_mmap[current_doc_idx + 1])
            if end <= start:
                continue
            tokens = bin_mmap[start:end].tolist()

            if self._bos_id is not None and tokens[0] != self._bos_id:
                tokens = [self._bos_id] + tokens

            # pack the document, splitting it across sequences when it does not fit:
            # the remainder continues at the start of the next sequence (its own attention segment)
            offset = 0
            while offset < len(tokens):
                space = self.max_length - len(current_seq)
                take = min(len(tokens) - offset, space)
                current_seq.extend(tokens[offset:offset + take])
                offset += take

                pad_to_add = 0
                if offset == len(tokens):  # document finished => EOS + MTP separator pads
                    pad_to_add = min(self.num_mtp_tokens, self.max_length - len(current_seq))
                    if pad_to_add > 0:
                        current_seq.extend([self._sep_id] + [self.tokenizer.pad_token_id] * (pad_to_add - 1))
                current_seq_sections.append((take, pad_to_add))

                if len(current_seq) >= self.max_length:
                    push_sequence()
                    if len(current_batch_input_ids) == self.batch_size:
                        yield yield_batch()

        if len(current_seq) > 0:
            push_sequence()
        if len(current_batch_input_ids) > 0:
            yield yield_batch()

    def __iter__(self) -> Iterator[dict]:
        return self._batch_iterator()
