import os
import json
import random
import torch
import pandas as pd
from utils import logger
from pathlib import Path
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
from typing import Iterator


class FileIterator:
    """iterates an ordered list of (path, column) data files.

    the order is decided by Dataset and handed in, so every worker shards over the same global order (indexed by global_idx) 
    """
    def __init__(self, files: list[tuple[str, str]], start_file_idx: int = 0):
        self.files = files
        self.start_file_idx = start_file_idx

    def __iter__(self):
        indexed = list(enumerate(self.files))
        # shard files across DataLoader workers to avoid every worker yielding
        # the same batches (IterableDataset is replicated per worker otherwise)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if worker_info is not None and worker_info.num_workers > 1:
            indexed = indexed[worker_info.id::worker_info.num_workers]

        start = self.start_file_idx
        logger.info(f"[worker {worker_id}] FileIterator starting: {len(self.files)} total files, resuming from file_idx={start}")

        for global_idx, (file_path, column) in indexed:
            # fast forward on resume: skip already consumed files cheaply
            if global_idx < self.start_file_idx:
                continue
            logger.info(f"[worker {worker_id}] Reading file {global_idx}/{len(self.files)-1}: {file_path}")
            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path, columns=[column])
                    records = df[column].dropna().tolist()
                elif file_path.endswith('.jsonl') or file_path.endswith('.json'):
                    df = pd.read_json(file_path, lines=True)
                    if column in df.columns:
                        records = df[column].dropna().tolist()
                    else:
                        logger.warning(f"column {column} not found in {file_path}")
                        continue
                else:
                    continue
                
                if records:
                    yield global_idx, records, column

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

class Dataset(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 64,
        max_length: int = 1024,
        mode: str = "pretrain",
        config_path: str = "data_config.json",
        padding: str = "max_length",
        num_mtp_tokens: int = 1,
        start_file_idx: int = 0,
        seed: int = 42,
        max_tokens_per_shard: int | None = None,
        start_shard_token_count: int = 0,
        start_record_idx: int = 0,
    ) -> None:
        """
        Dataset for LLM training. Configured via config (`config_path`)

        Args:
            tokenizer: tokenizer to use for tokenization
            batch_size: number of samples per batch
            max_length: maximum sequence length for tokenization
            mode: which mode to use from config (eg. "pretrain", "sft")
            config_path: path to json config file specifying data sources
            padding: padding strategy for tokenization (default: "max_length")
            num_mtp_tokens: number of separator tokens appended after each document. The first one
                is EOS (supervised, so the model learns to terminate documents), the rest are pads.
                Must stay >= the models number of MTP heads to keep MTP from being supervised across document boundaries.
            start_file_idx: position in the current file_order to start reading at. 
                On resume this skips the already consumed files without tokenizing them. Mutate this attribute between epochs (set back to 0 for fresh epochs).
            seed: base RNG seed for the cross-source file shuffle
            max_tokens_per_shard: if set, move on to the next shard once this many tokens have been taken from the current one
            start_shard_token_count: tokens already taken from the shard at start_file_idx, so the shard budget continues instead of restarting on resume
            start_record_idx: records already consumed from the shard at start_file_idx, so resume continues mid shard. 
                A value of 0 (also what legacy checkpoints decode to) re-reads that shard from the top.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding
        self.num_mtp_tokens = num_mtp_tokens
        self.start_file_idx = start_file_idx
        self.seed = seed
        self.max_tokens_per_shard = max_tokens_per_shard
        self.start_shard_token_count = start_shard_token_count
        self.start_record_idx = start_record_idx
        self.worker_start = None

        # document framing: prepend BOS ourselves if the tokenizer doesn't (e.g. the DeepSeek
        # tokenizer adds no BOS even with add_special_tokens=True), and terminate each document
        # with a supervised EOS so the model learns to stop generating
        self._bos_id = getattr(tokenizer, "bos_token_id", None)
        self._eos_id = getattr(tokenizer, "eos_token_id", None)
        self._sep_id = self._eos_id if self._eos_id is not None else tokenizer.pad_token_id
        self._supervise_eos = self._eos_id is not None

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if mode not in config:
            raise ValueError(f"mode '{mode}' not found in {config_path}")

        self.sources = config[mode]

        # deterministic / sorted file list
        self._all_files = self._discover_files()
        self.file_order = self._shuffled(self._all_files, self.seed)
        logger.info(f"Discovered {len(self._all_files)} data files across {len(self.sources)} sources")

    def _discover_files(self) -> list[tuple[str, str]]:
        """canonical (sorted, deterministic) list of (path, column) data files"""
        files = []
        for src in self.sources:
            root = src.get("root")
            column = src.get("column", "text")
            # optional filename glob so stray non data files in a root are not ingested as training data
            pattern = src.get("glob", "*")
            if root is None or not os.path.exists(root):
                logger.warning(f"root path {root} does not exist. Skipping")
                continue
            # sort so the canonical order is deterministic across runs
            for path in sorted(Path(root).rglob(pattern)):
                if path.is_file() and path.suffix in [".parquet", ".jsonl", ".json"]:
                    files.append((str(path), column))
        return files

    @staticmethod
    def _shuffled(files: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
        order = list(files)
        random.Random(seed).shuffle(order)
        return order

    def reshuffle(self, seed: int) -> None:
        """reshuffle the file list across sources. call at the start of each fresh epoch with a seed and start over at the top of the new order"""
        self.file_order = self._shuffled(self._all_files, seed)
        self.start_file_idx = 0
        self.start_record_idx = 0
        self.start_shard_token_count = 0
        self.worker_start = None

    def _resume_for_worker(self, worker_id: int) -> tuple[int, int, int]:
        """(start_file_idx, start_record_idx, start_shard_token_count) for this worker.

        With worker_start set each worker resumes at its own saved position
        Otherwise everyone gets the global values and only the worker owning start_file_idx applies the within shard offsets (the rest just skip past it). 
        A worker with no saved position starts at the top of its first shard
        """
        if self.worker_start is not None:
            return self.worker_start.get(worker_id, (0, 0, 0))
        return self.start_file_idx, self.start_record_idx, self.start_shard_token_count

    def set_order(self, order: list) -> None:
        """restore a file ordering saved in a checkpoint"""
        saved = [tuple(e) for e in order]
        saved_paths = {p for p, _ in saved}
        new_files = [f for f in self._all_files if f[0] not in saved_paths]
        if new_files:
            random.Random(self.seed).shuffle(new_files)
        self.file_order = saved + new_files

    def build_legacy_order(self, consumed_count: int, seed: int) -> None:
        """resume from an old checkpoint that only stored a file_idx into the sorted order.

        the first consumed_count sorted files keep their slots (so start_file_idx skips exactly them); only the remaining unused shards get shuffled across sources
        """
        consumed = self._all_files[:consumed_count]
        remaining = self._shuffled(self._all_files[consumed_count:], seed)
        self.file_order = consumed + remaining

    def _batch_iterator(self) -> Iterator[dict]:
        # use this workers own resume point or the shared global one
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        wstart_file, wstart_record, wstart_shard = self._resume_for_worker(worker_id)

        file_iter = FileIterator(self.file_order, start_file_idx=wstart_file)

        current_batch_input_ids = []
        current_batch_doc_ids = []  # per sample [max_length] segment id list (batch aligned)
        current_batch_labels = []
        # position within the data, all three checkpointed together for resume:
        # file being consumed, record (row) within it, tokens taken from it so far
        current_file_idx = wstart_file
        current_record_idx = wstart_record
        shard_token_count = wstart_shard

        current_seq = []
        current_seq_sections = [] # list of tuples (text_len, num_pad)

        def push_sequence():
            # pad to max_length
            pad_len = self.max_length - len(current_seq)
            padded_seq = current_seq + [self.tokenizer.pad_token_id] * pad_len

            # per-token segment id covering all max_length positions: each document block is one causal segment
            # any trailing padding positions become length-1 (self attention only) segments
            # the model turns these into flash-attn cu_seqlens (equivalent to the old block mask)
            # emitted as a batch aligned [max_length] id list so accelerates batch
            # handling treats it like input_ids instead of truncating a ragged cu_seqlens
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
            for _ in range(start, self.max_length):  # trailing pad -> own length-1 segment
                doc_ids.append(seg)
                seg += 1

            label_seq = torch.tensor(padded_seq)
            label_mask = torch.zeros(self.max_length, dtype=torch.bool)
            start = 0
            for text_len, num_pad in current_seq_sections:
                l_end = min(start + text_len, self.max_length)
                if l_end > start + 1: # mask to predict all tokens except the first one of each block
                    label_mask[start+1:l_end] = True
                # supervise the EOS separator right after the document so the model learns to
                # terminate documents (the remaining separator pads stay unsupervised)
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
                # global index of the data file being read when this batch was assembled.
                # carried per sample (shape [B]) so accelerates batch splitting handles it like any other tensor
                # the trainer reads it to checkpoint the resume position.
                "file_idx": torch.full((len(current_batch_input_ids),), current_file_idx, dtype=torch.long),
                # record within that file, and tokens taken from it, when this batch was assembled
                "record_idx": torch.full((len(current_batch_input_ids),), current_record_idx, dtype=torch.long),
                "shard_token_count": torch.full((len(current_batch_input_ids),), shard_token_count, dtype=torch.long),
                # trainer keys the per worker resume positions by this
                "worker_id": torch.full((len(current_batch_input_ids),), worker_id, dtype=torch.long),
            }
            current_batch_input_ids = []
            current_batch_doc_ids = []
            current_batch_labels = []
            return batch

        for current_file_idx, records, column in file_iter:
            # the resume offsets apply to one shard only: the one this worker was mid reading at checkpoint time.
            # every later shard starts fresh with a full token budget
            if current_file_idx == wstart_file and wstart_record > 0:
                skip_records = wstart_record
                shard_token_count = wstart_shard
                logger.info(f"[worker {worker_id}] Resuming shard file_idx={current_file_idx} at record {skip_records} (skipping already-consumed records)")
            else:
                skip_records = 0
                shard_token_count = 0
            for current_record_idx, record in enumerate(records):
                # drop records consumed before the checkpoint without re-tokenizing them
                if current_record_idx < skip_records:
                    continue
                if column == "messages" and (isinstance(record, list) or isinstance(record, dict)):
                    try:
                        text = self.tokenizer.apply_chat_template(record, tokenize=False)
                    except Exception:
                        text = str(record)
                else:
                    text = str(record)

                tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
                if not tokens:
                    continue
                if self._bos_id is not None and tokens[0] != self._bos_id:
                    tokens = [self._bos_id] + tokens

                shard_token_count += len(tokens)
                if self.max_tokens_per_shard and shard_token_count > self.max_tokens_per_shard:
                    logger.info(f"Shard token limit ({self.max_tokens_per_shard:,}) reached at file_idx={current_file_idx}, moving to next shard")
                    break

                # pack the document, splitting it across sequences when it does not fit:
                # the remainder continues at the start of the next sequence (its own attention segment)
                offset = 0
                while offset < len(tokens):
                    space = self.max_length - len(current_seq)
                    take = min(len(tokens) - offset, space)
                    current_seq.extend(tokens[offset:offset + take])
                    offset += take

                    pad_to_add = 0
                    if offset == len(tokens):  # document finished -> EOS + MTP separator pads
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
