"""Packed SFT batches with an explicit per-token loss mask (PLAN.md Step 12).

Same shape of machinery as ``modules/data/dataset.py`` -- memmapped flat corpus, document packing
into ``max_length`` rows, batch-aligned ``document_ids`` the trainer turns into varlen
``cu_seqlens`` -- with three differences that the SFT objective forces:

  * **A third file.** ``{split}.mask`` is one uint8 per token in ``{split}.bin``, 1 where the token
    is supervised (assistant content and its terminating EOS, see ``modules/data/chat.py``) and 0
    everywhere else. Pretraining could derive labels structurally from position; SFT cannot,
    because prompt and completion interleave inside a multi-turn conversation.
  * **Documents are never split across rows.** The pretraining dataset continues an over-long
    document at the start of the next sequence, which is correct for a token stream and wrong for a
    conversation: the tail half would be supervised with its prompt missing. Anything that does not
    fit in an empty row is skipped entirely (prep already filters these out; this is the backstop),
    and anything that does not fit in the *current* row starts a new one. That wastes some trailing
    padding -- roughly half a mean document per row, ~5-10% at ``max_length=4096`` -- which is the
    price of never training on a decapitated conversation.
  * **Documents are shuffled, per epoch.** Pretraining reads in on-disk order on purpose (Step 11
    baked the mix ratio into that order). SFT runs multiple epochs over a much smaller corpus, so
    seeing the same order every epoch would make the gradient sequence identical each time. The
    order is a seeded permutation of ``[0, num_docs)``, reseeded from ``(seed, epoch)``, which every
    worker derives independently -- so it stays a pure function of the checkpointed
    ``(epoch, doc position)`` pair and resume works exactly like the pretraining one.

Resume granularity is a *position in the permuted order*, not a raw document id: workers shard it
with the same ``position % num_workers`` arithmetic, so one conservative min-across-workers scalar
(the trainer's ``global_offset``) is still enough. Reading it back under a different ``seed`` or
``epoch`` would resume against a different permutation -- both are checkpointed for that reason.

Every batch also carries **``loss_weights``**, a ``[B, S]`` float tensor holding ``1 / (supervised
tokens in this conversation)`` on each supervised position and 0 elsewhere -- NEXT.md Phase 2's
fix #3. Under plain per-token CE every supervised token counts the same, so a conversation's
influence on the gradient is proportional to how long its answer is; SQuAD v2's ~6-token refusal is
then the cheapest loss reduction in the corpus, and the real SFT run duly collapsed onto one
literal refusal string on 78.4% of *answerable* questions. Weighting by these makes the objective a
mean over conversations of the mean CE *within* a conversation, so a refusal and a multi-sentence
answer carry the same weight regardless of length. It is emitted unconditionally and cheaply (one
float per token, computed in the worker); whether the loss actually uses it is the trainer's
decision (``SFTConfig.conversation_loss_weighting``), so the same corpus serves both objectives.
"""
import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from utils import logger


class SFTDataset(IterableDataset):
    """streams packed, loss-masked SFT batches from a ``{split}.bin``/``.idx``/``.mask`` triple."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 4,
        max_length: int = 4096,
        split: str = "sft_train",
        num_mtp_tokens: int = 1,
        start_doc_idx: int = 0,
        seed: int = 42,
        epoch: int = 0,
        shuffle: bool = True,
    ) -> None:
        """
        Args:
            data_dir: directory holding ``{split}.bin`` / ``{split}.idx`` / ``{split}.mask``.
            tokenizer: only its pad/eos/bos ids are used; tokens come pre-tokenized from
                ``scripts/prepare_sft_data.py``.
            batch_size: samples per batch (batches are assembled here, so the DataLoader takes
                ``batch_size=None``).
            max_length: row length to pack conversations into.
            split: file stem, e.g. "sft_train" or "sft_val".
            num_mtp_tokens: separator slots appended after each conversation. All pads, unlike the
                pretraining dataset's ``EOS + pads`` -- an SFT document already ends with its own
                *supervised* EOS (the model has to learn to stop), so a second one would be a
                duplicate. The slots still exist so an MTP head reading past a document's last
                supervised position lands on padding rather than the next conversation.
            start_doc_idx: position in this epoch's permuted document order to resume from.
            seed: base seed for the per-epoch permutation. Must match what the checkpoint was
                written under, or the resume position points into a different order.
            epoch: which epoch's permutation to generate; set via ``set_epoch``.
            shuffle: False reads in on-disk order (used for the validation split, where a stable
                order makes successive eval numbers comparable).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_mtp_tokens = num_mtp_tokens
        self.start_doc_idx = start_doc_idx
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle

        self.bin_path = os.path.join(data_dir, f"{split}.bin")
        self.idx_path = os.path.join(data_dir, f"{split}.idx")
        self.mask_path = os.path.join(data_dir, f"{split}.mask")
        for path in (self.bin_path, self.idx_path, self.mask_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} is missing -- run `python scripts/prepare_sft_data.py` first "
                    f"(PLAN.md Step 12)"
                )

        self._bos_id = getattr(tokenizer, "bos_token_id", None)
        self._pad_id = tokenizer.pad_token_id

        # idx is tiny (8 bytes/doc); stat it rather than holding a memmap open outside the worker
        # processes that will actually read it
        self.num_docs = os.path.getsize(self.idx_path) // 8 - 1
        n_tokens = os.path.getsize(self.bin_path) // 2
        if os.path.getsize(self.mask_path) != n_tokens:
            raise ValueError(
                f"{self.mask_path} has {os.path.getsize(self.mask_path):,} entries but "
                f"{self.bin_path} has {n_tokens:,} tokens -- the pair is out of sync, rebuild it"
            )
        logger.info(
            f"SFTDataset[{split}]: {self.num_docs:,} conversations, {n_tokens:,} tokens, "
            f"shuffle={shuffle}"
        )

    def set_epoch(self, epoch: int) -> None:
        """Select which epoch's permutation the next iteration uses (call before iterating)."""
        self.epoch = epoch

    def document_order(self, num_docs: int) -> np.ndarray:
        """This epoch's document order, identical in every worker.

        A permutation of ``num_docs`` int64s is ~8MB per million conversations -- small enough to
        materialize per worker per epoch, which keeps the order a pure function of ``(seed, epoch)``
        with no cross-worker coordination and no state to checkpoint beyond those two numbers.
        """
        if not self.shuffle:
            return np.arange(num_docs, dtype=np.int64)
        return np.random.default_rng([self.seed, self.epoch]).permutation(num_docs)

    def _batch_iterator(self) -> Iterator[dict]:
        # reopened per worker/epoch rather than held on the object -- a long-lived memmap handed
        # across DataLoader worker restarts is a known leak vector (same reasoning as dataset.py)
        bin_mmap = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        idx_mmap = np.memmap(self.idx_path, dtype=np.uint64, mode="r")
        mask_mmap = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        num_docs = idx_mmap.shape[0] - 1
        order = self.document_order(num_docs)

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # positions (not raw doc ids) are what gets sharded and checkpointed -- see the module
        # docstring. same arithmetic as dataset.py so the resume story is unchanged.
        first = self.start_doc_idx + ((worker_id - self.start_doc_idx) % num_workers)
        logger.info(
            f"[worker {worker_id}] SFT epoch {self.epoch}: starting at position {first}/{num_docs} "
            f"(global resume point {self.start_doc_idx})"
        )

        batch_input_ids, batch_doc_ids, batch_labels, batch_weights = [], [], [], []
        current_seq, current_labels, current_weights, current_sections = [], [], [], []
        # position of the last conversation actually copied into a row. Every yield below happens
        # with ``current_seq`` empty, so at that moment this is exactly "the last position whose
        # tokens are inside the batch being handed out" -- which is what the trainer's
        # min-across-workers + num_workers resume arithmetic needs. Reporting the *loop* variable
        # instead would be off by one on the "doesn't fit, flush first" path, where the batch is
        # yielded before the conversation that triggered the flush has been copied anywhere, and
        # the resume would skip that conversation outright.
        committed_position = first - num_workers
        skipped_too_long = 0

        def push_sequence():
            pad_len = self.max_length - len(current_seq)
            padded = current_seq + [self._pad_id] * pad_len
            labels = current_labels + [-100] * pad_len
            weights = current_weights + [0.0] * pad_len

            # one causal segment per conversation block; trailing row padding becomes length-1
            # (self-attention only) segments, exactly as in the pretraining dataset
            doc_ids, seg, start = [], 0, 0
            for block_len in current_sections:
                block_end = min(start + block_len, self.max_length)
                if block_end > start:
                    doc_ids.extend([seg] * (block_end - start))
                    seg += 1
                start = block_end
            for _ in range(start, self.max_length):
                doc_ids.append(seg)
                seg += 1

            batch_input_ids.append(padded)
            batch_doc_ids.append(doc_ids)
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_weights.append(torch.tensor(weights, dtype=torch.float32))

            current_seq.clear()
            current_labels.clear()
            current_weights.clear()
            current_sections.clear()

        def yield_batch():
            nonlocal batch_input_ids, batch_doc_ids, batch_labels, batch_weights
            batch = {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "document_ids": torch.tensor(batch_doc_ids, dtype=torch.long),
                "labels": torch.stack(batch_labels),
                # per-conversation loss weights; see the module docstring. Aligned with input_ids,
                # so a consumer shifts them exactly like it shifts labels.
                "loss_weights": torch.stack(batch_weights),
                # [B]-shaped so accelerate's batch splitting treats them like input_ids; the
                # trainer takes the min across workers at checkpoint time (PLAN.md Step 9)
                "doc_idx": torch.full((len(batch_input_ids),), committed_position, dtype=torch.long),
                "worker_id": torch.full((len(batch_input_ids),), worker_id, dtype=torch.long),
            }
            batch_input_ids, batch_doc_ids, batch_labels, batch_weights = [], [], [], []
            return batch

        for position in range(first, num_docs, num_workers):
            doc = int(order[position])
            start, end = int(idx_mmap[doc]), int(idx_mmap[doc + 1])
            if end <= start:
                continue
            tokens = bin_mmap[start:end].tolist()
            supervised = mask_mmap[start:end].tolist()

            # idempotent BOS, matching dataset.py: prep writes one, but a corpus built by other
            # means (or a future template change) must not silently lose it
            if self._bos_id is not None and tokens[0] != self._bos_id:
                tokens = [self._bos_id] + tokens
                supervised = [0] + supervised

            block_len = len(tokens) + self.num_mtp_tokens
            if block_len > self.max_length:
                # prep's --max-doc-tokens filter should have removed these; count and drop rather
                # than truncate, because a truncated conversation loses its supervised EOS and
                # would teach the model not to stop
                skipped_too_long += 1
                continue

            if len(current_seq) + block_len > self.max_length:
                push_sequence()
                if len(batch_input_ids) == self.batch_size:
                    yield yield_batch()

            # 1 / supervised tokens, so every conversation contributes the same total weight
            # regardless of answer length. A conversation with nothing supervised cannot occur (the
            # chat template rejects those at prep time), but a 0 here would be a NaN there.
            n_supervised = sum(supervised)
            per_token_weight = 1.0 / n_supervised if n_supervised else 0.0

            current_seq.extend(tokens)
            current_labels.extend(
                token if flag else -100 for token, flag in zip(tokens, supervised)
            )
            current_weights.extend(per_token_weight if flag else 0.0 for flag in supervised)
            # separator slots: pad ids, never supervised (the conversation's own EOS already is)
            current_seq.extend([self._pad_id] * self.num_mtp_tokens)
            current_labels.extend([-100] * self.num_mtp_tokens)
            current_weights.extend([0.0] * self.num_mtp_tokens)
            current_sections.append(block_len)
            committed_position = position

            if len(current_seq) == self.max_length:
                push_sequence()
                if len(batch_input_ids) == self.batch_size:
                    yield yield_batch()

        if len(current_seq) > 0:
            push_sequence()
        if len(batch_input_ids) > 0:
            yield yield_batch()

        if skipped_too_long:
            logger.warning(
                f"[worker {worker_id}] skipped {skipped_too_long} conversation(s) longer than "
                f"max_length={self.max_length}; re-run prep with a matching --max-doc-tokens"
            )

    def __iter__(self) -> Iterator[dict]:
        return self._batch_iterator()
