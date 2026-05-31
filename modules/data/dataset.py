import os
import json
import logging
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
from typing import Iterator

logger = logging.getLogger(__name__)

class FileIterator:
    """iterates through data files (parquet/jsonl/json) in a given list of roots"""
    def __init__(self, sources: list[dict]):
        self.sources = sources
        self.files = []
        for src in self.sources:
            root = src.get("root")
            column = src.get("column", "text")
            if root is None or not os.path.exists(root):
                logger.warning(f"root path {root} does not exist. Skipping")
                continue
            
            for path in Path(root).rglob("*"):
                if path.is_file() and path.suffix in [".parquet", ".jsonl", ".json"]:
                    self.files.append((str(path), column))
                    
    def __iter__(self):
        for file_path, column in self.files:
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
                    yield records, column
                    
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
        start_step: int = 0,
        num_mtp_tokens: int = 1
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
            start_step: global step to resume from
            num_mtp_tokens: number of padding tokens appended after each text for multi-token prediction
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding
        self.start_step = start_step
        self.num_mtp_tokens = num_mtp_tokens
        self._current_step = 0
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        if mode not in config:
            raise ValueError(f"mode '{mode}' not found in {config_path}")
            
        self.sources = config[mode]
        
        self._skip_tokenization = False
        self._skip_batches = 0

    def _batch_iterator(self) -> Iterator[dict]:
        file_iter = FileIterator(self.sources)
        
        current_batch_input_ids = []
        current_batch_attention_mask = []
        current_batch_labels = []
        
        current_seq = []
        current_seq_sections = [] # list of tuples (text_len, num_pad)
        
        def push_sequence():
            # pad to max_length
            pad_len = self.max_length - len(current_seq)
            padded_seq = current_seq + [self.tokenizer.pad_token_id] * pad_len
            
            mask = torch.zeros((1, self.max_length, self.max_length), dtype=torch.bool)
            start = 0
            for text_len, num_pad in current_seq_sections:
                block_len = text_len + num_pad
                block_end = min(start + block_len, self.max_length)
                actual_block_len = block_end - start
                
                if actual_block_len > 0:
                    mask[0, start:block_end, start:block_end] = torch.tril(torch.ones((actual_block_len, actual_block_len), dtype=torch.bool))
                
                start = block_end
                
            for i in range(start, self.max_length):
                mask[0, i, i] = True
                
            label_seq = torch.tensor(padded_seq)
            label_mask = torch.zeros(self.max_length, dtype=torch.bool)
            start = 0
            for text_len, num_pad in current_seq_sections:
                l_end = min(start + text_len, self.max_length)
                if l_end > start:
                    label_mask[start:l_end] = True
                start += text_len + num_pad
            
            label_seq[~label_mask] = -100
            
            current_batch_input_ids.append(padded_seq)
            current_batch_attention_mask.append(mask)
            current_batch_labels.append(label_seq)
            
            current_seq.clear()
            current_seq_sections.clear()
            
        def yield_batch():
            nonlocal current_batch_input_ids, current_batch_attention_mask, current_batch_labels
            batch = {
                "input_ids": torch.tensor(current_batch_input_ids, dtype=torch.long),
                "attention_mask": torch.cat(current_batch_attention_mask, dim=0),
                "labels": torch.stack(current_batch_labels)
            }
            current_batch_input_ids = []
            current_batch_attention_mask = []
            current_batch_labels = []
            return batch
        
        for records, column in file_iter:
            for record in records:
                if self._current_step < self.start_step:
                    current_seq.append(None)
                    if len(current_seq) >= self.batch_size:
                        yield {
                            "input_ids": torch.zeros((self.batch_size, self.max_length), dtype=torch.long),
                            "attention_mask": None,
                            "labels": None
                        }
                        current_seq.clear()
                        self._current_step += 1
                    continue

                if column == "messages" and (isinstance(record, list) or isinstance(record, dict)):
                    try:
                        text = self.tokenizer.apply_chat_template(record, tokenize=False)
                    except Exception:
                        text = str(record)
                else:
                    text = str(record)
                    
                tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                    
                if len(current_seq) + len(tokens) > self.max_length:
                    if len(current_seq) > 0:
                        push_sequence()
                        if len(current_batch_input_ids) == self.batch_size:
                            yield yield_batch()
                            self._current_step += 1
                            
                if len(tokens) == self.max_length:
                    current_seq.extend(tokens)
                    current_seq_sections.append((len(tokens), 0))
                    push_sequence()
                    if len(current_batch_input_ids) == self.batch_size:
                        yield yield_batch()
                        self._current_step += 1
                    continue
                    
                current_seq.extend(tokens)
                
                pad_to_add = min(self.num_mtp_tokens, self.max_length - len(current_seq))
                if pad_to_add > 0:
                    current_seq.extend([self.tokenizer.pad_token_id] * pad_to_add)
                    
                current_seq_sections.append((len(tokens), pad_to_add))
                
                if len(current_seq) == self.max_length:
                    push_sequence()
                    if len(current_batch_input_ids) == self.batch_size:
                        yield yield_batch()
                        self._current_step += 1
                        
        if len(current_seq) > 0:
            push_sequence()
        if len(current_batch_input_ids) > 0:
            yield yield_batch()
            self._current_step += 1

    def __iter__(self) -> Iterator[dict]:
        return self._batch_iterator()
