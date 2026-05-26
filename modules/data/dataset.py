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
        padding: str = "max_length"
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
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        if mode not in config:
            raise ValueError(f"mode '{mode}' not found in {config_path}")
            
        self.sources = config[mode]

    def _batch_iterator(self) -> Iterator[dict]:
        file_iter = FileIterator(self.sources)
        current_batch = []
        
        for records, column in file_iter:
            for record in records:
                # if column specifies messages and theres a list of dicts, apply chat template (likely SFT)
                if column == "messages" and (isinstance(record, list) or isinstance(record, dict)):
                    try:
                        text = self.tokenizer.apply_chat_template(record, tokenize=False)
                    except Exception:
                        text = str(record)
                else:
                    text = str(record)
                    
                current_batch.append(text)
                
                if len(current_batch) == self.batch_size:
                    yield self._tokenize_batch(current_batch)
                    current_batch = []
                    
        # yield remaining
        if current_batch:
            yield self._tokenize_batch(current_batch)

    def _tokenize_batch(self, texts: list[str]) -> dict:
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors="pt"
        )
        
        labels = tokenized["input_ids"].clone()
        if "attention_mask" in tokenized:
            labels = labels.masked_fill(tokenized["attention_mask"] == 0, -100)
            
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get("attention_mask"),
            "labels": labels
        }

    def __iter__(self) -> Iterator[dict]:
        return self._batch_iterator()
