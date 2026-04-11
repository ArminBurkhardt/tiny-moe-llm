"""Streaming dataset that yields tokenized LLM batches of similar texts.

This wraps ``FileLoader`` (parquet reader) and ``VectorizedDataset`` (embedding +
 similarity search) so each returned batch contains texts that are mutually
 similar above a cosine threshold. Each yielded item is already tokenized for a
 causal LLM: ``input_ids``, ``attention_mask``, and ``labels`` where padding is
 masked to ``-100``. Use with a standard torch DataLoader (no extra collator
 needed unless you want dynamic padding)."""

from typing import Iterator, Optional
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from modules.data.dataloader import FileLoader
from modules.data.vectorized_dataset import VectorizedDataset, _EmbeddingGemmaModel
from utils import BASE_DIR, logger, DIR


class Dataset(IterableDataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        *,
        sources: Optional[list[tuple[str, str]]] = None,
        texts: Optional[list[str]] = None,
        batch_size: int = 64,
        similarity_delta: float = 0.7,
        text_column: Optional[str] = None,
        max_loaded_embeddings: int = 1_000_000,
        device: Optional[str] = None,
        embedding_dim: int = 256,
        embedding_model: Optional[_EmbeddingGemmaModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 512,
        padding: str = "max_length",
        return_embeddings: bool = False,
        return_texts: bool = False,
    ) -> None:
        """
        Args:
            data_root: Root folder containing parquet shards (walked by FileLoader). Legacy, prefer sources.
            sources: List of (root_dir, text_column) tuples.
            texts: Optional in-memory texts. If provided, FileLoader is skipped.
            batch_size: Number of samples to return per batch.
            similarity_delta: Cosine similarity threshold passed to VectorizedDataset.get_similar_batch.
            text_column: Column name to read from parquet files. Legacy, prefer sources.
            max_loaded_embeddings: Upper bound of embeddings kept in memory for similarity search.
            device: Optional device for the embedding model (e.g. "cuda").
            embedding_dim: Embedding dimension for _EmbeddingGemmaModel.
            embedding_model: Optional prebuilt embedding model (useful for testing/mocking).
            tokenizer: Hugging Face tokenizer used to build input_ids / masks.
            max_length: Max tokens; texts are truncated to this length.
            padding: Padding strategy passed to tokenizer (e.g. "max_length" or "longest").
            return_embeddings: If True, include stacked embeddings alongside tokenized batch.
            return_texts: If True, include raw texts for debugging/logging.
        """

        if texts is None and data_root is None and sources is None:
            raise ValueError("Either texts, data_root, or sources must be provided.")

        if tokenizer is None:
            raise ValueError("tokenizer must be provided to build LLM batches.")

        self.batch_size = batch_size
        self.similarity_delta = similarity_delta
        self.max_loaded_embeddings = max_loaded_embeddings
        self.return_embeddings = return_embeddings
        self.return_texts = return_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.show_progress_bar = False

        if sources is not None:
            self.sources = sources
        elif data_root is not None and text_column is not None:
            self.sources = [(data_root, text_column)]
        elif data_root is not None:
            self.sources = [(data_root, "content")]
        else:
            self.sources = []

        self.model = embedding_model or _EmbeddingGemmaModel(device=device, embedding_dim=embedding_dim)
        # Light convenience: if device not given but cuda is available, move model if supported.
        if (torch.cuda.is_available()) and (device is None):
            self._maybe_to_device("cuda")

        # State for the active vectorized dataset built from the current parquet shard or provided texts.
        self.current_vector_ds: Optional[VectorizedDataset] = None
        self.current_batches_left: int = 0
        
        self.texts_mode = texts is not None
        self.source_iter = self._shard_iterator() if not self.texts_mode else None

        # If texts are given eagerly build the vectorized dataset so __iter__ works immediately.
        if texts is not None:
            self._load_vectorized_dataset(texts)

    def _shard_iterator(self) -> Iterator[list[str]]:
        import os
        for root_dir, text_column in self.sources:
            if not os.path.exists(root_dir):
                logger.warning(f"Data root not found, skipping: {root_dir}")
                continue
            
            file_loader = FileLoader(root_dir)
            for df in file_loader:
                if text_column not in df.columns:
                    continue
                
                texts = df[text_column].dropna().astype(str).tolist()
                if len(texts) > 0:
                    yield texts

    def __iter__(self) -> Iterator[dict]:
        return self._batch_iterator()

    def _batch_iterator(self) -> Iterator[dict]:
        while True:
            # Ensure we have a vectorized dataset with remaining batches.
            if self.current_vector_ds is None or self.current_batches_left == 0:
                if not self._advance_to_next_file():
                    break  # Exhausted all data

            batch = self.current_vector_ds.get_similar_batch(
                batch_size=self.batch_size,
                delta=self.similarity_delta,
                text_only=not self.return_embeddings,
            )

            # A small guard for very small shards: skip empty batches.
            if len(batch) == 0:
                self.current_batches_left = 0
                continue

            self.current_batches_left = max(self.current_batches_left - 1, 0)
            texts, embeddings = self._extract_texts_and_embeddings(batch)

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=self.padding,
                return_tensors="pt",
            )

            labels = tokenized["input_ids"].clone()
            if "attention_mask" in tokenized:
                labels = labels.masked_fill(tokenized["attention_mask"] == 0, -100)

            result = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized.get("attention_mask"),
                "labels": labels,
            }

            if self.return_embeddings and embeddings is not None:
                result["embeddings"] = embeddings
            if self.return_texts:
                result["texts"] = texts

            yield result

    def _advance_to_next_file(self) -> bool:
        """Load the next parquet shard and build its VectorizedDataset."""
        # If we were instantiated with in-memory texts, do not advance further.
        if self.texts_mode:
            return False

        try:
            texts = next(self.source_iter)
            self._load_vectorized_dataset(texts)
            return True
        except StopIteration:
            return False

    def _load_vectorized_dataset(self, texts: list[str]) -> None:
        # Respect max_loaded_embeddings by truncating the shard if necessary.
        if len(texts) > self.max_loaded_embeddings:
            texts = texts[: self.max_loaded_embeddings]

        vector_ds = VectorizedDataset(texts, self.model, max_loaded_embeddings=self.max_loaded_embeddings)
        vector_ds.compute_embeddings(show_progress_bar=self.show_progress_bar)

        self.current_vector_ds = vector_ds
        # Estimate how many batches we can serve from this shard; at least one.
        self.current_batches_left = max(len(texts) // self.batch_size, 1)

    def _extract_texts_and_embeddings(self, batch):
        """Normalize batch to (texts, embeddings_or_None)."""
        # If text_only was True, batch is List[str]; otherwise List[dict].
        if isinstance(batch[0], str):
            return batch, None

        texts = [item["text"] for item in batch]
        embeddings = torch.stack([item["embedding"] for item in batch]) if self.return_embeddings else None
        return texts, embeddings

    def _maybe_to_device(self, device: str) -> None:
        # Avoid attribute errors when using a mocked embedding model.
        if hasattr(self.model, "to_device"):
            self.model.to_device(device)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model = self.model.model.to(device)





def test_dataset():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DIR.GEMMA_3_DIR)

    dataset = Dataset(
        data_root=DIR.UFW_V1_4_DIR,
        batch_size=8,
        similarity_delta=0.8,
        text_column="content",
        max_loaded_embeddings=10000,
        device="cuda",
        embedding_dim=256,
        tokenizer=tokenizer,
        max_length=128,
        padding="max_length",
        return_embeddings=True,
        return_texts=True,
    )
    
    dataset.show_progress_bar = True

    for i, batch in enumerate(dataset):
        logger.info("Batch %s:", i)
        logger.info("Input IDs shape: %s", batch["input_ids"].shape)
        logger.info("Attention Mask shape: %s", batch["attention_mask"].shape)
        logger.info("Labels shape: %s", batch["labels"].shape)
        if "embeddings" in batch:
            logger.info("Embeddings shape: %s", batch["embeddings"].shape)
        if "texts" in batch:
            logger.info("Texts: %s", batch["texts"])

        if i >= 2:
            break





