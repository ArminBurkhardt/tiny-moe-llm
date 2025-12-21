import torch
from sentence_transformers import SentenceTransformer
from utils import BASE_DIR
from typing import Literal
from torch.utils.data import Dataset

# uses EmbeddingGemma-300m
# https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers?hl=de
# https://huggingface.co/google/embeddinggemma-300m



class _EmbeddingGemmaModel:
    def __init__(self, device: str = None, embedding_dim: Literal[128, 256, 512, 768] = 256):
        self.model = SentenceTransformer(BASE_DIR + "/ckpts/pretrained/embeddinggemma-300m")
        self.embedding_dim = embedding_dim
        if device:
            self.model = self.model.to(device)
        
    def encode(self, texts: list[str], 
               convert_to_tensor: bool = True, 
               batch_size: int = 64,
               show_progress_bar: bool = False, 
               precision: Literal["float32", "int8"] = "float32",
               normalize_embeddings: bool = True) -> torch.Tensor:
        return self.model.encode(texts, 
                                 convert_to_tensor=convert_to_tensor, 
                                 show_progress_bar=show_progress_bar,
                                 batch_size=batch_size,
                                 truncate_dim=self.embedding_dim,
                                 normalize_embeddings=normalize_embeddings,
                                 precision=precision)

    def similarity_speedy(self, text0: torch.Tensor, others: torch.Tensor) -> torch.Tensor:        
        old_sim = self.model.similarity_fn_name
        self.model.similarity_fn_name = "dot"
        sim = self.model.similarity(text0, others)
        self.model.similarity_fn_name = old_sim
        return sim



class VectorizedDataset:
    def __init__(self, texts: list[str], model: _EmbeddingGemmaModel, max_loaded_embeddings: int = 1_000_000):
        self.texts = texts
        self.model = model
        self.max_loaded_embeddings = max_loaded_embeddings
        self.embeddings = None
        
        
    @classmethod
    def from_texts(cls, texts: list[str], model: _EmbeddingGemmaModel):
        dataset = cls(texts, model)
        dataset.compute_embeddings()
        return dataset
    
    def save(self, path: str):
        """ Save the dataset to a file. """
        torch.save({
            "texts": self.texts,
            "embeddings": self.embeddings
        }, path)
        
    @classmethod
    def load(cls, path: str, model: _EmbeddingGemmaModel):
        """ Load the dataset from a file. """
        data = torch.load(path)
        dataset = cls(data["texts"], model)
        dataset.embeddings = data["embeddings"]
        return dataset
    
    def compute_embeddings(self, batch_size: int = 64, show_progress_bar: bool = True):
        self.embeddings = self.model.encode(self.texts, 
                                            convert_to_tensor=True, 
                                            batch_size=batch_size,
                                            normalize_embeddings=True,
                                            show_progress_bar=show_progress_bar)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "embedding": self.embeddings[idx]
        }

    def get_similar_batch(self, batch_size: int, delta: float, text_only: bool = False) -> list[dict | str]:
        """ Get a batch of items where all items are similar to each other above a certain cosine similarity threshold.
        
        Args:
            batch_size (int): Number of items in the batch.
            delta (float): Cosine similarity threshold (between 0 and 1).
            text_only (bool): If True, return only texts instead of dicts with text and embedding.

        Raises:
            ValueError: If embeddings are not computed.

        Returns:
            List[Dict[str, Any]]: A list of items from the dataset that are similar to each other.
        """
        
        if self.embeddings is None:
            raise ValueError("Embeddings not computed. Call compute_embeddings() first.")
        n = len(self.embeddings)
        # Select a random index
        idx = torch.randint(0, n, (1,)).item()
        selected_emb = self.embeddings[idx]
        
        # Collect similar indices in chunks to reduce memory usage
        similar_indices = []
        for i in range(0, n, self.max_loaded_embeddings):
            chunk_end = min(i + self.max_loaded_embeddings, n)
            chunk_embeddings = self.embeddings[i:chunk_end]
            # Compute similarities for this chunk
            sims = torch.matmul(chunk_embeddings, selected_emb)
            # Find similar indices in this chunk
            similar_in_chunk = (sims > delta).nonzero(as_tuple=True)[0] + i
            similar_indices.extend(similar_in_chunk.tolist())
        
        similar_indices = torch.tensor(similar_indices)
        
        if len(similar_indices) < batch_size:
            # If not enough similar items, return all similar ones
            batch_indices = similar_indices
        else:
            # Sample batch_size from similar indices
            perm = torch.randperm(len(similar_indices))
            batch_indices = similar_indices[perm[:batch_size]]
        # Return the batch as list of dicts or texts
        if text_only:
            return [self.texts[i.item()] for i in batch_indices]
        else:
            return [self[i.item()] for i in batch_indices]

