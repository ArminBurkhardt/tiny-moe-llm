import torch
from sentence_transformers import SentenceTransformer
from utils import BASE_DIR
from typing import Literal
from torch.utils.data import Dataset

# uses EmbeddingGemma-300m
# https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers?hl=de
# https://huggingface.co/google/embeddinggemma-300m

# or
# https://huggingface.co/LiquidAI/LFM2-ColBERT-350M 


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


class GemmaVectorDataset:
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

    def get_similar_batch(self, batch_size: int, delta: float, text_only: bool = False) -> tuple[list[dict | str], float]:
        """ Get a batch of items where all items are similar to each other above a certain cosine similarity threshold.
        
        Args:
            batch_size (int): Number of items in the batch.
            delta (float): Cosine similarity threshold (between 0 and 1).
            text_only (bool): If True, return only texts instead of dicts with text and embedding.

        Raises:
            ValueError: If embeddings are not computed.

        Returns:
            Tuple[List[Dict[str, Any] | str], float]: A tuple containing the batch of items and the mean pairwise cosine similarity of the batch.
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
        
        # TODO: Should pop items from the dataset to avoid duplicates in future batches
        if len(similar_indices) < batch_size:
            # If not enough similar items, return all similar ones
            batch_indices = similar_indices
        else:
            # Sample batch_size from similar indices
            perm = torch.randperm(len(similar_indices))
            batch_indices = similar_indices[perm[:batch_size]]
            
        if len(batch_indices) > 0:
            batch_embeddings = self.embeddings[batch_indices]
            sim_matrix = torch.matmul(batch_embeddings, batch_embeddings.T)
            mean_similarity = sim_matrix.mean().item()
        else:
            mean_similarity = 0.0
            
        # Return the batch as list of dicts or texts
        if text_only:
            return [self.texts[i.item()] for i in batch_indices], mean_similarity
        else:
            return [self[i.item()] for i in batch_indices], mean_similarity



################
# LFM2 ColBERT #
################



class _EmbeddingLFM2ColBERTModel:
    def __init__(self, device: str = None, index_folder: str = "pylate-index"):
        from pylate import indexes, models, retrieve
        self.model = models.ColBERT(
            model_name_or_path = BASE_DIR + "/ckpts/pretrained/LFM2-ColBERT-350M",
        )
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        if device:
            self.model = self.model.to(device)

        self.index_folder = BASE_DIR + "/data/index/" + index_folder
        self.index = None
        self.retriever = None

    def encode_documents(self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = True, padding: bool = False):
        return self.model.encode(texts, batch_size=batch_size, is_query=False, show_progress_bar=show_progress_bar, convert_to_tensor=True, padding=padding)

    def encode_queries(self, queries: list[str], batch_size: int = 32, show_progress_bar: bool = False, padding: bool = False):
        return self.model.encode(queries, batch_size=batch_size, is_query=True, show_progress_bar=show_progress_bar, convert_to_tensor=True, padding=padding)

    def build_index(self, document_ids: list[str], documents_embeddings):
        from pylate import indexes, retrieve
        self.index = indexes.PLAID(
            index_folder=self.index_folder,
            index_name="index",
            override=True,
        )
        self.index.add_documents(
            documents_ids=document_ids,
            documents_embeddings=documents_embeddings,
        )
        self.retriever = retrieve.ColBERT(index=self.index)
        
    def load_index(self):
        from pylate import indexes, retrieve
        self.index = indexes.PLAID(
            index_folder=self.index_folder,
            index_name="index",
            override=False,
        )
        self.retriever = retrieve.ColBERT(index=self.index)

    def retrieve(self, queries_embeddings, k: int = 10):
        if self.retriever is None:
            raise ValueError("Retriever is not initialized. Build or load the index first.")
        return self.retriever.retrieve(queries_embeddings=queries_embeddings, k=k)

    def similarity(self, batch_of_embeddings: torch.Tensor):
        """ Compute pairwise similarity between embeddings in a batch. """
        # batch_of_embeddings shape: (batch_size, max_tokens, embedding_dim)
        # compute similarity between all pairs of embeddings in the batch
        # return a matrix of shape (batch_size, batch_size)
        # TODO: implement this
        return NotImplemented


class LFM2ColBERTVectorDataset:
    def __init__(self, texts: list[str], model: _EmbeddingLFM2ColBERTModel):
        self.texts = {str(i): text for i, text in enumerate(texts)}
        self.model = model
        
    @classmethod
    def from_texts(cls, texts: list[str], model: _EmbeddingLFM2ColBERTModel):
        dataset = cls(texts, model)
        dataset.compute_embeddings()
        return dataset
    
    def save(self, path: str):
        """ Save the dataset to a file. """
        torch.save({
            "texts": self.texts,
        }, path)
    
    @classmethod
    def load(cls, path: str, model: _EmbeddingLFM2ColBERTModel):
        """ Load the dataset from a file. """
        data = torch.load(path)
        dataset = cls(list(data["texts"].values()), model)
        dataset.texts = data["texts"]
        return dataset
    
    def compute_embeddings(self, batch_size: int = 32, show_progress_bar: bool = True):
        texts_list = list(self.texts.values())
        ids_list = list(self.texts.keys())
        embeddings = self.model.encode_documents(texts_list, batch_size=batch_size, show_progress_bar=show_progress_bar)
        self.model.build_index(ids_list, embeddings)
        
    def __len__(self):
        return len(self.texts)

    def get_similar_batch(self, batch_size: int, text_only: bool = False, delta: float = 0.5, return_embeddings: bool = False) -> tuple[list[dict | str], float]:
        """ Get a batch of items where all items are similar to each other above a certain cosine similarity threshold.
        
        Args:
            batch_size (int): Number of items in the batch.
            text_only (bool): If True, return only texts instead of dicts with text and embedding.
            return_embeddings (bool): If True, return embeddings for the texts.
            [deprecated] delta (float): Cosine similarity threshold (between 0 and 1).

        Raises:
            ValueError: If embeddings are not computed.

        Returns:
            Tuple[List[Dict[str, Any] | str], float]: A batch and the mean pairwise similarity.
        """
        # TODO: solve conflict with delta (similarity threshold) and batch_size (number of items to return). Currently, delta is ignored.
        # FIXME: remove all usages of delta, thus making mandatory to ignore delta
        if not self.texts:
            raise ValueError("Dataset is empty.")
        if self.model.retriever is None:
            raise ValueError("Embeddings and index not computed. Call compute_embeddings() first.")
            
        # Select a random document as query
        query_id = list(self.texts.keys())[torch.randint(0, len(self.texts), (1,)).item()]
        query_text = self.texts[query_id]
        
        query_embedding = self.model.encode_queries([query_text], batch_size=1)
        
        # Retrieve batch_size items (might return fewer if dataset is small or matched items are few)
        # We request a bit more in case some IDs were already popped, though the index might still return them.
        # But we actually want to pop them so we can't fetch them next time.
        # pylate retrieve returns a list of dicts for each query.
        retrieved = self.model.retrieve(query_embedding, k=batch_size * 2)
        
        results = retrieved[0] # Results for the first (and only) query
        
        batch_items = []
        
        # If it's a dict mapping id -> rank/score
        if type(results) is dict:
            sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for doc_id, _ in sorted_items:
                if str(doc_id) in self.texts:
                    batch_items.append((str(doc_id), self.texts.pop(str(doc_id))))
                if len(batch_items) == batch_size:
                    break
        elif type(results) is list:
            for item in results:
                doc_id = item['id'] if isinstance(item, dict) else item
                if str(doc_id) in self.texts:
                    batch_items.append((str(doc_id), self.texts.pop(str(doc_id))))
                if len(batch_items) == batch_size:
                    break
                    
        # If we couldn't get enough from retriever, just pop random elements
        while len(batch_items) < batch_size and self.texts:
            doc_id = list(self.texts.keys())[0]
            batch_items.append((str(doc_id), self.texts.pop(str(doc_id))))
            
        if text_only:
            batch_result = [text for _, text in batch_items]
        else:
            if return_embeddings:
                embeddings = self.model.encode_documents([text for _, text in batch_items], batch_size=batch_size, padding=True)
                batch_result = [{"id": doc_id, 
                        "text": text, 
                        "embedding": embedding} 
                        for (doc_id, text), embedding in zip(batch_items, embeddings)]
            else:
                batch_result = [{"id": doc_id, 
                        "text": text} 
                        for doc_id, text in batch_items]
        
        return batch_result, 0.0


VectorDataset = GemmaVectorDataset
"""
Wrapper to provide a consistent interface for different embedding models.
"""

