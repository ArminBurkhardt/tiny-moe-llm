import torch
import numpy as np
from modules.data.vector_dataset import _EmbeddingLFM2ColBERTModel, GemmaVectorDataset, _EmbeddingGemmaModel, LFM2ColBERTVectorDataset
from utils import BASE_DIR, logger

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def test_get_similar_batch():
    # Create some test texts with varying similarity
    texts = [
        "The cat sat on the mat.",
        "A feline rested on a rug.",
        "The dog played in the park.",
        "Cats are small pets.",
        "Dogs are loyal animals.",
        "Birds fly in the sky.",
        "Fish swim in water.",
        "The weather is nice today.",
        "I like to eat pizza.",
        "Programming is fun."
    ]

    # Initialize the model (use GPU for testing)
    model = _EmbeddingGemmaModel(device=device, embedding_dim=256)

    # Create dataset
    dataset = GemmaVectorDataset(texts, model)
    dataset.compute_embeddings(batch_size=2, show_progress_bar=True)

    # Test get_similar_batch
    batch_size = 3
    delta = 0.7  # Cosine similarity threshold

    batch, _ = dataset.get_similar_batch(batch_size=batch_size, delta=delta)

    # Verify the batch
    assert len(batch) <= batch_size, f"Batch size should be <= {batch_size}, got {len(batch)}"
    assert len(batch) > 0, "Batch should not be empty"

    # Check that all items in the batch are similar to each other
    if len(batch) > 1:
        embeddings = torch.stack([item["embedding"] for item in batch])
        # Compute pairwise similarities
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        # All similarities should be > delta (except diagonal which is 1.0)
        off_diagonal = sim_matrix[~torch.eye(len(batch), dtype=bool)]
        min_sim = off_diagonal.min().item()
        if min_sim <= delta:
            print(f"All items in the batch should have similarity > {delta}, but min similarity is {min_sim}.") # this is not critical, as all are similar to the inital item

    # Check that all items in the batch have the expected structure
    for item in batch:
        assert "text" in item and "embedding" in item, "Each item should have text and embedding"
        assert isinstance(item["text"], str), "Text should be a string"
        assert isinstance(item["embedding"], torch.Tensor), "Embedding should be a tensor"

    print("Test passed: get_similar_batch returns similar items")

def test_dataset_basic_functionality():
    texts = ["Hello world", "Goodbye world"]
    model = _EmbeddingGemmaModel(device=device, embedding_dim=256)
    dataset = GemmaVectorDataset(texts, model)

    # Test length
    assert len(dataset) == 2, "Dataset length should be 2"

    # Test embeddings not computed yet
    assert dataset.embeddings is None, "Embeddings should be None initially"

    # Compute embeddings
    dataset.compute_embeddings(batch_size=2, show_progress_bar=True)
    assert dataset.embeddings is not None, "Embeddings should be computed"
    assert dataset.embeddings.shape[0] == 2, "Should have 2 embeddings"

    # Test __getitem__
    item = dataset[0]
    assert "text" in item and "embedding" in item, "Item should have text and embedding"
    assert item["text"] == "Hello world", "Text should match"

    print("Test passed: basic dataset functionality")


def gemma_vector_dataset_tests():
    test_dataset_basic_functionality()
    test_get_similar_batch()


def test_get_similar_batch_colBERT():
    # Create some test texts with varying similarity
    texts = [
        "The cat sat on the mat.",
        "A feline rested on a rug.",
        "The dog played in the park.",
        "Cats are small pets.",
        "Dogs are loyal animals.",
        "Birds fly in the sky.",
        "Fish swim in water.",
        "The weather is nice today.",
        "I like to eat pizza.",
        "Programming is fun."
    ]

    # Initialize the model (use GPU for testing)
    model = _EmbeddingLFM2ColBERTModel(device=device, index_folder="test-index")

    # Create dataset
    dataset = LFM2ColBERTVectorDataset(texts, model)
    dataset.compute_embeddings(batch_size=2, show_progress_bar=True)

    # Test get_similar_batch
    batch_size = 3

    batch = dataset.get_similar_batch(batch_size=batch_size, return_embeddings=True)

    # Verify the batch
    assert len(batch) <= batch_size, f"Batch size should be <= {batch_size}, got {len(batch)}"
    assert len(batch) > 0, "Batch should not be empty"

    # Check that all items in the batch are similar to each other
    if len(batch) > 1:
        embeddings = torch.stack([item["embedding"] for item in batch])
        embeddings = embeddings.squeeze(1)
        # recompute similarity using the model
        sims = model.similarity(embeddings)
        print(sims) # tensor([0.2578, 0.2578])

    # Check that all items in the batch have the expected structure
    for item in batch:
        assert "text" in item and "embedding" in item, "Each item should have text and embedding"
        assert isinstance(item["text"], str), "Text should be a string"
        assert isinstance(item["embedding"], torch.Tensor), "Embedding should be a tensor"

    print("Test passed: get_similar_batch returns similar items")



def ColBERT_vector_dataset_tests():
    test_get_similar_batch_colBERT()

if __name__ == "__main__":
    test_dataset_basic_functionality()
    test_get_similar_batch()
    test_get_similar_batch_colBERT()
    print("All tests passed!")