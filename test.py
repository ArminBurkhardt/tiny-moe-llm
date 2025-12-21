from modules.data.test_vectorized_dataset import test_get_similar_batch, test_dataset_basic_functionality
from modules.data.dataloader import test_fileloader, test_dataloader
import torch
import logging

logging.basicConfig(level=logging.INFO)

logging.info(f"Torch version: {torch.__version__}")
logging.info(f"Torch CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    logging.info("Testing dataset functionalities...")
    test_dataset_basic_functionality()
    
    logging.info("Testing similarity batch retrieval...")
    test_get_similar_batch()
    
    logging.info("Testing file loader...")
    test_fileloader()
    
    logging.info("Testing data loader...")
    test_dataloader()
    
    logging.info("All tests passed!")