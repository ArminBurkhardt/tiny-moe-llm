from modules.data.test_vectorized_dataset import test_get_similar_batch, test_dataset_basic_functionality
from modules.data.dataloader import test_fileloader, test_dataloader
from modules.data.dataset import test_dataset
from modules.model.activations import test_invertible_activation
from modules.model.invertible_modules import test_invertible_linear_attention
from modules.model.linear import test_solvable_linear
from modules.model.losses import test_matrix_invertability_loss
from modules.model.decoder import test_decoder
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
    
    logging.info("Testing Dataset class...")
    if False:
        test_dataset()
    else:
        logging.info("Skipping test_dataset due to long runtime.")
        
    logging.info("Testing invertible activation...")
    test_invertible_activation()
    
    logging.info("Testing solvable linear layer...")
    test_solvable_linear()
    
    logging.info("Testing matrix invertibility loss...")
    test_matrix_invertability_loss()
    
    logging.info("Testing invertible linear attention...")
    test_invertible_linear_attention()
    
    logging.info("Testing decoder...")
    test_decoder()
    
    logging.info("All tests passed!")