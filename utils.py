import os
import logging
import torch
from torch import nn
from abc import ABC, abstractmethod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class DIR:
    BASE_DIR = BASE_DIR
    GEMMA_EMBEDDING_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "embeddinggemma-300m")
    GEMMA_3_1B_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "gemma-3-1b-it")
    GEMMA_3_270M_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "gemma-3-270m-it")
    GEMMA_2_T5_270M_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "t5-gemma-2-270m-270m")
    
    GEMMA_3_DIR = GEMMA_3_1B_DIR  # Alias 

    DATA_DIR = os.path.join(BASE_DIR, "data")

    UFW_V1_4_DIR = os.path.join(DATA_DIR, "datasets", "parquet", "fineweb")
    
    KIMI_DIR = os.path.join(DATA_DIR, "datasets", "parquet", "KIMI-K2.5-550000x")
    
    REASNONING_DIR = os.path.join(DATA_DIR, "datasets", "parquet", "reasoning")


PATH = DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set warning color to yellow
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter("\033[93m%(levelname)s: %(message)s\033[0m")
ch.setFormatter(formatter)
logger.addHandler(ch)




FP64 = torch.float64
FP32 = torch.float32





class InvertibleModule(ABC):
    """Base class for invertible modules."""

    @abstractmethod
    def inverse(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Inverse method not implemented.")

    @abstractmethod
    def auto_inverse(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Automatically choose between exact and approximate inverse based on module properties."""
        raise NotImplementedError("Auto-inverse method not implemented.")



class SolvableModule(ABC):
    """Base class for modules that can be solved from a batch."""

    @abstractmethod
    def solve_from_batch(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Solve method not implemented.")
