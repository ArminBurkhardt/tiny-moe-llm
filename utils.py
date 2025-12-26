import os
import logging
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class DIR:
    BASE_DIR = BASE_DIR
    GEMMA_EMBEDDING_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "embeddinggemma-300m")
    GEMMA_3_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "gemma-3-1b-it")

    DATA_DIR = os.path.join(BASE_DIR, "data")

    UFW_V1_4_DIR = os.path.join(DATA_DIR, "datasets", "ultrafineweb_en_v1_4")

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










