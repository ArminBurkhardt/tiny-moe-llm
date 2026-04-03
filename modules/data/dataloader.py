import os
import pandas as pd
from utils import BASE_DIR, DIR, logger

__DATASETS = [
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    "https://huggingface.co/datasets/TeichAI/claude-4.5-opus-high-reasoning-250x",
    "https://huggingface.co/datasets/openbmb/Ultra-FineWeb", # https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_en_v1_4 
]



class FileLoader:
    def __init__(self, root: str):
        self.root = root
        self.subdirs = sorted(os.listdir(root))
        self.current_subdir = 0
        self.current_subfile = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        next_file = self._get_next_file()
        if next_file is None:
            raise StopIteration
        data = self.load_file(next_file)
        return data
    
    def reset(self):
        self.current_subdir = 0
        self.current_subfile = 0

    def _get_next_file(self):
        if self.current_subdir >= len(self.subdirs):
            return None
        subdir_path = os.path.join(self.root, self.subdirs[self.current_subdir])
        subfiles = sorted(os.listdir(subdir_path))
        if self.current_subfile >= len(subfiles):
            self.current_subdir += 1
            self.current_subfile = 0
            return self._get_next_file()
        file_path = os.path.join(subdir_path, subfiles[self.current_subfile])
        self.current_subfile += 1
        return file_path

    @staticmethod
    def load_file(parquet_file_path: str) -> pd.DataFrame:
        df = pd.read_parquet(parquet_file_path)
        return df
        
        

class DataLoader:
    def __init__(self, data_root: str, drop_last: bool = False, batch_size: int = 32, minimum_score: float = 0.0, target_column: str = None):
        self.data_root = data_root
        self.file_loader = FileLoader(data_root)
        self.current_data = None
        self.current_idx = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.minimum_score = minimum_score
        self.target_column = target_column
        self.load_next_file()
        
    def load_next_file(self):
        try:
            self.current_data = next(self.file_loader)
            if self.minimum_score > 0.0:
                self.current_data = self.current_data[self.current_data["score"] >= self.minimum_score]
            self.current_idx = 0
        except StopIteration:
            self.current_data = None
            
    def get_next_file(self) -> pd.DataFrame | None:
        try:
            data = next(self.file_loader)
            if self.minimum_score > 0.0:
                data = data[data["score"] >= self.minimum_score]
            if self.target_column is not None:
                data = data[[self.target_column]]
            return data
        except StopIteration:
            return None
    
    def get_next_batch(self, batch_size: int) -> pd.DataFrame:
        if self.current_data is None or self.current_idx >= len(self.current_data):
            self.load_next_file()
            if self.current_data is None:
                return None
        start_idx = self.current_idx
        end_idx = min(start_idx + batch_size, len(self.current_data))
        if self.drop_last and (end_idx - start_idx) < batch_size:
            return None
        batch = self.current_data.iloc[start_idx:end_idx]
        self.current_idx = end_idx
        if self.target_column is not None:
            batch = batch[[self.target_column]]
        return batch
    
    def __iter__(self):
        return self
    
    def __next__(self) -> pd.DataFrame:
        batch = self.get_next_batch(batch_size=self.batch_size)
        if batch is None:
            raise StopIteration
        return batch
    





def test_fileloader():
    data_roots = [DIR.UFW_V1_4_DIR]
    for data_root in data_roots:
        loader = FileLoader(data_root)
        for i, data in enumerate(loader):
            logger.info(f"Loaded file {i} with {len(data)} records.")
            logger.info(data.head())
    
        
def test_dataloader():
    data_roots = [DIR.UFW_V1_4_DIR]
    for data_root in data_roots:
        dataloader = DataLoader(data_root, drop_last=True, batch_size=2048, minimum_score=0.5, target_column="content")
        for i, batch in enumerate(dataloader):
            logger.info(f"Batch {i} with {len(batch)} records.")
        
        logger.info(batch.head())



