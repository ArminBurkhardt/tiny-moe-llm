import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import BASE_DIR, DIR, logger

print(f"BASE_DIR: {BASE_DIR}")



def inspect_file(data_root: str):
    subdirs = sorted(os.listdir(data_root))
    for subdir in subdirs:
        subdir_path = os.path.join(data_root, subdir)
        if os.path.isdir(subdir_path):
            files = sorted(os.listdir(subdir_path))
            for file in files:
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path) and file.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    print(f"File: {file_path}, Records: {len(df)}, Columns: {df.columns.tolist()}")
                    print(df.head(10))


def count_pretraing_tokens(data_root: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    total_tokens = 0
    subdirs = sorted(os.listdir(data_root))
    for subdir in subdirs:
        subdir_path = os.path.join(data_root, subdir)
        if os.path.isdir(subdir_path):
            files = sorted(os.listdir(subdir_path))
            for file in tqdm(files, desc=f"Processing files in {subdir_path}"):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path) and file.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    # concat and tokenize all content in the 'content' column
                    contents = df['content'].astype(str).tolist()
                    tokens = tokenizer(contents, truncation=False, padding=False)['input_ids']
                    total_tokens += sum(len(t) for t in tokens) * len(files) # multiply by number of files in subdir to estimate total tokens in subdir
                break # only process one file per subdir for estimation purposes
    print(f"Total training tokens in {data_root}: {total_tokens / 1e9:.2f} billion")



if __name__ == "__main__":
    data_roots = [DIR.KIMI_DIR, DIR.REASNONING_DIR]
    for data_root in data_roots:
        print(f"Inspecting files in {data_root}...")
        inspect_file(data_root)
    
    
    count_pretraing_tokens(DIR.UFW_V1_4_DIR)
        


