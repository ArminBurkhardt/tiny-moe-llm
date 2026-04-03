import os
import sys
import pandas as pd

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

if __name__ == "__main__":
    data_roots = [DIR.UFW_V1_4_DIR, DIR.KIMI_DIR, DIR.REASNONING_DIR]
    for data_root in data_roots:
        print(f"Inspecting files in {data_root}...")
        inspect_file(data_root)
        
        



