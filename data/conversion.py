import math
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import BASE_DIR, DIR

def convert_jsonl_to_parquet(jsonl_file_path: str, parquet_file_path: str, target_10kentries: int = 5):
    df = pd.read_json(jsonl_file_path, lines=True)
    
    df.dropna(inplace=True)
    
    total_memory_bytes = df.memory_usage(deep=True).sum()
    target_bytes = target_10kentries * 1024 * 1024
    
    num_chunks = max(1, math.ceil(total_memory_bytes / target_bytes))
    rows_per_chunk = int(len(df) / num_chunks)
    
    for i in range(num_chunks):
        start_row = i * rows_per_chunk
        end_row = (i + 1) * rows_per_chunk if i < num_chunks - 1 else len(df)
        chunk_df = df.iloc[start_row:end_row]
        
        chunk_parquet_file_path = parquet_file_path.replace('.parquet', f'-part{i+1:03d}-of-{num_chunks}.parquet')
        chunk_df.to_parquet(chunk_parquet_file_path, index=False)
        print(f"Converted chunk {i+1}/{num_chunks} to {chunk_parquet_file_path} with {len(chunk_df)} records.")



def convert_all_jsonl_in_directory(jsonl_dir: str, parquet_dir: str):
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)
    
    for filename in os.listdir(jsonl_dir):
        if filename.endswith('.jsonl'):
            jsonl_file_path = os.path.join(jsonl_dir, filename)
            parquet_file_name = filename.replace('.jsonl', '.parquet')
            parquet_file_path = os.path.join(parquet_dir, parquet_file_name)
            convert_jsonl_to_parquet(jsonl_file_path, parquet_file_path)



def convert_kimi_jsonl_to_parquet():
    jsonl_dir = os.path.join(DIR.DATA_DIR, "datasets", "jsonl", "KIMI-K2.5-550000x")
    parquet_dir = DIR.KIMI_DIR
    convert_all_jsonl_in_directory(jsonl_dir, parquet_dir)


def convert_reasoning_jsonl_to_parquet():
    jsonl_dir = os.path.join(DIR.DATA_DIR, "datasets", "jsonl", "claude-opus-4.6-10000x")
    parquet_dir = os.path.join(DIR.DATA_DIR, "datasets", "parquet", "reasoning")
    convert_all_jsonl_in_directory(jsonl_dir, parquet_dir)
    
    jsonl_dir = os.path.join(DIR.DATA_DIR, "datasets", "jsonl", "Claude-Sonnet-4.6-Reasoning-1100x")
    parquet_dir = os.path.join(DIR.DATA_DIR, "datasets", "parquet", "reasoning")
    convert_all_jsonl_in_directory(jsonl_dir, parquet_dir)

if __name__ == "__main__":
    convert_reasoning_jsonl_to_parquet()





