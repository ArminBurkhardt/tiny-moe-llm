import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import DIR

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def generate_synthetic_reasoning_data(num_samples=None):
    """
    Generates a dataset using templates to ensure diversity and separation.
    Returns disjoint train and test sets.
    """
    subjects = ["The movie", "The food", "The book", "The game", "The weather", "The result", "The answer", "The service", "The play", "The song"]
    pos_adjs = ["good", "great", "excellent", "amazing", "nice", "correct", "right", "happy", "superb", "wonderful"]
    neg_adjs = ["bad", "terrible", "awful", "horrible", "nasty", "wrong", "incorrect", "sad", "poor", "dreadful"]
    
    data = []
    
    # 1. Simple Positive: "{Subject} was {Pos}" -> 1
    for s in subjects:
        for adj in pos_adjs:
            data.append((f"{s} was {adj}.", 1))
            
    # 2. Simple Negative: "{Subject} was {Neg}" -> 0
    for s in subjects:
        for adj in neg_adjs:
            data.append((f"{s} was {adj}.", 0))
            
    # 3. Negated Negative (Positive meaning): "{Subject} was not {Neg}" -> 1
    for s in subjects:
        for adj in neg_adjs:
            data.append((f"{s} was not {adj}.", 1))
            
    # 4. Negated Positive (Negative meaning): "{Subject} was not {Pos}" -> 0
    for s in subjects:
        for adj in pos_adjs:
            data.append((f"{s} was not {adj}.", 0))
            
    # Shuffle and Split
    random.seed(42)
    random.shuffle(data)
    
    # Split 80/20
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_texts = [x[0] for x in train_data]
    train_labels = np.array([x[1] for x in train_data])
    
    test_texts = [x[0] for x in test_data]
    test_labels = np.array([x[1] for x in test_data])
    
    print(f"Generated {len(train_texts)} training samples and {len(test_texts)} test samples.")
    return train_texts, train_labels, test_texts, test_labels

def extract_hidden_states(model, tokenizer, texts, device):
    """
    Extracts hidden states from ALL layers for the given texts.
    Handles both Decoder-only (Gemma) and Encoder-Decoder (T5) architectures.
    """
    model.eval()
    all_layer_states = []
    
    print(f"Extracting features from {len(texts)} samples...")
    
    # Process in small batches to save RAM
    batch_size = 8
    
    # Initialize list to hold lists of layer outputs
    # Structure: [Layer_Index][Sample_Index] -> vector
    num_layers = model.config.num_hidden_layers + 1 # +1 for embedding layer
    layer_data = [[] for _ in range(num_layers)]

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # --- Architecture Handling ---
        
        # Case A: Encoder-Decoder (e.g., T5) -> We probe the ENCODER
        if model.config.is_encoder_decoder:
            # stack: (num_layers, batch, seq, dim)
            hidden_states = outputs.encoder_hidden_states
            
            # Pooling: Mean pool over non-pad tokens for the encoder
            mask = inputs.attention_mask.unsqueeze(-1)
            for layer_idx, layer_tensor in enumerate(hidden_states):
                # layer_tensor: (batch, seq, dim)
                sum_embeddings = torch.sum(layer_tensor * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask # (batch, dim)
                
                layer_data[layer_idx].append(pooled.cpu().numpy())

        # Case B: Decoder-only (e.g., Gemma) -> We probe the DECODER
        else:
            hidden_states = outputs.hidden_states
            
            # Pooling: Take the LAST token (standard for causal reasoning)
            # Use attention_mask to find the last non-pad token index
            batch_indices = torch.arange(inputs.input_ids.shape[0], device=device)
            last_token_indices = inputs.attention_mask.sum(dim=1) - 1
            
            for layer_idx, layer_tensor in enumerate(hidden_states):
                # Select specific token vectors
                pooled = layer_tensor[batch_indices, last_token_indices, :] # (batch, dim)
                layer_data[layer_idx].append(pooled.cpu().numpy())

    # Concatenate batches
    final_layer_data = [np.concatenate(batches, axis=0) for batches in layer_data]
    return final_layer_data

def run_linear_probe(model_name, dataset_size=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    
    try:
        # Load generic model to get access to raw hidden states
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. Generate Data
    train_texts, train_labels, test_texts, test_labels = generate_synthetic_reasoning_data()
    
    # 2. Extract Hidden States
    # Combine to extract in one go
    all_texts = train_texts + test_texts
    
    # Returns list of numpy arrays: [layer_0_features, layer_1_features, ...]
    layer_features = extract_hidden_states(model, tokenizer, all_texts, device)
    
    split_idx = len(train_texts)
    
    # 3. Train Probes
    print("\nTraining Linear Probes on each layer...")
    accuracies = []
    
    for layer_idx, X_all in enumerate(layer_features):
        # Split data manually
        X_train = X_all[:split_idx]
        X_test = X_all[split_idx:]
        y_train = train_labels
        y_test = test_labels
        
        # Train simple linear classifier (Logistic Regression)
        # fast solver for small datasets
        clf = LogisticRegression(max_iter=1000, solver='liblinear', C=0.1) 
        clf.fit(X_train, y_train)
        
        # Evaluate
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        # print(f"Layer {layer_idx}: {acc:.4f}") # Uncomment for verbose output

    # 4. Results
    best_layer = np.argmax(accuracies)
    # find last occurrence of best accuracy in case of ties
    best_accuracy = accuracies[best_layer]
    for i in range(len(accuracies)-1, -1, -1):
        if accuracies[i] == best_accuracy:
            best_layer = i
            break
    print("\n" + "="*40)
    print(f"ANALYSIS RESULTS FOR: {model_name}")
    print("="*40)
    print(f"Total Layers found: {len(accuracies)}")
    print(f"Peak 'Thinking' Layer: {best_layer} (Accuracy: {accuracies[best_layer]:.2%})")
    
    # Simple ASCII Plot
    print("\nAccuracy Trajectory:")
    for i, acc in enumerate(accuracies):
        bar = "#" * int((acc - 0.5) * 50) if acc > 0.5 else ""
        prefix = ">> " if i == best_layer else "   "
        print(f"{prefix}Layer {i:02d}: {acc:.2f} | {bar}")

    print("\nRECOMMENDATION:")
    print(f"Insert your new fine-tuning layer at index: {best_layer}")
    print("="*40)

if __name__ == "__main__":
    # REPLACE THIS with your specific local path or huggingface ID
    # Examples: "google/gemma-2b" or "google/t5-base"
    # For your specific request:
    TARGET_MODEL = DIR.GEMMA_3_1B_DIR  # "google/gemma-2-2b-it" 
    
    # Note: Using "gemma-2-2b-it" as a proxy since 270m isn't public yet. 
    # Swap this string with your local path to the 270m model.
    
    run_linear_probe(TARGET_MODEL)

"""
Gemma 3 1B Model Results Sample Output:
Accuracy Trajectory:
   Layer 00: 0.70 | #########
   Layer 01: 1.00 | #########################
   Layer 02: 1.00 | #########################
   Layer 03: 1.00 | #########################
   Layer 04: 1.00 | #########################
   Layer 05: 1.00 | #########################
   Layer 06: 1.00 | #########################
   Layer 07: 1.00 | #########################
   Layer 08: 1.00 | #########################
>> Layer 09: 1.00 | #########################
   Layer 10: 0.99 | ########################
   Layer 11: 0.99 | ########################
   Layer 12: 0.99 | ########################
   Layer 13: 0.99 | ########################
   Layer 14: 0.99 | ########################
   Layer 15: 0.99 | ########################
   Layer 16: 0.97 | #######################
   Layer 17: 0.97 | #######################
   Layer 18: 0.97 | #######################
   Layer 19: 0.97 | #######################
   Layer 20: 0.97 | #######################
   Layer 21: 0.97 | #######################
   Layer 22: 0.99 | ########################
   Layer 23: 0.99 | ########################
   Layer 24: 0.99 | ########################
   Layer 25: 0.99 | ########################
   Layer 26: 0.99 | ########################
"""

