import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from config import ModelConfig
from utils import BASE_DIR, BF16


def load_model(checkpoint_path: str, device: str):
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16)
    model.set_checkpointing(False, False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def generate(
    model: TinyMoETransformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        # truncate to max_seq_len to avoid OOM on long contexts
        ids = input_ids[:, -ModelConfig.Params["max_seq_len"]:]
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        next_logits = logits[:, -1, :]  # (1, vocab)

        if temperature > 0:
            next_logits = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated = input_ids[0, tokenizer.encode(prompt, return_tensors="pt").shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_loop(model, tokenizer, args, device):
    print("TinyMoE inference — type your prompt and press Enter. Ctrl-C or 'quit' to exit.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        output = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, device)
        print(output)
        print()



def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    best_ts, best_path = 0, None
    if not os.path.isdir(checkpoint_dir):
        return None
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("checkpoint") and fname.endswith(".pt"):
            fpath = os.path.join(checkpoint_dir, fname)
            ts = os.path.getmtime(fpath)
            if ts > best_ts:
                best_ts, best_path = ts, fpath
    return best_path

def main():
    parser = argparse.ArgumentParser(description="TinyMoE inference CLI")
    parser.add_argument(
        "--checkpoint", "-c",
        default=find_latest_checkpoint(os.path.join(BASE_DIR, "ckpts", "training")),
        help="Path to a .pt checkpoint file",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default=os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer"),
        help="Path to the tokenizer directory (default: DeepSeek-V4-Pro-tokenizer)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Single prompt to generate from. Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--max-new-tokens", "-n",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; 0 = greedy (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering; 0 = disabled (default: 50)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer} …")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading checkpoint from {args.checkpoint} …")
    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {args.device} ({sum(p.numel() for p in model.parameters()):,} params)\n")

    if args.prompt is not None:
        output = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.device)
        print(output)
    else:
        interactive_loop(model, tokenizer, args, args.device)


if __name__ == "__main__":
    main()
