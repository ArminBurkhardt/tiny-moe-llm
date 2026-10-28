"""Chat read: does the model hold a chat turn and stop at the end of it?

Decodes a fixed set of prompts (``scripts/reads/chat_prompts.jsonl``: everyday questions, factual
ones, instructions, two multi-turn conversations and two with a system prompt) greedily through the
SFT chat template, no evidence attached, up to ``--max-new-tokens`` new tokens, and reports

* stop rate: the share of completions that ended with EOS before the cap,
* mean length in new tokens (the EOS counts),
* loop rate: the share of completions with a repeated 4-gram (a 4-token window that occurs again
  later in the same completion, so a degenerate repeat shows at once).

Every generation is written to the json for a human to read: the numbers say whether the model stops
and repeats, not whether the answers are any good. The prompt builder is ``ChatTemplate.encode_prompt``
as ``scripts/inference.py --chat`` uses it. The decode is the KV cached greedy loop of
``inference.stream_generate`` without its text streaming, because the stop rate needs the token ids.

Usage:
    python scripts/eval_chat.py -c CKPT [--prompts scripts/reads/chat_prompts.jsonl]
        [--max-new-tokens 256] [--json-out OUT.json]

Args:
    -c, --checkpoint: model checkpoint (any file ``inference.load_model`` reads).
    --prompts: jsonl, one ``{"id", "category", "messages"}`` per line.
    --max-new-tokens: cap per completion (default 256).
    --json-out: summary plus every generation.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reads", "chat_prompts.jsonl")


def has_repeated_ngram(ids, n: int = 4) -> bool:
    """Whether some n-gram of ``ids`` occurs at two or more positions."""
    seen = set()
    for i in range(len(ids) - n + 1):
        gram = tuple(ids[i:i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def summarise(records) -> dict:
    """Stop rate, mean length and repeated 4-gram rate over per prompt records.

    Args:
        records: dicts with ``stopped`` (bool), ``new_tokens`` (int) and ``repeat4`` (bool).
    """
    n = max(len(records), 1)
    return {
        "n": len(records),
        "stop_rate": sum(r["stopped"] for r in records) / n,
        "mean_new_tokens": sum(r["new_tokens"] for r in records) / n,
        "repeat4_rate": sum(r["repeat4"] for r in records) / n,
    }


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--checkpoint", required=True)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from modules.data.chat import ChatTemplate
    from modules.model.kv_cache import KVCache
    from config import ModelConfig
    from scripts.inference import load_model
    from utils import TOKENIZER_DIR

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    model = load_model(a.checkpoint, a.device)
    max_len = ModelConfig.Params["max_seq_len"]
    eos = tokenizer.eos_token_id
    prompts = load_prompts(a.prompts)
    records = []
    with torch.no_grad():
        for p in prompts:
            prompt_ids = template.encode_prompt(p["messages"])[-(max_len - a.max_new_tokens - 1):]
            kv = KVCache.for_model(model, n_loops=None)
            step = torch.tensor([prompt_ids], device=a.device)
            out, stopped = [], False
            while len(out) < a.max_new_tokens:
                res = model(step, skip_mtp=True, n_loops=None, kv_cache=kv)
                logits = res[0] if isinstance(res, tuple) else res
                nxt = int(logits[:, -1, :].argmax(dim=-1).item())
                out.append(nxt)
                if nxt == eos:
                    stopped = True
                    break
                step = torch.tensor([[nxt]], device=a.device)
            text = tokenizer.decode([t for t in out if t != eos], skip_special_tokens=True)
            records.append({
                "id": p["id"], "category": p["category"], "messages": p["messages"],
                "generation": text, "stopped": stopped, "new_tokens": len(out),
                "repeat4": has_repeated_ngram(out),
            })
            print(f"[{p['id']:2d}] {p['category']:<11} new {len(out):3d} stopped {stopped} repeat4 {records[-1]['repeat4']}", flush=True)
    summary = summarise(records)
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    summary["by_category"] = {c: summarise(rs) for c, rs in by_cat.items()}
    print(f"chat read: n {summary['n']}  stop rate {summary['stop_rate']:.3f}  mean new tokens "
          f"{summary['mean_new_tokens']:.1f}  repeated 4-gram rate {summary['repeat4_rate']:.3f}")
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": a.checkpoint, "max_new_tokens": a.max_new_tokens,
                       "summary": summary, "records": records}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
