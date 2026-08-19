import os
import sys
import argparse
from typing import Iterator

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from modules.model.kv_cache import KVCache
from modules.data.chat import ChatTemplate
from config import ModelConfig
from utils import BASE_DIR, BF16, TOKENIZER_DIR


def load_model(checkpoint_path: str, device: str):
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16)
    model.set_checkpointing(False, False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _apply_repetition_penalty(next_logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Standard (Keskar et al.) repetition penalty: shrink already-seen tokens toward 0 logit."""
    seen = torch.unique(generated_ids)
    seen_logits = next_logits[0, seen]
    next_logits[0, seen] = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
    return next_logits


def _banned_ngram_tokens(generated: list[int], no_repeat_ngram_size: int) -> list[int]:
    """Tokens that would complete an n-gram already seen earlier in `generated`."""
    n = no_repeat_ngram_size
    if n <= 0 or len(generated) < n:
        return []
    prefix = tuple(generated[-(n - 1):]) if n > 1 else ()
    banned = [
        generated[i + n - 1]
        for i in range(len(generated) - n + 1)
        if tuple(generated[i:i + n - 1]) == prefix
    ]
    return banned


def _sample_next_token(next_logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Greedy (temperature <= 0) or temperature/top-k/top-p sampling. next_logits: [1, vocab]."""
    if temperature <= 0:
        return next_logits.argmax(dim=-1, keepdim=True)

    next_logits = next_logits / temperature
    if top_k > 0:
        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < v[:, [-1]]] = float("-inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cum_probs > top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove_mask = torch.zeros_like(sorted_remove).scatter(1, sorted_idx, sorted_remove)
        next_logits = next_logits.masked_fill(remove_mask, float("-inf"))

    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def stream_generate(
    model: TinyMoETransformer,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    top_p: float = 0.0,
    n_loops: int | None = None,
    num_mtp_tokens: int = 0,
    use_kv_cache: bool = True,
    converge_tol: float | None = None,
    min_loops: int = 1,
) -> Iterator[str]:
    """Generate tokens one step at a time, yielding the newly decoded text after each step.

    With ``use_kv_cache`` (default), each step only forwards the tokens newly appended since the
    previous step -- earlier tokens' attention K/V, at every dense-decoder layer and every MoE
    loop's attention experts, are reused from a ``KVCache`` built fresh for this call (see
    ``modules/model/kv_cache.py``; the model's causal structure is what makes that reuse exact, not
    approximate). With it disabled, every step re-runs the full (truncated) prefix -- a slower
    reference path, useful for cross-checking the cached path's output.

    ``num_mtp_tokens`` additionally drafts up to that many extra tokens per step from the MTP
    head's parallel predictions off the SAME step's final hidden state (self-speculative,
    greedily accepted with no rejection sampling against the main path -- see CLAUDE.md's MTP
    invariant). 0 disables it; it is a no-op on a checkpoint with no MTP head.

    ``converge_tol`` turns on the parameter-free depth policy: stop looping once the last position's
    readout stops moving (see ``TinyMoETransformer._convergence_exit``). It **forces the KV cache
    off** -- an exited loop appends no K/V for that token, so a later full-depth step would attend
    over a cache with a hole in it. Whether that trade is worth it depends on the checkpoint;
    ``scripts/eval_calibration.py`` prints the per-transition agreement/log-prob-gap numbers that
    say what threshold to use and what stopping early costs.

    Text is streamed by re-decoding the full generated id sequence each step and yielding only the
    new suffix, rather than decoding each step's tokens in isolation -- a lone step's tokens can
    decode differently out of context (subword/space merges), so this avoids visible artifacts.
    """
    max_len = ModelConfig.Params["max_seq_len"]
    prompt_ids = prompt_ids[-(max_len - 1):]  # always leave room for at least one generated token
    prompt_len = len(prompt_ids)
    all_ids = torch.tensor([prompt_ids], device=device)

    use_mtp = num_mtp_tokens > 0 and model.has_mtp
    # the two are mutually exclusive at the model level; resolve it here rather than letting the
    # assertion fire deep inside the loop
    if converge_tol is not None and use_kv_cache:
        use_kv_cache = False
    kv_cache = KVCache.for_model(model, n_loops=n_loops) if use_kv_cache else None

    def forward_logits(ids: torch.Tensor):
        kwargs = {"n_loops": n_loops}
        if converge_tol is not None:
            kwargs["converge_tol"] = converge_tol
            kwargs["min_loops"] = min_loops
        if kv_cache is not None:
            kwargs["kv_cache"] = kv_cache
        if use_mtp:
            x_all, extra = model(ids, return_hidden=True, **kwargs)
            logits = model.lm_head(x_all[-1])
            n_extra = min(num_mtp_tokens, model.mtp_head.num_extra_tokens)
            mtp_logits = [model.mtp_head.lm_head(extra[:, -1:, i, :]) for i in range(n_extra)]
            return logits, mtp_logits
        out = model(ids, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        return logits, []

    generated = 0
    prev_text = ""
    step_ids = all_ids
    while generated < max_new_tokens and all_ids.shape[1] < max_len:
        logits, mtp_logits = forward_logits(step_ids)
        next_logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0:
            next_logits = _apply_repetition_penalty(next_logits, all_ids, repetition_penalty)
        if no_repeat_ngram_size > 0:
            banned = _banned_ngram_tokens(all_ids[0].tolist(), no_repeat_ngram_size)
            if banned:
                next_logits[0, banned] = float("-inf")

        next_token = int(_sample_next_token(next_logits, temperature, top_k, top_p).item())
        new_tokens = [next_token]
        eos_hit = next_token == tokenizer.eos_token_id

        for draft_logits in ([] if eos_hit else mtp_logits):
            draft_token = int(draft_logits[:, -1, :].argmax(dim=-1).item())
            new_tokens.append(draft_token)
            if draft_token == tokenizer.eos_token_id:
                eos_hit = True
                break

        room = max_new_tokens - generated
        if len(new_tokens) > room:
            new_tokens = new_tokens[:room]
        generated += len(new_tokens)

        new_tok_tensor = torch.tensor([new_tokens], device=device)
        all_ids = torch.cat([all_ids, new_tok_tensor], dim=1)

        full_text = tokenizer.decode(all_ids[0, prompt_len:], skip_special_tokens=True)
        if full_text != prev_text:
            yield full_text[len(prev_text):]
            prev_text = full_text

        if eos_hit or generated >= max_new_tokens:
            break
        step_ids = new_tok_tensor if kv_cache is not None else all_ids[:, -max_len:]


def generate(
    model: TinyMoETransformer,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    **kwargs,
) -> str:
    """One-shot wrapper around `stream_generate` -- returns the full generated text."""
    return "".join(stream_generate(
        model, tokenizer, prompt_ids, max_new_tokens, temperature, top_k, device,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, **kwargs,
    ))


def build_prompt_ids(tokenizer, prompt: str, args) -> list[int]:
    if args.chat:
        template = ChatTemplate(tokenizer)
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        return template.encode_prompt(messages)
    return prepend_bos(tokenizer, tokenizer.encode(prompt))


def prepend_bos(tokenizer, ids: list[int]) -> list[int]:
    """Prepend BOS if it isn't already there -- mirrors modules/data/dataset.py's packing rule,
    where every training document is BOS-prefixed. This tokenizer has ``add_bos_token: false``, so
    plain ``tokenizer.encode(...)`` never adds it: without this, a raw completion prompt starts
    mid-document from the model's point of view, which it never saw once in training."""
    bos_id = tokenizer.bos_token_id
    if bos_id is not None and (not ids or ids[0] != bos_id):
        return [bos_id] + ids
    return ids


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
        prompt_ids = build_prompt_ids(tokenizer, prompt, args)
        for chunk in stream_generate(
            model, tokenizer, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, device,
            repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_p=args.top_p, n_loops=args.n_loops, num_mtp_tokens=args.num_mtp_tokens,
            use_kv_cache=not args.no_kv_cache, converge_tol=args.converge_tol,
            min_loops=args.min_loops,
        ):
            print(chunk, end="", flush=True)
        print("\n")



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
        default=TOKENIZER_DIR,
        help="Path to the tokenizer directory (default: utils.TOKENIZER_DIR)",
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
        "--top-p",
        type=float,
        default=0.0,
        help="Nucleus sampling threshold in (0, 1); 0 or 1 = disabled (default: 0, disabled)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalize tokens already generated; 1.0 = disabled, >1.0 discourages repeats (default: 1.0)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Block repeating any n-gram of this size; 0 = disabled (default: 0)",
    )
    parser.add_argument(
        "--n-loops",
        type=int,
        default=None,
        help="Override the model's configured MoE loop count for this run (default: use the "
             "checkpoint's configured n_loops).",
    )
    parser.add_argument(
        "--num-mtp-tokens",
        type=int,
        default=0,
        help="Draft up to this many extra tokens per step from the MTP head (self-speculative, "
             "unverified -- trades a little quality for fewer forward passes). 0 disables it; "
             "capped at the checkpoint's mtp_num_extra_tokens. Default: 0.",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable the KV cache and re-run the full prefix every step (slow reference path).",
    )
    parser.add_argument(
        "--converge-tol",
        type=float,
        default=None,
        help="Stop looping once the last position's readout stops moving: the top-1 token is "
             "unchanged AND its log-probability moved less than this between consecutive loops. "
             "Parameter-free, no loss term. Forces the KV cache off (an exited loop stores no K/V "
             "for that token). Run scripts/eval_calibration.py to pick a value. Default: off.",
    )
    parser.add_argument(
        "--min-loops",
        type=int,
        default=1,
        help="Floor on the loop count when --converge-tol is set (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Wrap the prompt in the SFT chat template (<｜User｜>...<｜Assistant｜>). "
             "Use for SFT checkpoints; leave off for base/pretrained checkpoints.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt, only used with --chat.",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer} …")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading checkpoint from {args.checkpoint} …")
    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {args.device} ({sum(p.numel() for p in model.parameters()):,} params)\n")

    if args.prompt is not None:
        prompt_ids = build_prompt_ids(tokenizer, args.prompt, args)
        for chunk in stream_generate(
            model, tokenizer, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, args.device,
            repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
            top_p=args.top_p, n_loops=args.n_loops, num_mtp_tokens=args.num_mtp_tokens,
            use_kv_cache=not args.no_kv_cache, converge_tol=args.converge_tol,
            min_loops=args.min_loops,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        interactive_loop(model, tokenizer, args, args.device)


if __name__ == "__main__":
    main()
