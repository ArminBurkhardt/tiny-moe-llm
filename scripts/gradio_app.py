"""Gradio UI for TinyMoE inference -- every generation/model knob adjustable, KV-cached decoding.

Not part of the training pipeline; a standalone entry point like scripts/inference.py, reusing its
`stream_generate` (KV cache + self-speculative MTP drafting) so the CLI and this UI never drift.

Run: python scripts/gradio_app.py [--host 0.0.0.0] [--port 7860] [--share]
"""
import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import gradio as gr
from transformers import AutoTokenizer

from scripts.inference import load_model, stream_generate, prepend_bos
from modules.data.chat import ChatTemplate
from config import ModelConfig
from utils import BASE_DIR, TOKENIZER_DIR

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "trained")

# Short and imperative, deliberately -- this mirrors the register of SQUAD_INSTRUCTION
# (scripts/prepare_sft_data.py), the one instruction-style prompt the SFT mix actually supervises
# at scale (squad_v2 is 20% of the token budget). A long, multi-clause system prompt is off the
# training distribution for a 330M model and tends to get partially ignored rather than followed.
# Doesn't tell the model to hedge/abstain -- it already over-abstains on answerable questions (see
# CLAUDE.md/project memory on the Step 12 SFT abstention collapse), so this only asks for honesty
# about not knowing, not for caution as a default stance.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly and concisely. "
    "If you don't know, say so plainly instead of guessing."
)

# module-level cache: this app is a single local GPU serving one user at a time (same usage
# pattern as inference.py's interactive CLI), so a global "currently loaded" slot -- reloaded only
# when the requested checkpoint/tokenizer/device actually changes -- is the right lifetime for a
# multi-GB model. Per-session state would mean reloading it on every browser tab.
_STATE = {"model": None, "tokenizer": None, "checkpoint": None, "tokenizer_dir": None, "device": None}


def discover_checkpoints() -> list[str]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    paths = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("checkpoint") and f.endswith(".pt")
    ]
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths


def _ensure_loaded(checkpoint_path: str, tokenizer_dir: str, device: str):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise gr.Error(f"Checkpoint not found: {checkpoint_path!r}")
    changed = (
        _STATE["checkpoint"] != checkpoint_path
        or _STATE["tokenizer_dir"] != tokenizer_dir
        or _STATE["device"] != device
    )
    if changed:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = load_model(checkpoint_path, device)
        _STATE.update(
            model=model, tokenizer=tokenizer, checkpoint=checkpoint_path,
            tokenizer_dir=tokenizer_dir, device=device,
        )
    return _STATE["model"], _STATE["tokenizer"]


def load_and_report(checkpoint_path: str, tokenizer_dir: str, device: str) -> str:
    model, _ = _ensure_loaded(checkpoint_path, tokenizer_dir, device)
    total = sum(p.numel() for p in model.parameters())
    mtp = f"{model.mtp_head.num_extra_tokens} extra tokens" if model.has_mtp else "disabled"
    return (
        f"**Loaded** `{os.path.basename(checkpoint_path)}` on `{device}` -- {total / 1e6:.1f}M params. "
        f"n_loops={model.moe.n_loops}, MTP={mtp}."
    )


def _seed(seed: int):
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))


def run_generate(
    prompt, checkpoint_path, tokenizer_dir, device,
    max_new_tokens, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size,
    n_loops, num_mtp_tokens, use_kv_cache, converge_tol, min_loops, seed,
):
    if not prompt or not prompt.strip():
        yield ""
        return
    model, tokenizer = _ensure_loaded(checkpoint_path, tokenizer_dir, device)
    _seed(seed)
    prompt_ids = prepend_bos(tokenizer, tokenizer.encode(prompt))
    text = ""
    for chunk in stream_generate(
        model, tokenizer, prompt_ids, int(max_new_tokens), float(temperature), int(top_k), device,
        repetition_penalty=float(repetition_penalty), no_repeat_ngram_size=int(no_repeat_ngram_size),
        top_p=float(top_p), n_loops=(None if int(n_loops) == 0 else int(n_loops)),
        num_mtp_tokens=int(num_mtp_tokens), use_kv_cache=bool(use_kv_cache),
        converge_tol=(None if float(converge_tol) <= 0 else float(converge_tol)),
        min_loops=int(min_loops),
    ):
        text += chunk
        yield text


def run_chat(
    message, history, system_prompt, checkpoint_path, tokenizer_dir, device,
    max_new_tokens, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size,
    n_loops, num_mtp_tokens, use_kv_cache, converge_tol, min_loops, seed,
):
    model, tokenizer = _ensure_loaded(checkpoint_path, tokenizer_dir, device)
    _seed(seed)
    template = ChatTemplate(tokenizer)
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for turn in history:
        role = turn.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})
    prompt_ids = template.encode_prompt(messages)

    partial = ""
    for chunk in stream_generate(
        model, tokenizer, prompt_ids, int(max_new_tokens), float(temperature), int(top_k), device,
        repetition_penalty=float(repetition_penalty), no_repeat_ngram_size=int(no_repeat_ngram_size),
        top_p=float(top_p), n_loops=(None if int(n_loops) == 0 else int(n_loops)),
        num_mtp_tokens=int(num_mtp_tokens), use_kv_cache=bool(use_kv_cache),
        converge_tol=(None if float(converge_tol) <= 0 else float(converge_tol)),
        min_loops=int(min_loops),
    ):
        partial += chunk
        yield partial


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="TinyMoE Inference") as demo:
        gr.Markdown("# TinyMoE Inference")

        with gr.Accordion("Model", open=True):
            with gr.Row():
                checkpoint_dd = gr.Dropdown(
                    choices=discover_checkpoints(),
                    value=(discover_checkpoints() or [None])[0],
                    label="Checkpoint (.pt)",
                    allow_custom_value=True,
                    scale=4,
                )
                refresh_btn = gr.Button("Refresh list", scale=1)
            with gr.Row():
                tokenizer_tb = gr.Textbox(value=TOKENIZER_DIR, label="Tokenizer directory", scale=3)
                device_dd = gr.Dropdown(choices=["cuda", "cpu"], value=DEFAULT_DEVICE, label="Device", scale=1)
            load_btn = gr.Button("Load / reload model", variant="primary")
            status_md = gr.Markdown(
                "No model loaded yet -- it will load automatically on first generation, or click above."
            )

        with gr.Accordion("Generation settings", open=True):
            with gr.Row():
                max_new_tokens = gr.Slider(1, 4096, value=200, step=1, label="Max new tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature (0 = greedy)")
            with gr.Row():
                top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = disabled)")
                top_p = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Top-p (0 or 1 = disabled)")
            with gr.Row():
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, step=0.01, label="Repetition penalty")
                no_repeat_ngram = gr.Slider(0, 8, value=0, step=1, label="No-repeat n-gram size (0 = disabled)")
            with gr.Row():
                n_loops = gr.Slider(
                    0, max(8, ModelConfig.Params["n_loops"] * 2), value=0, step=1,
                    label=f"MoE loop count override (0 = checkpoint default: {ModelConfig.Params['n_loops']})",
                )
                num_mtp_tokens = gr.Slider(
                    0, max(1, ModelConfig.Params["mtp_num_extra_tokens"]), value=0, step=1,
                    label="Self-speculative MTP draft tokens/step (0 = off, greedy & unverified)",
                )
            with gr.Row():
                # the parameter-free depth policy that replaced the halt head. >0 forces the KV
                # cache off, since an exited loop stores no K/V for that token.
                converge_tol = gr.Slider(
                    0.0, 2.0, value=0.0, step=0.01,
                    label="Convergence exit tolerance (0 = off, forces KV cache off)",
                )
                min_loops = gr.Slider(1, 8, value=1, step=1, label="Minimum loops before exiting")
            with gr.Row():
                use_kv_cache = gr.Checkbox(value=True, label="Use KV cache")
                seed = gr.Number(value=-1, precision=0, label="Random seed (-1 = random)")

        gen_inputs_tail = [
            max_new_tokens, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram,
            n_loops, num_mtp_tokens, use_kv_cache, converge_tol, min_loops, seed,
        ]

        with gr.Tabs():
            with gr.Tab("Chat (SFT checkpoints)"):
                system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=2)
                gr.ChatInterface(
                    fn=run_chat,
                    additional_inputs=[system_prompt, checkpoint_dd, tokenizer_tb, device_dd] + gen_inputs_tail,
                )
            with gr.Tab("Raw completion (pretrained checkpoints)"):
                prompt_tb = gr.Textbox(label="Prompt", lines=6, placeholder="Once upon a time,")
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary")
                    stop_btn = gr.Button("Stop")
                output_tb = gr.Textbox(label="Output", lines=14, interactive=False, buttons=["copy"])
                gen_event = gen_btn.click(
                    run_generate,
                    inputs=[prompt_tb, checkpoint_dd, tokenizer_tb, device_dd] + gen_inputs_tail,
                    outputs=output_tb,
                )
                stop_btn.click(None, cancels=[gen_event])

        refresh_btn.click(lambda: gr.Dropdown(choices=discover_checkpoints()), outputs=checkpoint_dd)
        load_btn.click(load_and_report, inputs=[checkpoint_dd, tokenizer_tb, device_dd], outputs=status_md)

    return demo


def main():
    parser = argparse.ArgumentParser(description="TinyMoE Gradio inference UI")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
