#!/bin/bash
# TINY_LLM_ROOT/TINY_LLM_ENV_INIT let this run unchanged on WSL (defaults match the local dev box)
# and on the rented box (e.g. `TINY_LLM_ROOT=/workspace/tiny-llm TINY_LLM_ENV_INIT=vast_init`).
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}"
source "${TINY_LLM_ENV_INIT:-env_init}"
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn MISSING:", e)
try:
    import transformer_engine.pytorch as te
    print("transformer_engine ok")
except Exception as e:
    print("TE MISSING:", e)
from transformers import AutoTokenizer
# read the shared constant instead of a literal path, so this check cannot drift away from what
# the trainer actually loads (it used to point at the *unpruned* tokenizer, which never gets
# downloaded onto the box at all)
import sys; sys.path.insert(0, ".")
from utils import TOKENIZER_DIR
tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
print("tokenizer vocab_size:", tok.vocab_size, "len:", len(tok))
print("pad:", tok.pad_token, tok.pad_token_id, "bos:", tok.bos_token, tok.bos_token_id, "eos:", tok.eos_token_id)
EOF
