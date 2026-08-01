#!/bin/bash
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm
source env_init
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
tok = AutoTokenizer.from_pretrained("ckpts/pretrained/DeepSeek-V4-Pro-tokenizer")
print("tokenizer vocab_size:", tok.vocab_size, "len:", len(tok))
print("pad:", tok.pad_token, tok.pad_token_id, "bos:", tok.bos_token, tok.bos_token_id, "eos:", tok.eos_token_id)
EOF
