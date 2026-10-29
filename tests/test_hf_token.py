"""get_hf_token resolution order and TOKENIZER_DIR env override. No GPU, no TE."""
import os, sys, tempfile, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

# 1. $HF_TOKEN wins over everything else
os.environ["HF_TOKEN"] = "  env_token  "
assert utils.get_hf_token() == "env_token", "HF_TOKEN must win and be stripped"
print("[ok] $HF_TOKEN takes precedence and is stripped")

# 2. with HF_TOKEN unset, the repo-root huggingface.key is read
del os.environ["HF_TOKEN"]
key_path = os.path.join(utils.BASE_DIR, "huggingface.key")
backup = None
if os.path.isfile(key_path):
    with open(key_path) as f:
        backup = f.read()
try:
    with open(key_path, "w") as f:
        f.write("file_token\n")
    assert utils.get_hf_token() == "file_token", "huggingface.key must be the second source"
    print("[ok] huggingface.key is read when HF_TOKEN is unset")

    # 3. an empty key file must not shadow the cache/None fallback
    with open(key_path, "w") as f:
        f.write("\n")
    assert utils.get_hf_token() != "", "an empty huggingface.key must not resolve to empty string"
    print("[ok] empty huggingface.key falls through instead of returning ''")
finally:
    if backup is None:
        os.remove(key_path)
    else:
        with open(key_path, "w") as f:
            f.write(backup)

# 4. TOKENIZER_DIR honours the env override at import time
os.environ["TINY_LLM_TOKENIZER"] = os.path.join(tempfile.mkdtemp(), "tok")
importlib.reload(utils)
assert utils.TOKENIZER_DIR == os.environ["TINY_LLM_TOKENIZER"], utils.TOKENIZER_DIR
print("[ok] TINY_LLM_TOKENIZER overrides TOKENIZER_DIR")

del os.environ["TINY_LLM_TOKENIZER"]
importlib.reload(utils)
assert utils.TOKENIZER_DIR.endswith("DeepSeek-V4-Pro-tokenizer-65536"), utils.TOKENIZER_DIR
assert utils.TOKENIZER_REPO == "ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536"
assert utils.HF_UPLOAD_REPO == "ikeafisch4/temp-train"
print("[ok] defaults point at the pruned tokenizer and the right repos")

print("\nHF TOKEN / TOKENIZER CONSTANT CHECKS PASSED")
