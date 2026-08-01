"""End-to-end regression test for the cu_seqlens/accelerate NaN.

Runs the REAL training pipeline in miniature: real Dataset over a fineweb shard -> DataLoader with
workers -> accelerate.prepare(split_batches=True, grad accumulation) -> real TinyMoETransformer ->
compute_mtp_loss -> accelerator.backward/step. cu_seqlens is derived in-thread from document_ids,
exactly as scripts/pretrain.py does. Loss must stay finite past the first optimizer update (where the
truncated-cu_seqlens bug surfaced as NaN). GPU required; BF16 like real training.
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss

assert torch.cuda.is_available(), "GPU required"
torch.manual_seed(0)
dev = "cuda"

tok = AutoTokenizer.from_pretrained("ckpts/pretrained/DeepSeek-V4-Pro-tokenizer")
shard_root = "data/datasets/parquet/fineweb/CC-MAIN-2021-49"
shard = sorted(os.listdir(shard_root))[0]
d = tempfile.mkdtemp(); root = os.path.join(d, "root"); os.makedirs(root)
os.symlink(os.path.abspath(os.path.join(shard_root, shard)), os.path.join(root, shard))
cfg = os.path.join(d, "cfg.json")
with open(cfg, "w") as f:
    json.dump({"pretrain": [{"root": root, "column": "content"}]}, f)

B, S = 2, 1024
from modules.data.dataset import Dataset
ds = Dataset(tok, batch_size=B, max_length=S, mode="pretrain", config_path=cfg, num_mtp_tokens=2)
dl = DataLoader(ds, batch_size=None, num_workers=2, prefetch_factor=2)

# small model, but full architecture (MoE loops, attention experts, MTP, PLE)
P = dict(vocab_size=len(tok), max_seq_len=S, hidden_size=256, intermediate_size=512, head_dim=32,
         num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1, top_k=2, n_loops=2,
         num_ir_experts=1, num_ir_entries=256, ir_dim=64, dropout=0.0, ple_embeddings_size=32,
         mtp_num_extra_tokens=2)
model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
model.set_checkpointing(False, False)
model.delayed_mtp_loss(True)
model._token_tracker.pad_token_id = tok.pad_token_id
opt = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.02)

acc = Accelerator(device_placement=True, split_batches=True, gradient_accumulation_steps=8)
model, opt, dl = acc.prepare(model, opt, dl)
unwrapped = acc.unwrap_model(model)

N_STEPS = 24  # > grad_accumulation_steps so several real optimizer updates happen
seen = 0
for step, batch in enumerate(dl):
    input_ids = batch["input_ids"].to(dev)
    document_ids = batch["document_ids"].to(dev)
    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)  # in-thread, like pretrain.py
    labels = batch["labels"].to(dev)
    pad_mask = input_ids == tok.pad_token_id

    with acc.accumulate(model):
        logits, aux_loss, mtp = model(input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                                      identity_skew=0.8, return_aux_loss=True, return_hidden=True)
        loss, loss_ce = compute_mtp_loss(logits, labels, mtp_outputs=mtp,
                                         lm_head=unwrapped.mtp_head.lm_head, lambda_mtp=0.1,
                                         main_lm_head=unwrapped.lm_head, pad_mask=pad_mask)
        loss = loss + 0.01 * aux_loss
        acc.backward(loss)
        if acc.sync_gradients:
            acc.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    v = loss.item()
    assert torch.isfinite(loss), f"non-finite loss at step {step}: {v}"
    if step % 4 == 0:
        print(f"step {step}: loss={v:.4f} ce={loss_ce.item():.4f} aux={aux_loss.item():.4f} "
              f"segments={cu_seqlens.numel() - 1}")
    seen += 1
    if step >= N_STEPS:
        break

assert seen > 16, f"expected to run >16 steps, only ran {seen}"
print(f"\nTRAIN SMOKE PASSED: {seen} steps, loss finite past the first optimizer updates "
      "(cu_seqlens derived in-thread -> no accelerate truncation)")
