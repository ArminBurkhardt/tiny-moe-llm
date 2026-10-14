"""Where the IR read's signal dies: table content, the attenuation chain, and gradient reach.

Answers the one question the sharpening arms left open. The read ablates to 0.0002 nats on both
arms; this says whether that is because the table holds nothing (values never moved) or because
everything downstream of it discards what it holds (values moved, contribution attenuated).

The chain, per loop, all as RMS over tokens:

    x_norm -> down_proj -> [read] retrieved_y -> g_proj -> up_proj -> information
           -> attn(Q=x_norm, K/V=information) -> expert out -> * router weight -> * loop_scale

Read-only: forward hooks on modules the model already runs.
"""
import os
import sys
import argparse

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.model.experts import InformationRetrievalExpert
from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids
from config import ModelConfig
from utils import TOKENIZER_DIR, logger
from scripts.eval_calibration import load_model


def rms(t):
    return t.detach().float().pow(2).mean().sqrt().item()


def table_stats(path):
    """state dict only: did the value table leave its zero init, and what did the keys do."""
    sd = torch.load(path, map_location="cpu")["model_state_dict"]
    out = {}
    for k, v in sd.items():
        if k.endswith("ir_module.y_values"):
            f = v.float()
            row = f.norm(dim=-1)
            out["y_rms"] = f.pow(2).mean().sqrt().item()
            out["y_row_norm_mean"] = row.mean().item()
            out["y_row_norm_p05"] = row.quantile(0.05).item()
            out["y_row_norm_p95"] = row.quantile(0.95).item()
            out["y_rows_near_zero"] = (row < 0.01 * row.mean()).float().mean().item()
        if k.endswith("ir_module.z_keys"):
            out["z_rms"] = v.float().pow(2).mean().sqrt().item()
        if k.endswith("ir_module.log_temperature"):
            out["temperature"] = v.float().exp().item()
        if k.endswith("ir_module.temperature_scale"):
            out["temp_scale"] = v.float().item()
        if k.endswith("ir_module.g_proj.weight"):
            out["g_proj_rms"] = v.float().pow(2).mean().sqrt().item()
        if k.endswith("2.up_proj.weight"):
            out["up_proj_rms"] = v.float().pow(2).mean().sqrt().item()
    return out


class Chain:
    """capture the per-loop norms along the IR pathway, plus the read-zeroed counterfactual."""

    def __init__(self, model):
        moe = model.moe
        idx = [i for i, e in enumerate(moe.experts) if isinstance(e, InformationRetrievalExpert)]
        assert len(idx) == 1
        self.expert = moe.experts[idx[0]]
        self.ir = self.expert.ir_module
        self.rec = {}
        self.handles = []
        self._tap("x_norm", self.expert.norm)
        self._tap("down", self.expert.down_proj)
        self._tap("ir_out", self.ir.g_proj)
        self._tap("information", self.expert.up_proj)
        self._tap("expert_out", self.expert.attn)
        # g_proj's INPUT is the retrieved value itself -- the thing the table actually returns
        self.handles.append(self.ir.g_proj.register_forward_hook(
            lambda m, i, o: self.rec.setdefault("retrieved_y", []).append(rms(i[0]))))

    def _tap(self, name, module):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            self.rec.setdefault(name, []).append(rms(t))
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def attenuation(model, batch, device):
    """one forward, then the same forward with the retrieved value forced to zero."""
    ids = batch["input_ids"].to(device)
    doc = batch["document_ids"].to(device)
    cu, _ = cu_seqlens_from_doc_ids(doc)

    ch = Chain(model)
    model(ids, cu_seqlens=cu, max_seqlen=ids.shape[1], skip_mtp=True)
    live = {k: list(v) for k, v in ch.rec.items()}
    ch.close()

    # counterfactual: zero the read on its way out of the module, leaving everything else identical.
    # what survives is g_proj(0) = 0 (no bias), so the delta IS the read's whole contribution
    ir = ch.ir
    zero_h = ir.g_proj.register_forward_pre_hook(lambda m, i: (torch.zeros_like(i[0]),))
    ch2 = Chain(model)
    model(ids, cu_seqlens=cu, max_seqlen=ids.shape[1], skip_mtp=True)
    dead = {k: list(v) for k, v in ch2.rec.items()}
    ch2.close()
    zero_h.remove()
    return live, dead


def grad_reach(model, batch, device):
    """one backward: relative gradient size on the table vs the trunk it has to compete with."""
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    doc = batch["document_ids"].to(device)
    cu, _ = cu_seqlens_from_doc_ids(doc)

    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(True)
    hidden = model(ids, cu_seqlens=cu, max_seqlen=ids.shape[1], return_hidden=True, skip_mtp=True)
    h = hidden[-1][:, :-1]
    tgt = labels[:, 1:]
    # a slice, not the whole batch: this is a gradient RATIO, and the full vocab projection over
    # 16k positions is a 2GB fp32 transient for no extra information
    h = h.reshape(-1, h.shape[-1])[:2048]
    tgt = tgt.reshape(-1)[:2048]
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.float(), tgt, ignore_index=-100)
    loss.backward()

    named = dict(model.named_parameters())
    rows = []
    watch = [k for k in named if ".ir_module." in k or
             (k.startswith("moe.experts.2.") and ("down_proj" in k or "up_proj" in k))]
    trunk = ["moe.shared_mlp.gate_proj.weight", "moe.experts.0.attn.q_proj.weight",
             "model.layers.0.mlp.down_proj.weight", "lm_head.proj_out.weight"]
    for k in watch + [t for t in trunk if t in named]:
        p = named[k]
        if p.grad is None:
            rows.append((k, float("nan"), float("nan"), float("nan")))
            continue
        g = p.grad.detach().float().norm().item()
        w = p.detach().float().norm().item()
        rows.append((k, g, w, g / w if w > 0 else float("nan")))
    return loss.item(), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--phase", default="ir_val")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model, _ = load_model(args.checkpoint, args.device)

    print(f"\n########## {args.label or os.path.basename(args.checkpoint)} ##########")
    print("\n=== 1. Table content (state dict) ===")
    for k, v in table_stats(args.checkpoint).items():
        print(f"  {k:>18}: {v:.6f}")

    ds = Dataset(data_dir=args.data_dir, tokenizer=tok, batch_size=args.batch_size,
                 max_length=ModelConfig.Params["max_seq_len"], split=args.phase,
                 num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"], start_doc_idx=0)
    batch = next(iter(ds))

    live, dead = attenuation(model, batch, args.device)
    n = len(live["x_norm"])
    print("\n=== 2. Attenuation chain (RMS over tokens, per loop) ===")
    order = ["x_norm", "down", "retrieved_y", "ir_out", "information", "expert_out"]
    print("  stage           " + "".join(f"  loop {i+1:<8}" for i in range(n)))
    for name in order:
        vals = live.get(name, [])
        print(f"  {name:<14}" + "".join(f"  {v:>12.6f}" for v in vals))
    print("  -- same forward with the retrieved value forced to zero --")
    for name in ["ir_out", "information", "expert_out"]:
        vals = dead.get(name, [])
        print(f"  {name:<14}" + "".join(f"  {v:>12.6f}" for v in vals))
    print("  -- the read's share of the expert's output --")
    share = [abs(live["expert_out"][i] - dead["expert_out"][i]) / max(live["expert_out"][i], 1e-9)
             for i in range(n)]
    print("  rel delta     " + "".join(f"  {v:>12.6f}" for v in share))

    ls = model.moe.loop_scale.detach().float().tolist()
    print(f"\n  loop_scale: {[round(v, 4) for v in ls]}")

    loss, rows = grad_reach(model, batch, args.device)
    print(f"\n=== 3. Gradient reach (one backward, CE={loss:.4f}) ===")
    print(f"  {'parameter':<46}{'||grad||':>12}{'||w||':>12}{'ratio':>12}")
    for k, g, w, r in rows:
        print(f"  {k:<46}{g:>12.3e}{w:>12.3e}{r:>12.3e}")


if __name__ == "__main__":
    main()
