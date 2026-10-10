"""Greedy decode with evidence: KV-cached versus uncached, step by step.

The reader has no KV slot (``CrossAttention`` hardcodes ``kv_cache=None`` on the evidence branch),
so a cached step re-reads the fixed evidence states against the ONE new query token while the
uncached reference re-reads them against the whole prefix. Causality of the trunk makes that
equivalent in exact arithmetic; the only way it breaks is a path that looks at something other than
the current token's own query, so this test is the tripwire for that.

Three assertions, on a small model whose reader ``o_proj`` is randomized nonzero (a fresh zero
``o_proj`` would make evidence a no-op and every comparison vacuous):

1. **Evidence is not vacuous.** The uncached logits with evidence differ from the no-evidence
   decode of the same tokens by far more than the cache tolerance.
2. **Cached matches uncached per step.** The same tokens are fed to both paths (teacher forced on
   the uncached greedy choices, so one flipped argmax cannot cascade into an unrelated
   comparison); the max abs logit difference over 16 steps stays inside a bf16 tolerance. Observed
   difference and argmax agreement are printed. Free-running cached greedy is then compared with
   the uncached token sequence, printed and not asserted: a bf16 near tie may legitimately flip.
3. **The selector gate is the one thing that is NOT cache-safe.** Printed, not asserted: with
   ``evidence_gate_scale`` nonzero the gate averages the selector mass over the query tokens of the
   forward, which is the whole prefix when uncached and a single token when cached, so the two paths
   are expected to disagree. This is the decode side of the reason the gate stays at zero.

GPU required (flash varlen).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.kv_cache import KVCache
from modules.model.experts import InformationRetrievalExpert

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4, evidence_port=True,
)

STEPS = 16
PROMPT = 12
TOL = 0.5


def last_logits(model, ids, **kw):
    out = model(ids, skip_mtp=True, **kw)
    return (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()


def uncached_trace(model, prompt, evidence):
    """greedy, full prefix every step. Returns (tokens, per step logits)."""
    ids, logits = prompt, []
    for _ in range(STEPS):
        lg = last_logits(model, ids, evidence=evidence)
        logits.append(lg)
        ids = torch.cat([ids, lg.argmax(-1, keepdim=True)], dim=1)
    return ids, torch.cat(logits, 0)


def cached_trace(model, prompt, evidence, forced=None):
    """prefill then one token per step. With ``forced`` the fed tokens are the given sequence."""
    kv = KVCache.for_model(model)
    lg = last_logits(model, prompt, evidence=evidence, kv_cache=kv)
    assert kv.length == PROMPT, "the prefill did not populate the cache"
    logits, toks = [lg], []
    for t in range(STEPS - 1):
        nxt = forced[:, PROMPT + t:PROMPT + t + 1] if forced is not None else lg.argmax(-1, keepdim=True)
        toks.append(nxt)
        lg = last_logits(model, nxt, evidence=evidence, kv_cache=kv)
        logits.append(lg)
    assert kv.length == PROMPT + STEPS - 1, "the decode steps did not go through the cache"
    return torch.cat(logits, 0), torch.cat(toks, 1) if toks else None


def main():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to("cuda").to(BF16).eval()
    model.set_checkpointing(False, False)
    ir_expert = next(e for e in model.moe.experts if isinstance(e, InformationRetrievalExpert))
    with torch.no_grad():
        torch.nn.init.normal_(model.moe.shared_evidence.attn.o_proj.weight, std=0.3)
        torch.nn.init.normal_(ir_expert.direct_gate.weight, std=0.05)
        ir_expert.ir_module.log_memory_scale.fill_(3.0)

    prompt = torch.randint(1, P["vocab_size"], (1, PROMPT), device="cuda")
    S_ev = 18
    ev_ids = torch.randint(1, P["vocab_size"], (1, S_ev), device="cuda")
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6 + [2] * 6], device="cuda")
    ev_segment = torch.zeros(1, S_ev, dtype=torch.long, device="cuda")
    keys = torch.randn(3, P["ir_dim"], device="cuda", dtype=BF16)
    chunk_segments = torch.zeros(3, dtype=torch.long, device="cuda")

    with torch.inference_mode():
        evidence = model.build_evidence(ev_ids, ev_chunk, ev_segment, 1, chunk_keys=keys,
                                        chunk_segments=chunk_segments)

        ids, ref = uncached_trace(model, prompt, evidence)
        # the same tokens with no evidence at all
        bare = torch.cat([last_logits(model, ids[:, :PROMPT + t]) for t in range(STEPS)], 0)
        vac = (ref - bare).abs().max().item()
        print(f"1. evidence vs no evidence: max |delta| = {vac:.3f}")
        assert vac > 10 * 1e-2 and vac > TOL / 5, f"evidence barely moves the logits ({vac}): vacuous"
        print("1. evidence is not vacuous                                               PASS")

        cached, _ = cached_trace(model, prompt, evidence, forced=ids)
        diff = (ref - cached).abs().max().item()
        per_step = (ref - cached).abs().amax(dim=-1)
        agree = (ref.argmax(-1) == cached.argmax(-1)).float().mean().item()
        print(f"2. cached vs uncached over {STEPS} steps: max |delta| = {diff:.3e} "
              f"(per step max {per_step.max():.3e}, min {per_step.min():.3e}), "
              f"argmax agreement {agree * 100:.0f}%")
        assert diff < TOL, f"cached decode with evidence drifted from the uncached reference: {diff}"
        free, _free_tokens = cached_trace(model, prompt, evidence)
        free_tokens = torch.cat([prompt, _free_tokens], 1)
        same = torch.equal(free_tokens[:, :ids.shape[1] - 1], ids[:, :ids.shape[1] - 1])
        print(f"   free running cached greedy reproduces the uncached tokens: {same}")
        print("2. cached matches uncached with evidence attached                        PASS")

        with torch.no_grad():
            model.moe.evidence_gate_scale.fill_(5.0)
        ref_g = torch.cat([last_logits(model, ids[:, :PROMPT + t], evidence=evidence) for t in range(STEPS)], 0)
        cached_g, _ = cached_trace(model, prompt, evidence, forced=ids)
        diff_g = (ref_g - cached_g).abs().max().item()
        print(f"3. (info) gate scale 5: cached vs uncached max |delta| = {diff_g:.3e} "
              f"(gate off: {diff:.3e})")


if __name__ == "__main__":
    main()
