"""Offline probe: can a small head on the frozen hidden state retrieve from the store?

The selector scores a chunk with ``normalize(down_proj(norm(h)) + loop_query_bias)`` against the
adapted bge key. This script reads, at each question prompt's last token with ``evidence=None``,
the input of the IR expert's ``down_proj`` (the normed hidden state ``H`` the selector's query is
built from) and the model's own unit query, then asks how much of bge's retrieval a head fitted on
``H`` alone recovers. Each head regresses ``H`` onto bge's question vector and retrieves against
the raw bge keys, so a high score says the trunk's state already carries the question and only the
head is missing; a low one says the state does not carry it and a separate tower is needed.

Predictors, every one scored on held-out folds only:

* ``model``: the checkpoint's own query against ``normalize(key_adapter(keys))``, as
  ``eval_store.py recall`` reports it. Its cosine is to ``normalize(key_adapter(bge question))``.
* ``ridge_<alpha>``: closed-form ridge from centred ``H`` to the bge question vector.
* ``mlp``: two layers (GELU) on standardized ``H``, cosine loss, AdamW.
* ``mean``: the training folds' mean bge question vector, the floor that reads the cosine
  (bge question vectors share a large common direction, so a cosine near this row's carries nothing).
* ``bge``: bge's own question vector, the ceiling.

```bash
TINY_LLM_CONFIG=ckpts/inject/config_micro_infonce_e3.yaml python scripts/probe_query_head.py \\
  -c ckpts/evidence_infonce_w25e3/checkpoint_evidence_final.pt --store data/index/openqa \\
  --batch-size 64 --json-out ckpts/inject/probe_w25e3.json
```
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.data.store import load_store
from scripts.eval_store import (QueryReadout, adapted_index, headline_loop, hit_vector, load_eval_model,
                                question_prompt, recall_summary, topk_ids)
from utils import TOKENIZER_DIR, logger

KS = (5, 20)


class HiddenReadout(QueryReadout):
    """``QueryReadout`` plus the input of every IR expert's ``down_proj`` at the same positions.

    A pre-hook on ``down_proj`` stashes the rows; the IR module's own pre-hook, which runs next in
    the same expert call and knows the loop, files them under ``hidden[(ir_index, loop)]`` next to
    the unit query in ``captured``. When ``pool_weights`` (``[B, S]``, each row summing to one) is
    set, ``pooled[(ir_index, loop)]`` also holds the weighted mean of every position's input, so
    the full ``[B, S, hidden]`` tensor never leaves the hook.

    Args:
        model: a ``TinyMoETransformer`` with at least one IR expert.
    """

    def __init__(self, model):
        super().__init__(model)
        from modules.model.experts import InformationRetrievalExpert

        self.experts = [e for e in model.moe.experts if isinstance(e, InformationRetrievalExpert)]
        assert [e.ir_module for e in self.experts] == self.modules
        self.hidden: Dict[Tuple[int, int], torch.Tensor] = {}
        self.pooled: Dict[Tuple[int, int], torch.Tensor] = {}
        self.pool_weights: Optional[torch.Tensor] = None
        self._stash: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

    def _down_hook(self, ir_index: int):
        def hook(module, args):
            if self.positions is None:
                return None
            flat = args[0].reshape(-1, args[0].shape[-1])
            with torch.no_grad():
                last = flat.index_select(0, self.positions).float()
                mean = None
                if self.pool_weights is not None:
                    grid = flat.float().reshape(*self.pool_weights.shape, flat.shape[-1])
                    mean = (grid * self.pool_weights.unsqueeze(-1)).sum(1)
                self._stash[ir_index] = (last, mean)
            return None
        return hook

    def _hook(self, ir_index: int):
        base = super()._hook(ir_index)

        def hook(module, args, kwargs):
            base(module, args, kwargs)
            if self.positions is not None:
                key = (ir_index, int(kwargs.get("loop_idx", 0)))
                self.hidden[key], mean = self._stash.pop(ir_index)
                if mean is not None:
                    self.pooled[key] = mean
            return None
        return hook

    def __enter__(self):
        super().__enter__()
        for i, expert in enumerate(self.experts):
            self._handles.append(expert.down_proj.register_forward_pre_hook(self._down_hook(i)))
        return self


@torch.inference_mode()
def extract(model, prompts: List[List[int]], *, ir: int, loop: int, pad_id: int, device: str,
            batch_size: int, pool: str = "last", skip_ids: Sequence[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """``H`` pooled over each prompt, and the unit query at its last token, for one IR expert and loop.

    Batching, padding and positions follow ``eval_store.extract_queries``.

    Args:
        model: the checkpoint's model, eval mode.
        prompts: token ids per question, ending at the answer-start position.
        ir: IR expert index.
        loop: zero-based loop index.
        pad_id: right padding id.
        device: model device.
        batch_size: prompts per forward.
        pool: ``last`` (the last prompt token's state), ``mean`` (the mean over the prompt's real
            tokens, ``skip_ids`` excluded) or ``meanlast`` (the two concatenated, mean first).
        skip_ids: token ids left out of the mean (the chat control tokens).

    Returns:
        ``(H [Q, hidden or 2 * hidden], query [Q, 384])`` float32, rows in prompt order.
    """
    from modules.model.attention import cu_seqlens_from_doc_ids

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    skip = np.array(sorted(set(int(t) for t in skip_ids if t is not None)), dtype=np.int64)
    hidden, query = None, None
    with HiddenReadout(model) as readout:
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            width = max(len(prompts[i]) for i in index)
            packed = np.full((len(index), width), pad_id, dtype=np.int64)
            real = np.zeros((len(index), width), dtype=np.int64)
            for row, i in enumerate(index):
                packed[row, :len(prompts[i])] = prompts[i]
                real[row, :len(prompts[i])] = 1
            ids = torch.from_numpy(packed).to(device)
            doc = torch.from_numpy(real).to(device)
            last = np.array([len(prompts[i]) - 1 for i in index], dtype=np.int64)
            readout.positions = torch.from_numpy(np.arange(len(index), dtype=np.int64) * width + last).to(device)
            readout.tokens = len(index) * width
            readout.captured, readout.hidden, readout.pooled = {}, {}, {}
            if pool != "last":
                keep = (real > 0) & ~np.isin(packed, skip)
                empty = ~keep.any(1)
                keep[empty] = real[empty] > 0
                weights = keep / keep.sum(1, keepdims=True)
                readout.pool_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
            cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(doc)
            model(input_ids=ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                  return_hidden=True, skip_mtp=True)
            parts = {"last": [readout.hidden], "mean": [readout.pooled],
                     "meanlast": [readout.pooled, readout.hidden]}[pool]
            h = torch.cat([part[(ir, loop)] for part in parts], dim=-1).cpu().numpy()
            q = readout.captured[(ir, loop)].cpu().numpy()
            if hidden is None:
                hidden = np.zeros((len(prompts), h.shape[-1]), dtype=np.float32)
                query = np.zeros((len(prompts), q.shape[-1]), dtype=np.float32)
            hidden[index] = h
            query[index] = q
    return hidden, query


def unit(x: np.ndarray) -> np.ndarray:
    """Row-normalized float32 copy."""
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fold_ids(n: int, folds: int, seed: int = 0) -> np.ndarray:
    """Fold index per question: a seeded permutation dealt round robin into ``folds`` folds."""
    perm = np.random.default_rng(seed).permutation(n)
    out = np.zeros(n, dtype=np.int64)
    out[perm] = np.arange(n) % folds
    return out


def ridge_predictions(h: np.ndarray, y: np.ndarray, fold: np.ndarray, alphas: List[float]) -> Dict[float, np.ndarray]:
    """Held-out ridge predictions from ``h`` to ``y`` per alpha, centred on each training fold.

    Args:
        h: ``[Q, d]`` features.
        y: ``[Q, 384]`` targets.
        fold: fold index per row.
        alphas: ridge strengths, added to ``X^T X`` of the centred training rows.

    Returns:
        ``{alpha: [Q, 384]}`` raw (unnormalized) predictions, each row from the fit that excluded it.
    """
    out = {a: np.zeros_like(y, dtype=np.float32) for a in alphas}
    h64, y64 = h.astype(np.float64), y.astype(np.float64)
    eye = np.eye(h.shape[1])
    for f in range(int(fold.max()) + 1):
        train, test = fold != f, fold == f
        mx, my = h64[train].mean(0), y64[train].mean(0)
        xc, yc = h64[train] - mx, y64[train] - my
        gram, cross = xc.T @ xc, xc.T @ yc
        for a in alphas:
            w = np.linalg.solve(gram + a * eye, cross)
            out[a][test] = ((h64[test] - mx) @ w + my).astype(np.float32)
    return out


def mlp_predictions(h: np.ndarray, y: np.ndarray, fold: np.ndarray, *, hidden: int, epochs: int,
                    device: str, batch: int = 256, lr: float = 1e-3) -> np.ndarray:
    """Held-out predictions of a two-layer GELU MLP trained on cosine loss, one model per fold.

    Features are standardized with the training fold's mean and std.

    Args:
        h: ``[Q, d]`` features.
        y: ``[Q, 384]`` targets.
        fold: fold index per row.
        hidden: MLP width.
        epochs: passes over each training fold.
        device: where it trains.
        batch: rows per step.
        lr: AdamW learning rate.
    """
    out = np.zeros_like(y, dtype=np.float32)
    target = torch.from_numpy(unit(y)).to(device)
    feats = torch.from_numpy(h.astype(np.float32)).to(device)
    for f in range(int(fold.max()) + 1):
        torch.manual_seed(f)
        train = torch.from_numpy(np.nonzero(fold != f)[0]).to(device)
        test = torch.from_numpy(np.nonzero(fold == f)[0]).to(device)
        mean, std = feats[train].mean(0), feats[train].std(0).clamp_min(1e-6)
        x = (feats - mean) / std
        net = nn.Sequential(nn.Linear(h.shape[1], hidden), nn.GELU(), nn.Linear(hidden, y.shape[1])).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        for _ in range(epochs):
            perm = train[torch.randperm(len(train), device=device)]
            for start in range(0, len(perm), batch):
                rows = perm[start:start + batch]
                loss = 1 - F.cosine_similarity(net(x[rows]), target[rows], dim=-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            out[test.cpu().numpy()] = net(x[test]).cpu().numpy()
    return out


def score(name: str, pred: np.ndarray, index, ref: np.ndarray, questions: List[dict], fold: np.ndarray,
          device: str) -> dict:
    """Recall@5/@20 against ``index`` and mean cosine to ``ref``, overall, per fold and per source.

    Args:
        name: predictor name, for the log.
        pred: ``[Q, 384]`` predictions (normalized here).
        index: ``[N, 384]`` unit index on ``device``.
        ref: ``[Q, 384]`` reference vectors for the cosine.
        questions: aligned question records.
        fold: fold index per row.
        device: where scoring runs.
    """
    pred = unit(pred)
    top = topk_ids(pred, index, max(KS), device)
    cos = (pred * unit(ref)).sum(-1)
    hits = {k: hit_vector(top, questions, k) for k in KS}
    per_fold = []
    for f in range(int(fold.max()) + 1):
        mask = fold == f
        per_fold.append({"n": int(mask.sum()), "cos": float(cos[mask].mean()),
                         **{f"recall@{k}": float(hits[k][mask].mean()) for k in KS}})
    by_source = recall_summary(top, questions, KS)
    logger.info(f"[probe] scored {name}")
    return {"cos": float(cos.mean()), **{f"recall@{k}": float(hits[k].mean()) for k in KS},
            "folds": per_fold,
            "by_source": {s: {f"recall@{k}": v[k]["recall"] for k in KS} for s, v in by_source.items()}}


def main():
    parser = argparse.ArgumentParser(description="probe a head on the frozen hidden state for store retrieval")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loop", type=int, default=1, help="1-based loop whose state is read")
    parser.add_argument("--ir", type=int, default=0, help="IR expert index")
    parser.add_argument("--pool", choices=("last", "mean", "meanlast"), default="last",
                        help="H from the last prompt token, the mean over the prompt, or both concatenated")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", default="1e-2,1e-1,1,10")
    parser.add_argument("--mlp-hidden", type=int, default=1024)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    store = load_store(args.store)
    if not store.questions or store.query_keys is None:
        raise SystemExit(f"{args.store} needs questions.jsonl and query_keys_bge.npy")
    questions = store.questions
    bge_q = store.query_keys.astype(np.float32)

    model, tokenizer, template = load_eval_model(args)
    loop = headline_loop(args, model)
    if not 0 <= args.ir < len(model.moe.ir_modules):
        raise SystemExit(f"--ir {args.ir} is outside 0..{len(model.moe.ir_modules) - 1}")
    prompts = [question_prompt(template, q["question"]) for q in questions]
    skip_ids = (template.bos_id, template.eos_id, template.user_id, template.assistant_id,
                template.sys_begin_id, template.sys_end_id)
    h, own = extract(model, prompts, ir=args.ir, loop=loop, pad_id=tokenizer.pad_token_id,
                     device=args.device, batch_size=args.batch_size, pool=args.pool, skip_ids=skip_ids)
    t_capture = time.time() - t0
    logger.info(f"[probe] captured H {tuple(h.shape)} and query {tuple(own.shape)} in {t_capture:.0f}s")

    adapted = adapted_index(model, store.keys, args.ir, args.device)
    module = model.moe.ir_modules[args.ir]
    with torch.inference_mode():
        weight_dtype = module.key_adapter.weight.dtype
        adapted_q = F.normalize(module.key_adapter(torch.from_numpy(bge_q).to(args.device, weight_dtype)),
                                p=2, dim=-1).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()

    fold = fold_ids(len(questions), args.folds, seed=0)
    alphas = [float(a) for a in args.ridge.split(",")]
    t1 = time.time()
    ridge = ridge_predictions(h, bge_q, fold, alphas)
    t_ridge = time.time() - t1
    t1 = time.time()
    mlp = mlp_predictions(h, bge_q, fold, hidden=args.mlp_hidden, epochs=args.mlp_epochs, device=args.device)
    t_mlp = time.time() - t1

    bge_index = torch.from_numpy(store.keys.astype(np.float32)).to(args.device)
    predictors = {"model": score("model", own, adapted, adapted_q, questions, fold, args.device)}
    del adapted
    for a in alphas:
        predictors[f"ridge_{a:g}"] = score(f"ridge_{a:g}", ridge[a], bge_index, bge_q, questions, fold, args.device)
    predictors["mlp"] = score("mlp", mlp, bge_index, bge_q, questions, fold, args.device)
    mean = np.zeros_like(bge_q)
    for f in range(args.folds):
        mean[fold == f] = bge_q[fold != f].mean(0)
    predictors["mean"] = score("mean", mean, bge_index, bge_q, questions, fold, args.device)
    predictors["bge"] = score("bge", bge_q, bge_index, bge_q, questions, fold, args.device)
    elapsed = time.time() - t0

    print(f"\nquery head probe, {len(questions):,} questions over {len(store.chunks):,} chunks, "
          f"H {h.shape[1]}-d ({args.pool}) from ir{args.ir} loop {args.loop}, {args.folds} folds held out")
    print("  " + "predictor".ljust(14) + "".join(f"@{k}".rjust(8) for k in KS) + "  cos-bge")
    for name, row in predictors.items():
        print("  " + name.ljust(14) + "".join(f"{row[f'recall@{k}']:8.3f}" for k in KS) + f"  {row['cos']:7.3f}")
    print("  (model cosine is to key_adapter(bge question), in the selector's own space)")
    print(f"  time: capture {t_capture:.0f}s, ridge {t_ridge:.1f}s, mlp {t_mlp:.0f}s, total {elapsed:.0f}s")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"checkpoint": args.checkpoint, "flags": vars(args), "n_questions": len(questions),
                       "n_chunks": len(store.chunks), "shapes": {"H": list(h.shape), "query": list(own.shape)},
                       "seconds": {"capture": t_capture, "ridge": t_ridge, "mlp": t_mlp, "total": elapsed},
                       "predictors": predictors}, handle, indent=2)
        logger.info(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
