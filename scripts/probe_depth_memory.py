"""Peak memory and time of one training step per loop depth, with and without a detached prefix.

Builds the model from a seed the way ``sft.py --evidence --reader-kv`` does (shape from the seed,
key/value reader, fp32 masters for every parameter, the AdamW the trainer builds), takes one real
batch from an evidence split through ``EvidenceDataset`` and ``evidence_from_batch``, and runs one
full step per setting: forward at the setting's depth with its detached prefix, the trainer's loss
terms (per loop CE with MTP, the aux term, the selection and groundedness terms on the batch's
labels), backward, grad clip, master sync and an optimizer step. Every group's learning rate is
zero, so the weights are the same for every setting; the optimizer state is allocated once before
the first measured step, so each peak includes it as a running trainer's does.

The depth is a forward argument (``n_loops``), not a construction value: the seed's per loop
tensors are sized for its own loop count, and passes past it reuse the last ``loop_scale`` and
``evidence_loop_scale`` entry and the sinusoidal loop encoding. A model built at the deeper count
holds a few more scalars and costs the same activations.

Per loop CE weights per setting: zero for the detached passes; for the ``live`` passes after them
the yaml's ``loop_ce_weights`` truncated and rescaled as ``pretrain.loop_ce_weights_for`` does when
``live`` is at most its length, else the yaml's weights with its first weight repeated in front.

    TINY_LLM_CONFIG=config_pilot.yaml python scripts/probe_depth_memory.py \\
        --json-out ckpts/inject/probe_depth_memory.json

Each setting runs in its own child process, so one setting's peak cannot carry into the next and
an allocation past the card cannot take the other rows with it: under WSL that shows up as
``CUDA driver error: device not ready``, which leaves the CUDA context unusable. A child that fails
with it or with an out of memory error is recorded as ``oom``; any other failure as ``error``. A
row whose peak reserved memory exceeds the card's total (a spill into shared memory) is marked
``spill``.

Args:
    -c/--checkpoint: seed to build from (default ``ckpts/inject/seed_pilot_grounded.pt``).
    --settings: comma list of ``depth:detach_loops`` (default ``3:0,5:0,5:2``).
    --batch-size: rows per step (default: the evidence profile's batch size).
    --data-dir: directory holding the split (default: the evidence profile's).
    --split: evidence split to take the batch from (default ``pilot_main_train``).
    --scan-batches: batches read from the split; the one with the most evidence tokens is used.
    --repeats: steps per setting; the peak is the maximum, the time the minimum.
    --json-out: write the report here.
    --in-process: run every setting in this process (a child is started with it).
"""
import os
import sys
import gc
import json
import time
import argparse
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import transformer_engine.pytorch as te
from transformers import AutoTokenizer

from config import EvidenceConfig, ModelConfig, TrainingConfig
from modules.data.evidence_dataset import EvidenceDataset, evidence_from_batch
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.moe import is_fresh_loop_param
from modules.model.mtp import compute_mtp_loss
from modules.model.optim import build_optimizer
from modules.model.transformer import TinyMoETransformer
from scripts.pretrain import (
    USE_LOW_PRECISION, answer_start_positions, chosen_recipe, predicting_positions,
    sync_master_grads_, sync_master_values_,
)
from scripts.sft import build_sft_param_groups, load_pretrained_weights
from utils import BASE_DIR, BF16, TOKENIZER_DIR, logger, model_params_for_state_dict

GB = 1e9


def parse_settings(text: str) -> list:
    """``"3:0,5:2"`` -> ``[(3, 0), (5, 2)]``, each checked as a valid depth and prefix."""
    settings = []
    for item in text.split(","):
        depth, detach = (int(v) for v in item.split(":"))
        if not 0 <= detach < depth:
            raise SystemExit(f"setting {item}: detach_loops must be in [0, depth)")
        settings.append((depth, detach))
    return settings


def probe_loop_ce_weights(depth: int, detach_loops: int, base: list) -> list:
    """Per loop CE weights for one setting: zeros on the prefix, the yaml's shape on the rest.

    Args:
        depth: passes run.
        detach_loops: passes run without a graph, weighted zero.
        base: the yaml's ``loop_ce_weights``.
    """
    live = depth - detach_loops
    if live <= len(base):
        weights = list(base[:live])
        weights = [w / weights[-1] for w in weights] if weights[-1] > 0 else weights
    else:
        weights = [base[0]] * (live - len(base)) + list(base)
    return [0.0] * detach_loops + weights


def build_model(seed_path: str, device: str):
    """The evidence profile's model, optimizer and masters, as ``sft.py --evidence --reader-kv``."""
    model_params = EvidenceConfig.model_params()
    seed_keys = torch.load(seed_path, map_location="cpu", mmap=True)["model_state_dict"]
    model_params = model_params_for_state_dict(seed_keys, model_params)
    del seed_keys
    if not model_params.get("evidence_port"):
        raise SystemExit("the key/value reader needs a seed with the evidence port")
    model_params["evidence_reader"] = "kv"
    model = TinyMoETransformer(**model_params).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    model.moe.set_contrastive_query_stash(
        TrainingConfig.evidence_contrastive_weight > 0.0
        and TrainingConfig.evidence_contrastive_trunk_grad < 1.0
    )
    model.moe.set_router_noise(0.0)
    if getattr(EvidenceConfig, "freeze_evidence_gate", False) and model.moe.evidence_gate_scale is not None:
        model.moe.evidence_gate_scale.requires_grad_(False)
    load_pretrained_weights(model, seed_path)
    param_groups, master_pairs = build_sft_param_groups(
        model, EvidenceConfig.weight_decay, fresh_lr=getattr(EvidenceConfig, "fresh_lr", None),
        is_fresh_param=is_fresh_loop_param,
    )
    names = {id(p): n for n, p in model.named_parameters()}
    names_by_master = {id(master): names[id(param)] for param, master in master_pairs}
    optimizer = build_optimizer(TrainingConfig.optimizer, param_groups, names_by_master, lr=EvidenceConfig.lr)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    return model, optimizer, master_pairs, model_params


def pick_batch(data_dir: str, split: str, tokenizer, batch_size: int, scan: int) -> dict:
    """The batch with the most evidence tokens among the first ``scan`` of the split."""
    dataset = EvidenceDataset(
        data_dir=data_dir, tokenizer=tokenizer, batch_size=batch_size,
        max_length=EvidenceConfig.Seq_length, split=split,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
        seed=EvidenceConfig.seed, shuffle=True,
        max_evidence_tokens=EvidenceConfig.max_evidence_tokens,
        loss_weight_floor_tokens=getattr(EvidenceConfig, "loss_weight_floor_tokens", 64),
    )
    best, best_count, counts = None, -1, []
    for i, batch in enumerate(dataset):
        if i >= scan:
            break
        count = int((batch["evidence_chunk_ids"] >= 0).sum()) if "evidence_chunk_ids" in batch else 0
        counts.append(count)
        if count > best_count:
            best, best_count = batch, count
    if best is None or best_count <= 0:
        raise SystemExit(f"no batch with evidence in the first {scan} of {split}")
    best["_scan_counts"] = counts
    return best


def run_step(model, batch: dict, depth: int, detach_loops: int, loop_ce_weights: list,
             optimizer, master_pairs, pad_token_id: int, device: str) -> dict:
    """One training step as ``pretrain.train_step`` takes it, at one depth and prefix."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    document_ids = batch["document_ids"].to(device)
    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
    pad_mask = input_ids == pad_token_id
    answerable = batch["answerable_ids"].to(device) if "answerable_ids" in batch else None
    evidence = evidence_from_batch(model, batch, cu_seqlens)
    with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
        out = model(
            input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            **ModelConfig.Forward, return_aux_loss=True, return_hidden=True,
            n_loops=depth, detach_loops=detach_loops, evidence=evidence, token_mask=~pad_mask,
        )
        hidden, aux_loss = out[0], out[1]
        mtp = out[2] if len(out) == 3 else None
        loss, loss_ce = compute_mtp_loss(
            hidden, labels, mtp_outputs=mtp,
            lm_head=model.mtp_head.lm_head if mtp is not None else None,
            lambda_mtp=TrainingConfig.lambda_mtp, main_lm_head=model.lm_head, pad_mask=pad_mask,
            loop_ce_weights=loop_ce_weights, loop_ce_subsample=TrainingConfig.loop_ce_subsample,
        )
        if TrainingConfig.moe_balancing == "aux":
            loss = loss + TrainingConfig.aux_loss_weight * aux_loss
        if TrainingConfig.evidence_selection_weight > 0.0 and evidence is not None and evidence.chunk_gold is not None:
            selection = model.moe.evidence_selection_term(
                evidence.chunk_gold, supervised=predicting_positions(labels).reshape(-1)
            )
            if selection is not None:
                loss = loss + TrainingConfig.evidence_selection_weight * selection
        if TrainingConfig.evidence_contrastive_weight > 0.0 and evidence is not None and evidence.chunk_gold is not None:
            contrastive = model.moe.evidence_contrastive_term(
                evidence.chunk_gold, answer_start_positions(labels).reshape(-1),
                temperature=TrainingConfig.evidence_contrastive_temperature,
                loops=TrainingConfig.evidence_contrastive_loops,
                trunk_grad=TrainingConfig.evidence_contrastive_trunk_grad,
            )
            if contrastive is not None:
                loss = loss + TrainingConfig.evidence_contrastive_weight * contrastive
        if (TrainingConfig.groundedness_weight > 0.0 and model.groundedness_head is not None
                and evidence is not None and evidence.chunk_gold is not None and answerable is not None):
            grounded = model.groundedness_term(evidence.chunk_gold, answerable, answer_start_positions(labels))
            if grounded is not None:
                loss = loss + TrainingConfig.groundedness_weight * grounded
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
    sync_master_grads_(master_pairs)
    optimizer.step()
    sync_master_values_(master_pairs)
    for bf16_param, _ in master_pairs:
        bf16_param.grad = None
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "loss_ce": float(loss_ce.detach()), "grad_norm": float(grad_norm)}


def allocate_optimizer_state(optimizer, master_pairs):
    """One zero gradient step at lr 0, so the moments exist before the first measured step."""
    for _, master in master_pairs:
        master.grad = torch.zeros_like(master)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def measure(args, settings: list) -> dict:
    """Every setting in this process: the batch, the model, and one row per setting."""
    device = "cuda"
    seed_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(BASE_DIR, args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    batch = pick_batch(data_dir, args.split, tokenizer, args.batch_size, args.scan_batches)
    evidence_tokens = int((batch["evidence_chunk_ids"] >= 0).sum())
    prompt_tokens = int((batch["input_ids"] != tokenizer.pad_token_id).sum())
    batch_info = {
        "source": "real", "split": args.split, "rows": int(batch["input_ids"].shape[0]),
        "seq_len": int(batch["input_ids"].shape[1]), "prompt_tokens": prompt_tokens,
        "evidence_tokens": evidence_tokens, "evidence_ratio": evidence_tokens / max(prompt_tokens, 1),
        "evidence_width": int(batch["evidence_ids"].shape[1]),
        "scanned_evidence_tokens": batch.pop("_scan_counts"),
    }
    logger.info(f"batch: {batch_info}")

    model, optimizer, master_pairs, model_params = build_model(seed_path, device)
    allocate_optimizer_state(optimizer, master_pairs)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    base_weights = list(TrainingConfig.loop_ce_weights)

    rows = []
    for depth, detach in settings:
        weights = probe_loop_ce_weights(depth, detach, base_weights)
        row = {"setting": f"{depth}:{detach}", "depth": depth, "detach_loops": detach,
               "loop_ce_weights": weights, "status": "ok"}
        peak_alloc, peak_reserved, times, result = 0, 0, [], None
        try:
            for _ in range(args.repeats):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = run_step(model, batch, depth, detach, weights, optimizer, master_pairs,
                                  tokenizer.pad_token_id, device)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
        except torch.cuda.OutOfMemoryError as err:
            row["status"] = "oom"
            row["error"] = str(err).splitlines()[0]
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
        if row["status"] == "ok":
            row.update(result)
            row.update({
                "peak_allocated_gb": peak_alloc / GB, "peak_reserved_gb": peak_reserved / GB,
                "seconds": min(times), "seconds_all": times,
            })
            if peak_reserved > total_memory:
                row["status"] = "spill"
        rows.append(row)
        logger.info(f"setting {row['setting']}: {row}")
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "checkpoint": args.checkpoint, "batch": batch_info, "repeats": args.repeats,
        "device_total_gb": total_memory / GB, "n_loops_built": model_params.get("n_loops"),
        "rows": rows,
    }


def run_child(args, depth: int, detach: int) -> dict:
    """One setting in a fresh process; a failed child becomes an ``oom`` or ``error`` row."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "row.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--in-process",
               "-c", args.checkpoint, "--settings", f"{depth}:{detach}",
               "--batch-size", str(args.batch_size), "--data-dir", args.data_dir,
               "--split", args.split, "--scan-batches", str(args.scan_batches),
               "--repeats", str(args.repeats), "--json-out", out]
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        sys.stderr.write(proc.stderr)
        if proc.returncode == 0 and os.path.isfile(out):
            with open(out, "r", encoding="utf-8") as f:
                return json.load(f)
    lines = [line for line in proc.stderr.splitlines() if line.strip()]
    oom = any(key in proc.stderr for key in ("out of memory", "OutOfMemoryError", "device not ready"))
    row = {"setting": f"{depth}:{detach}", "depth": depth, "detach_loops": detach,
           "status": "oom" if oom else "error", "error": lines[-1] if lines else "",
           "returncode": proc.returncode}
    return {"rows": [row]}


def print_report(report: dict, split: str):
    batch_info = report["batch"]
    print("\n=== depth memory probe ===")
    print(f"seed {os.path.basename(report['checkpoint'])}, batch {batch_info['rows']} x "
          f"{batch_info['seq_len']} ({batch_info['source']}, {split}), prompt tokens "
          f"{batch_info['prompt_tokens']:,}, evidence tokens {batch_info['evidence_tokens']:,} "
          f"(ratio {batch_info['evidence_ratio']:.3f}), card {report['device_total_gb']:.1f} GB")
    print(f"{'setting':>8} {'depth':>5} {'detached':>8} {'peak alloc GB':>14} {'peak resv GB':>13} "
          f"{'seconds':>8} {'loss':>8} {'status':>7}")
    for row in report["rows"]:
        if "peak_allocated_gb" not in row:
            print(f"{row['setting']:>8} {row['depth']:>5} {row['detach_loops']:>8} {row['status']:>14} "
                  f"{row['status']:>13} {'-':>8} {'-':>8} {row['status']:>7}")
        else:
            print(f"{row['setting']:>8} {row['depth']:>5} {row['detach_loops']:>8} "
                  f"{row['peak_allocated_gb']:>14.2f} {row['peak_reserved_gb']:>13.2f} "
                  f"{row['seconds']:>8.3f} {row['loss']:>8.4f} {row['status']:>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--checkpoint", default="ckpts/inject/seed_pilot_grounded.pt")
    parser.add_argument("--settings", default="3:0,5:0,5:2")
    parser.add_argument("--batch-size", type=int, default=EvidenceConfig.Batch_size)
    parser.add_argument("--data-dir", default=EvidenceConfig.data_dir)
    parser.add_argument("--split", default="pilot_main_train")
    parser.add_argument("--scan-batches", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()
    settings = parse_settings(args.settings)

    if args.in_process:
        report = measure(args, settings)
    else:
        report, rows = None, []
        for depth, detach in settings:
            child = run_child(args, depth, detach)
            if report is None and "batch" in child:
                report = child
            rows.extend(child["rows"])
        if report is None:
            raise SystemExit(f"no setting ran: {rows}")
        report["rows"] = rows
    print_report(report, args.split)

    if args.json_out:
        out =args.json_out if os.path.isabs(args.json_out) else os.path.join(BASE_DIR, args.json_out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
