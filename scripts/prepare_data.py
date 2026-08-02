"""Build phase1.bin/.idx and phase2.bin/.idx from the PLAN.md Step 11 source mix (PLAN.md Step 11).

Runs on the rented (interruptible, unattended) box, not locally: for each source, streams one
shard file at a time from the Hub, tokenizes it, appends raw content token ids to the target
`.bin` file and their cumulative offsets to the target `.idx` file, then deletes the local shard
-- peak disk stays at "final bin size + a few files' worth of scratch", never the full source
corpus. Sources are interleaved document-by-document at their PLAN.md mix-ratio weights (smooth
weighted round robin) so a straight sequential read at train time reproduces the mix, matching
`modules/data/dataset.py`'s "no shuffling, source order is already the mix" design.

**Interruption safety is the point, not an afterthought**: this is meant to run unattended on a
preemptible instance. Progress is checkpointed every `--checkpoint-docs` documents (or at each
phase's end): bin/idx are fsynced and a `_prepare_state_{phase}.json` sidecar records, per source,
which file/row it had reached. On restart, bin/idx are truncated back down to the last checkpoint
before continuing -- so a mid-flight crash redoes at most one checkpoint interval, never corrupts
the bin/idx pairing, and never silently skips or double-counts a document. This mirrors the
codebase's existing resume philosophy (see the mmap Dataset's doc-granular resume): bounded,
harmless redo beats exact-replay complexity.

Nemotron-CC-Math is Hub-gated -- set HF_TOKEN and accept the dataset's access request at
https://huggingface.co/datasets/<repo_id> before running, or that source fails fast with a
clear error instead of hanging. The code source (`common-pile/stackv2_edu_filtered`) is fully
public -- no token or access request needed.

NOTE: PLAN.md's phase-1 mix-ratio row sums to 90%, not 100% (55+10+7+12+3+3) -- likely a spec gap
rather than an intentional 10% gap. We preserve the relative ratios and renormalize to 100% of
`--phase1-tokens` (see `total_w` in run_phase's caller) rather than leaving 10% of the phase-1
budget unwritten; flagged here since it's a real deviation from the literal table.

Run from the repo root: `python scripts/prepare_data.py`.
"""

import os
import sys
import io
import json
import time
import random
import hashlib
import argparse
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

from utils import BASE_DIR, logger

MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
DEFAULT_TOKENIZER_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer-65536")


@dataclass
class SourceSpec:
    key: str
    repo_id: str
    file_prefix: str
    file_suffix: tuple
    format: str                                   # "parquet" | "jsonl.zst" | "jsonl.gz"
    text_columns: tuple = ("text",)                # first matching column wins, auto-detected at runtime
    messages_field: Optional[str] = None           # if set, render chat turns instead of text_columns
    file_filter: Optional[Callable[[str], bool]] = None
    gated: bool = False
    phase1_weight: float = 0.0
    phase2_weight: float = 0.0


def _no_think_split(path: str) -> bool:
    return "_no_think" in path


# PLAN.md Step 11 mix table. phase1 weights sum to 0.90 as written (see module docstring) --
# renormalized to 1.0 at run time, ratios among sources preserved.
SOURCES = [
    SourceSpec("fineweb", "HuggingFaceFW/fineweb-edu", "data/CC-MAIN-2025-26/", (".parquet",),
               format="parquet", text_columns=("text",), phase1_weight=0.55, phase2_weight=0.15),
    SourceSpec("dclm", "mlfoundations/dclm-baseline-1.0", "global-shard_01_of_10/", (".jsonl.zst",),
               format="jsonl.zst", text_columns=("text",), phase1_weight=0.10, phase2_weight=0.0),
    SourceSpec("finepdfs", "HuggingFaceFW/finepdfs-edu", "eng_Latn/", (".parquet",),
               format="parquet", text_columns=("text",), phase1_weight=0.07, phase2_weight=0.10),
    # Common Pile's stack-edu re-release: Stack-Edu's educational-quality code selection with
    # actual text materialized (unlike HuggingFaceTB/stack-edu, which only ships SWHIDs and
    # needs a separate Software Heritage S3 reconstruction step), filtered to openly-licensed
    # repos only (Blue Oak Council list) -- fully public, no gate, no access request.
    SourceSpec("code_edu", "common-pile/stackv2_edu_filtered", "", (".json.gz",),
               format="jsonl.gz", text_columns=("text",),
               phase1_weight=0.12, phase2_weight=0.22),
    SourceSpec("nemotron_math", "nvidia/Nemotron-CC-Math-v1", "4plus/", (".parquet",),
               format="parquet", text_columns=("text", "content"), gated=True,
               phase1_weight=0.03, phase2_weight=0.30),
    SourceSpec("wikipedia", "wikimedia/wikipedia", "20231101.en/", (".parquet",),
               format="parquet", text_columns=("text",), phase1_weight=0.03, phase2_weight=0.08),
    SourceSpec("smoltalk2", "HuggingFaceTB/smoltalk2", "SFT/", (".parquet",),
               format="parquet", messages_field="messages", file_filter=_no_think_split,
               phase1_weight=0.0, phase2_weight=0.15),
]


def pick_text_column(available: set, candidates: tuple, source_key: str) -> str:
    for c in candidates:
        if c in available:
            return c
    raise RuntimeError(
        f"source {source_key}: none of the candidate text columns {candidates} found in "
        f"actual columns {sorted(available)} -- update SourceSpec.text_columns for this source"
    )


def read_jsonl_zst(path: str) -> Iterator[dict]:
    import zstandard as zstd
    with open(path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_gz(path: str) -> Iterator[dict]:
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_document_texts(spec: SourceSpec, local_path: str, seed: int) -> list:
    """returns this file's documents as a list[str], shuffled at document granularity
    (PLAN.md: "shuffle at document granularity within each phase")."""
    if spec.messages_field is not None:
        df = pd.read_parquet(local_path, columns=[spec.messages_field])
        texts = []
        for msgs in df[spec.messages_field]:
            if msgs is None or len(msgs) == 0:
                continue
            rendered = "\n".join(f"{m['role']}: {m['content']}" for m in msgs if m.get("content"))
            if rendered:
                texts.append(rendered)
    elif spec.format == "parquet":
        df = pd.read_parquet(local_path)
        col = pick_text_column(set(df.columns), spec.text_columns, spec.key)
        texts = [t for t in df[col].dropna().astype(str).tolist() if t]
    elif spec.format in ("jsonl.zst", "jsonl.gz"):
        reader = read_jsonl_zst if spec.format == "jsonl.zst" else read_jsonl_gz
        texts, col = [], None
        for row in reader(local_path):
            if col is None:
                col = pick_text_column(set(row.keys()), spec.text_columns, spec.key)
            t = row.get(col)
            if t:
                texts.append(str(t))
    else:
        raise ValueError(f"unknown format {spec.format}")

    random.Random(seed).shuffle(texts)
    return texts


def make_hf_generator_factory(spec: SourceSpec, files: list, scratch_dir: str, hf_token: Optional[str], seed: int) -> Callable:
    """returns factory(start_file_idx, start_row_idx) -> generator of (file_idx, row_idx, text).

    downloads one shard file at a time, reads it fully into memory as a shuffled text list, then
    deletes the local copy immediately -- peak disk never holds more than one in-flight file per
    source regardless of how many files a source has queued.
    """
    def factory(start_file_idx: int, start_row_idx: int) -> Iterator:
        for file_idx in range(start_file_idx, len(files)):
            filename = files[file_idx]
            try:
                local_path = hf_hub_download(repo_id=spec.repo_id, filename=filename, repo_type="dataset",
                                              local_dir=scratch_dir, token=hf_token)
            except Exception as e:
                gate_hint = (
                    f" -- this dataset is Hub-gated: set HF_TOKEN and accept the access request at "
                    f"https://huggingface.co/datasets/{spec.repo_id}" if spec.gated else ""
                )
                raise RuntimeError(f"failed to download {spec.repo_id}/{filename}{gate_hint}: {e}") from e

            texts = load_document_texts(spec, local_path, seed ^ (hash(filename) & 0xFFFFFFFF))
            try:
                os.remove(local_path)
            except OSError:
                pass

            row_start = start_row_idx if file_idx == start_file_idx else 0
            for row_idx in range(row_start, len(texts)):
                yield file_idx, row_idx, texts[row_idx]

    return factory


def load_state(state_path: str) -> dict:
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state_atomic(state_path: str, state: dict) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, state_path)


def truncate_to_state(bin_path: str, idx_path: str, state: dict) -> None:
    """drop any bin/idx bytes beyond the last confirmed checkpoint -- a prior interruption may
    have left unflushed writes past this point that we can no longer trust."""
    tokens, docs = state.get("tokens_written", 0), state.get("doc_count", 0)
    for path, target in ((bin_path, tokens * 2), (idx_path, (docs + 1) * 8)):
        if os.path.exists(path):
            cur = os.path.getsize(path)
            if target < cur:
                with open(path, "r+b") as f:
                    f.truncate(target)
                logger.warning(f"truncated {path} from {cur} to {target} bytes (dropping uncommitted tail from a prior interruption)")


def run_phase(
    phase: str,
    source_entries: list,
    overall_target_tokens: int,
    tokenizer,
    data_dir: str,
    state_path: str,
    holdout_source_key: Optional[str] = None,
    checkpoint_docs: int = 2000,
    tokenize_batch: int = 256,
) -> dict:
    """core interleave/write/checkpoint loop, deliberately free of any HF Hub calls so it can be
    exercised in tests with a synthetic generator_factory.

    source_entries: list of {"key", "weight", "generator_factory": (start_file_idx, start_row_idx) -> iterator}
    """
    bin_path = os.path.join(data_dir, f"{phase}.bin")
    idx_path = os.path.join(data_dir, f"{phase}.idx")

    state = load_state(state_path)
    state.setdefault("doc_count", 0)
    state.setdefault("tokens_written", 0)
    state.setdefault("sources", {})
    state.setdefault("holdout_hashes", [])
    for e in source_entries:
        state["sources"].setdefault(e["key"], {"file_idx": 0, "row_idx": 0, "tokens": 0, "done": False})

    truncate_to_state(bin_path, idx_path, state)

    bin_f = open(bin_path, "ab")
    idx_f = open(idx_path, "ab")
    if os.path.getsize(idx_path) == 0:
        idx_f.write(np.array([0], dtype=np.uint64).tobytes())
        idx_f.flush()

    active = {}
    for e in source_entries:
        st = state["sources"][e["key"]]
        active[e["key"]] = {
            "weight": e["weight"],
            "target": int(overall_target_tokens * e["weight"]),
            "gen": e["generator_factory"](st["file_idx"], st["row_idx"]),
            "state": st,
        }

    swrr = {k: 0.0 for k in active}
    doc_count, tokens_written = state["doc_count"], state["tokens_written"]
    since_checkpoint = 0

    def checkpoint():
        bin_f.flush(); os.fsync(bin_f.fileno())
        idx_f.flush(); os.fsync(idx_f.fileno())
        state["doc_count"], state["tokens_written"] = doc_count, tokens_written
        save_state_atomic(state_path, state)

    def live_candidates():
        return [k for k, v in active.items() if not v["state"]["done"] and v["state"]["tokens"] < v["target"]]

    while tokens_written < overall_target_tokens:
        candidates = live_candidates()
        if not candidates:
            logger.warning(f"[{phase}] all sources exhausted or at target before reaching {overall_target_tokens:,} tokens "
                            f"(reached {tokens_written:,}) -- see per-source realized counts in the manifest")
            break

        batch_picks = []  # (source_key, file_idx, row_idx, text)
        while len(batch_picks) < tokenize_batch and candidates:
            # smooth weighted round robin: proportional interleave among currently-live sources
            total_w = sum(active[k]["weight"] for k in candidates)
            for k in candidates:
                swrr[k] += active[k]["weight"]
            pick = max(candidates, key=lambda k: swrr[k])
            swrr[pick] -= total_w
            try:
                file_idx, row_idx, text = next(active[pick]["gen"])
            except StopIteration:
                active[pick]["state"]["done"] = True
                candidates = live_candidates()
                continue
            # NOTE: file_idx/row_idx are *not* applied to state here -- only once the document is
            # actually committed below. Advancing it at pick time would let an early batch break
            # (target reached mid-batch) silently skip a picked-but-never-written document on the
            # next resume: state would claim it was consumed when it never made it into bin/idx.
            batch_picks.append((pick, file_idx, row_idx, text))
            candidates = live_candidates()

        if not batch_picks:
            break

        encoded = tokenizer([t for _, _, _, t in batch_picks], add_special_tokens=False, truncation=False)["input_ids"]

        for (pick, file_idx, row_idx, text), ids in zip(batch_picks, encoded):
            if not ids:
                continue
            bin_f.write(np.asarray(ids, dtype=np.uint16).tobytes())
            tokens_written += len(ids)
            doc_count += 1
            idx_f.write(np.asarray([tokens_written], dtype=np.uint64).tobytes())
            active[pick]["state"]["tokens"] += len(ids)
            active[pick]["state"]["file_idx"] = file_idx
            active[pick]["state"]["row_idx"] = row_idx + 1
            since_checkpoint += 1
            if pick == holdout_source_key:
                state["holdout_hashes"].append(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
            if tokens_written >= overall_target_tokens:
                break

        if since_checkpoint >= checkpoint_docs:
            checkpoint()
            since_checkpoint = 0

    checkpoint()
    bin_f.close()
    idx_f.close()
    return state


def main():
    parser = argparse.ArgumentParser(description="Build phase1/phase2 bin/idx corpora (PLAN.md Step 11)")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--phase1-tokens", type=int, default=25_500_000_000)
    parser.add_argument("--phase2-tokens", type=int, default=4_500_000_000)
    parser.add_argument("--phases", nargs="+", default=["phase1", "phase2"], choices=["phase1", "phase2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-docs", type=int, default=2000, help="fsync + persist resume state every N documents")
    parser.add_argument("--tokenize-batch", type=int, default=256)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    scratch_dir = os.path.join(args.data_dir, "_scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert len(tokenizer) <= 65536, f"tokenizer vocab {len(tokenizer)} must fit uint16 train.bin (Step 8)"

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    data_prep = manifest.get("data_prep", {})
    data_prep["tokenizer_dir"] = args.tokenizer_dir
    data_prep["seed"] = args.seed
    data_prep.setdefault("sources", {})

    hf_api = HfApi(token=args.hf_token)

    files_by_source, revision_by_source = {}, {}
    for spec in SOURCES:
        if spec.phase1_weight == 0.0 and spec.phase2_weight == 0.0:
            continue
        try:
            info = hf_api.dataset_info(spec.repo_id)
            all_files = hf_api.list_repo_files(spec.repo_id, repo_type="dataset")
        except Exception as e:
            gate_hint = f" (gated: accept access at https://huggingface.co/datasets/{spec.repo_id})" if spec.gated else ""
            raise RuntimeError(f"could not list files for {spec.repo_id}{gate_hint} -- check HF_TOKEN/connectivity: {e}") from e

        filtered = sorted(
            f for f in all_files
            if f.startswith(spec.file_prefix) and f.endswith(spec.file_suffix)
            and (spec.file_filter is None or spec.file_filter(f))
        )
        if not filtered:
            raise RuntimeError(f"no files matched source {spec.key} under prefix={spec.file_prefix!r} suffix={spec.file_suffix!r} "
                               f"-- the Hub layout may have changed, update SourceSpec")
        files_by_source[spec.key] = filtered
        revision_by_source[spec.key] = info.sha
        data_prep["sources"][spec.key] = {"repo_id": spec.repo_id, "revision": info.sha, "num_files_available": len(filtered), "gated": spec.gated}
        logger.info(f"source {spec.key}: {len(filtered)} files available (revision {info.sha[:10]})")

    for phase in args.phases:
        target_tokens = args.phase1_tokens if phase == "phase1" else args.phase2_tokens
        weight_attr = f"{phase}_weight"
        phase_sources = [s for s in SOURCES if getattr(s, weight_attr) > 0]
        total_w = sum(getattr(s, weight_attr) for s in phase_sources)
        # PLAN.md's phase1 row sums to 0.90, not 1.0 -- renormalize so 100% of the token budget
        # is actually written while preserving the relative ratios (see module docstring)
        if abs(total_w - 1.0) > 1e-6:
            logger.warning(f"[{phase}] source weights sum to {total_w:.3f}, renormalizing to 1.0 (ratios preserved)")

        source_entries = [
            {
                "key": spec.key,
                "weight": getattr(spec, weight_attr) / total_w,
                "generator_factory": make_hf_generator_factory(spec, files_by_source[spec.key], scratch_dir, args.hf_token, args.seed),
            }
            for spec in phase_sources
        ]

        state_path = os.path.join(args.data_dir, f"_prepare_state_{phase}.json")
        logger.info(f"=== {phase}: target {target_tokens:,} tokens across {len(source_entries)} sources ===")
        t0 = time.time()
        final_state = run_phase(
            phase, source_entries, target_tokens, tokenizer, args.data_dir, state_path,
            holdout_source_key="smoltalk2" if phase == "phase2" else None,
            checkpoint_docs=args.checkpoint_docs, tokenize_batch=args.tokenize_batch,
        )
        logger.info(f"[{phase}] wrote {final_state['tokens_written']:,} tokens / {final_state['doc_count']:,} docs in {time.time() - t0:.0f}s")

        per_source = {}
        for spec in phase_sources:
            target_k = int(target_tokens * (getattr(spec, weight_attr) / total_w))
            got = final_state["sources"][spec.key]["tokens"]
            pct = (got / target_k - 1) * 100 if target_k else 0.0
            logger.info(f"[{phase}] {spec.key}: {got:,} / {target_k:,} tokens ({pct:+.1f}%)")
            per_source[spec.key] = {
                "target_tokens": target_k,
                "realized_tokens": got,
                "files_used": files_by_source[spec.key][:final_state["sources"][spec.key]["file_idx"] + 1],
            }

        data_prep[phase] = {
            "target_tokens": target_tokens,
            "realized_tokens": final_state["tokens_written"],
            "doc_count": final_state["doc_count"],
            "per_source": per_source,
        }
        if phase == "phase2":
            data_prep["smoltalk2_holdout_hashes"] = final_state.get("holdout_hashes", [])

    # acceptance checks (PLAN.md Step 11)
    total_bin_bytes = 0
    for phase in args.phases:
        idx_path = os.path.join(args.data_dir, f"{phase}.idx")
        bin_path = os.path.join(args.data_dir, f"{phase}.bin")
        idx = np.fromfile(idx_path, dtype=np.uint64)
        assert np.all(np.diff(idx) >= 0), f"{idx_path} is not monotonically non-decreasing"
        assert int(idx[-1]) * 2 == os.path.getsize(bin_path), f"{idx_path}'s last entry != len({bin_path})"
        total_bin_bytes += os.path.getsize(bin_path)
        logger.info(f"{phase}: acceptance checks passed ({len(idx) - 1:,} docs, {int(idx[-1]):,} tokens)")

    if total_bin_bytes > 70 * 1024 ** 3:
        logger.warning(f"combined bin size {total_bin_bytes / 1e9:.1f}GB exceeds the ~70GB peak-disk acceptance bound")

    manifest["data_prep"] = data_prep
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"recorded data prep manifest in {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
