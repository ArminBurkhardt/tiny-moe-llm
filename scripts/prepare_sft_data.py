"""Build the SFT corpus from PLAN.md Step 12's five sources.

Writes ``{data_dir}/sft_train.{bin,idx,mask}`` and ``{data_dir}/sft_val.{bin,idx,mask}``: the same
flat uint16 token stream + uint64 document-offset layout the pretraining corpus uses, plus a third
uint8-per-token file marking which tokens are supervised (assistant content and its terminating
EOS -- see ``modules/data/chat.py``). ``modules/data/sft_dataset.py`` reads the triple.

| dataset | role |
|---|---|
| `HuggingFaceTB/smoltalk2` (no-think SFT splits) | general instruction following |
| `rajpurkar/squad_v2` | **primary abstention supervision** -- the unanswerable half is the point |
| `allenai/tulu-3-sft-personas-math` | short worked solutions |
| `openai/gsm8k` (socratic) | short numbered steps |
| `HuggingFaceH4/no_robots` | human-written; tone and refusal style |

Unlike ``scripts/prepare_data.py`` this runs **locally**, not on the rented box: the pretrained
checkpoint and ``manifest.json`` come down from the Hub once pretraining finishes, and everything
after that happens on the dev GPU. It keeps the same download-one-shard-then-delete and
checkpoint-the-resume-state structure anyway -- the sources total a few GB and the machinery is
already proven -- but the interruption-safety stakes are "redo an hour", not "redo forty".

Three things here are load-bearing for correctness rather than convenience:

  * **The smoltalk2 holdout is honoured.** Phase-2 pretraining already trained on some smoltalk2
    conversations and recorded their hashes in ``manifest.json``'s
    ``data_prep.smoltalk2_holdout_hashes``. Those hashes are of ``prepare_data.render_pretrain_chat``'s
    output, so the exclusion check reuses that exact function via import rather than reimplementing
    the rendering (PLAN.md Step 12: "Exclude the smoltalk2 holdout ids").
  * **Only train splits are consumed.** ``squad_v2``'s validation split and ``gsm8k``'s test split
    are left untouched -- they are the acceptance-metric eval sets (abstention precision/recall on
    the unanswerable split), and pulling them in here would make that number meaningless.
  * **Over-long conversations are dropped, never truncated.** A truncated conversation loses its
    supervised EOS, which teaches the model not to stop. Per-source drop counts land in the
    manifest so a mix that quietly lost most of a source is visible.

Run from the repo root: `python scripts/prepare_sft_data.py`.
"""

import os
import sys
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

from modules.data import abstention
from modules.data.chat import ChatTemplate
from config import TrainingConfig
from scripts.prepare_data import doc_hash, load_state, render_pretrain_chat, save_state_atomic
from utils import BASE_DIR, HF_UPLOAD_REPO, TOKENIZER_DIR, get_hf_token, logger

MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")


@dataclass
class SFTSource:
    key: str
    repo_id: str
    file_prefix: str
    file_suffix: tuple
    # how the parquet rows become conversations. "messages" reads a role/content list column
    # directly; the others are per-dataset renderers below.
    render: str
    weight: float
    file_filter: Optional[Callable[[str], bool]] = None
    holdout: bool = False              # check each document against the pretraining holdout hashes


def _no_think_sft_split(path: str) -> bool:
    # same filter prepare_data.py applies: smoltalk2's reasoning splits are deliberately excluded
    # (PLAN.md: no long-CoT traces at this scale -- Small Model Learnability Gap)
    return "_no_think" in path


# Weights are token-budget shares, not example counts. squad_v2 is deliberately over-weighted
# relative to its size: PLAN.md calls it the *primary* abstention supervision and abstention is the
# whole post-training target, so it gets a fifth of the budget rather than the ~4% its raw token
# count would earn in a proportional mix. gsm8k and no_robots are tiny (~7.5k and ~9.5k examples)
# and will exhaust well before their share -- the interleave redistributes to whatever is still
# live and logs the realized split, which is the honest outcome for a source that small.
#
# Note what that redistribution means for --target-tokens: squad_v2 (~130k QA pairs, ~30-35M tokens
# once rendered) exhausts at any target above ~175M, so its ABSOLUTE contribution is fixed and only
# its SHARE moves -- ~11% of a 300M corpus, ~6-7% of a 500M one. Growing the corpus therefore
# dilutes abstention supervision even though the weight below is unchanged. If eval_abstention.py's
# recall comes out weak, a smaller --target-tokens is the knob, not a bigger squad_v2 weight.
SOURCES = [
    SFTSource("smoltalk2", "HuggingFaceTB/smoltalk2", "SFT/", (".parquet",),
              render="messages", weight=0.35, file_filter=_no_think_sft_split, holdout=True),
    # general multi-turn chat. Exists so the corpus can be grown past ~300M tokens without the
    # entire increase landing on smoltalk2: it and smoltalk2 are the only two sources here with
    # enough data to absorb a redistribution, and a corpus that is 80% one source is a narrower
    # instruction distribution than the same token count split across two. NOT part of
    # prepare_data.py's pretraining mix, so unlike smoltalk2 it needs no holdout check. train_sft
    # only -- train_gen is the raw generation-prompt half, and the test_* splits stay untouched for
    # the same reason squad_v2's validation split does.
    SFTSource("ultrachat", "HuggingFaceH4/ultrachat_200k", "data/train_sft", (".parquet",),
              render="messages", weight=0.20),
    SFTSource("squad_v2", "rajpurkar/squad_v2", "squad_v2/train", (".parquet",),
              render="squad_v2", weight=0.20),
    SFTSource("tulu_math", "allenai/tulu-3-sft-personas-math", "data/train", (".parquet",),
              render="messages", weight=0.10),
    SFTSource("no_robots", "HuggingFaceH4/no_robots", "data/train", (".parquet",),
              render="messages", weight=0.10),
    SFTSource("gsm8k", "openai/gsm8k", "socratic/train", (".parquet",),
              render="gsm8k", weight=0.05),
]

SQUAD_INSTRUCTION = (
    "Answer the question using only the passage below. If the passage does not contain the "
    "answer, say so."
)


def render_squad_v2(row: dict, rng: random.Random) -> Optional[List[dict]]:
    """One SQuAD v2 row -> a single-turn conversation.

    The unanswerable half carries an empty reference answer, so an abstention has to be supplied;
    it comes from ``modules.data.abstention``'s small fixed set rather than free text, for the
    reasons in that module's docstring (the acceptance metric has to be able to *detect* an
    abstention, and a closed set makes that exact rather than a classification problem).

    The instruction explicitly licenses abstention. Without it, an unanswerable target is
    indistinguishable from a model that simply refuses on a question it could have answered --
    the prompt has to make "say so" a legitimate response for the supervision to mean anything.
    """
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    if not context or not question:
        return None

    answers = row.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    # pandas hands back a numpy array here, whose truthiness is ambiguous -- length-check it
    candidates = [str(t).strip() for t in (texts if texts is not None else []) if str(t).strip()]
    if candidates:
        answer = candidates[0]
    else:
        answer = abstention.pick(abstention.ABSTENTIONS_PASSAGE, rng)

    return [
        {"role": "user", "content": f"{SQUAD_INSTRUCTION}\n\nPassage:\n{context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": answer},
    ]


def render_gsm8k(row: dict, rng: random.Random) -> Optional[List[dict]]:
    """One GSM8K socratic row -> a single-turn conversation.

    Two rewrites of the reference solution, both about not teaching the model to emit markup it
    cannot use: the ``<<48/2=24>>`` calculator annotations are stripped (they are a dataset
    artifact of a tool the model does not have), and the ``#### 72`` answer marker becomes a plain
    sentence. The socratic config's leading sub-questions are kept as-is -- short numbered steps
    are exactly the role PLAN.md assigns this source.
    """
    import re

    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or not answer:
        return None

    answer = re.sub(r"<<[^>]*>>", "", answer)
    body, _, final = answer.partition("####")
    body = body.strip()
    final = final.strip()
    if final:
        body = f"{body}\nThe answer is {final}." if body else f"The answer is {final}."
    if not body:
        return None
    return [{"role": "user", "content": question}, {"role": "assistant", "content": body}]


def rows_to_conversations(spec: SFTSource, frame: pd.DataFrame, rng: random.Random):
    """Yield ``(conversation, holdout_key)`` for every usable row of one shard.

    ``holdout_key`` is the pretraining hash of the conversation (only computed for sources that
    participate in the holdout), or None.
    """
    if spec.render == "messages":
        if "messages" not in frame.columns:
            raise RuntimeError(
                f"source {spec.key}: expected a 'messages' column, got {sorted(frame.columns)}"
            )
        for msgs in frame["messages"]:
            if msgs is None or len(msgs) == 0:
                continue
            # normalize away numpy/pyarrow row objects so downstream code sees plain dicts
            conversation = [
                {"role": str(m["role"]), "content": m.get("content")}
                for m in msgs
                if m is not None and m.get("role") is not None
            ]
            if not conversation:
                continue
            # hashed from the RAW rows, not the normalized copy above: this has to reproduce
            # prepare_data.py's holdout hash byte-for-byte, and that one saw the raw rows
            key = doc_hash(render_pretrain_chat(msgs)) if spec.holdout else None
            yield conversation, key
        return

    renderer = {"squad_v2": render_squad_v2, "gsm8k": render_gsm8k}[spec.render]
    for row in frame.to_dict("records"):
        conversation = renderer(row, rng)
        if conversation:
            yield conversation, None


def make_generator_factory(spec: SFTSource, files: Sequence[str], scratch_dir: str,
                           hf_token: Optional[str], seed: int, revision: Optional[str]) -> Callable:
    """factory(start_file_idx, start_row_idx) -> generator of (file_idx, row_idx, conversation, key).

    One shard in flight at a time, deleted right after it is read -- same shape as
    ``prepare_data.make_hf_generator_factory``, and the revision is pinned for the same reason: a
    repo that updates mid-run would reshuffle the sorted file list and silently repoint the resume
    state's ``file_idx`` at a different file.
    """
    def factory(start_file_idx: int, start_row_idx: int) -> Iterator:
        for file_idx in range(start_file_idx, len(files)):
            filename = files[file_idx]
            try:
                local_path = hf_hub_download(
                    repo_id=spec.repo_id, filename=filename, repo_type="dataset",
                    local_dir=scratch_dir, token=hf_token, revision=revision,
                )
            except Exception as e:
                raise RuntimeError(f"failed to download {spec.repo_id}/{filename}: {e}") from e

            # pyarrow forced for the same reason prepare_data.py forces it: fastparquet silently
            # returns None for list<struct<role,content>> columns instead of raising, which would
            # make every chat source contribute zero conversations with no error at all
            frame = pd.read_parquet(local_path, engine="pyarrow")
            try:
                os.remove(local_path)
            except OSError:
                pass

            rng = random.Random(seed ^ (hash(filename) & 0xFFFFFFFF))
            conversations = list(rows_to_conversations(spec, frame, rng))
            rng.shuffle(conversations)
            del frame

            row_start = start_row_idx if file_idx == start_file_idx else 0
            for row_idx in range(row_start, len(conversations)):
                conversation, key = conversations[row_idx]
                yield file_idx, row_idx, conversation, key

    return factory


def truncate_to_state(bin_path: str, idx_path: str, mask_path: str, split_state: dict) -> None:
    """Drop bin/idx/mask bytes past the last confirmed checkpoint.

    All three files move together, so all three get trimmed together -- a mask left one document
    longer than its bin would fail ``SFTDataset``'s length check at train time, which is the right
    error but a needlessly late one.
    """
    tokens, docs = split_state.get("tokens_written", 0), split_state.get("doc_count", 0)
    for path, target in ((bin_path, tokens * 2), (idx_path, (docs + 1) * 8), (mask_path, tokens)):
        if os.path.exists(path):
            current = os.path.getsize(path)
            if target < current:
                with open(path, "r+b") as f:
                    f.truncate(target)
                logger.warning(
                    f"truncated {os.path.basename(path)} from {current} to {target} bytes "
                    f"(dropping the uncommitted tail of a prior interruption)"
                )


class SplitWriter:
    """Append-only writer for one ``{split}.{bin,idx,mask}`` triple."""

    def __init__(self, data_dir: str, split: str, state: dict):
        self.split = split
        self.bin_path = os.path.join(data_dir, f"{split}.bin")
        self.idx_path = os.path.join(data_dir, f"{split}.idx")
        self.mask_path = os.path.join(data_dir, f"{split}.mask")
        self.state = state
        truncate_to_state(self.bin_path, self.idx_path, self.mask_path, state)
        self.bin_f = open(self.bin_path, "ab")
        self.idx_f = open(self.idx_path, "ab")
        self.mask_f = open(self.mask_path, "ab")
        if os.path.getsize(self.idx_path) == 0:
            self.idx_f.write(np.array([0], dtype=np.uint64).tobytes())
            self.idx_f.flush()

    def write(self, ids: Sequence[int], mask: Sequence[int]) -> int:
        self.bin_f.write(np.asarray(ids, dtype=np.uint16).tobytes())
        self.mask_f.write(np.asarray(mask, dtype=np.uint8).tobytes())
        self.state["tokens_written"] += len(ids)
        self.state["doc_count"] += 1
        self.idx_f.write(np.asarray([self.state["tokens_written"]], dtype=np.uint64).tobytes())
        return len(ids)

    def sync(self) -> None:
        for f in (self.bin_f, self.idx_f, self.mask_f):
            f.flush()
            os.fsync(f.fileno())

    def close(self) -> None:
        for f in (self.bin_f, self.idx_f, self.mask_f):
            f.close()


def build_corpus(
    source_entries: List[dict],
    target_tokens: int,
    template: ChatTemplate,
    data_dir: str,
    state_path: str,
    holdout_hashes: Optional[set] = None,
    max_doc_tokens: int = 4094,
    val_fraction: float = 0.01,
    seed: int = 42,
    checkpoint_docs: int = 5000,
    encode_batch_size: int = 256,
) -> dict:
    """Interleave sources, encode with the chat template, write both splits.

    Deliberately free of Hub calls so ``tests/test_sft_dataset.py`` can drive it with synthetic
    in-memory sources, exactly as ``prepare_data.run_phase`` is driven by
    ``tests/test_prepare_data.py``.

    Args:
        source_entries: ``[{"key", "weight", "generator_factory"}]``, the factory taking
            ``(start_file_idx, start_row_idx)`` and yielding ``(file_idx, row_idx, conversation, key)``.
        target_tokens: total tokens to write across BOTH splits.
        template: the chat template doing the rendering and masking.
        holdout_hashes: conversation hashes phase-2 pretraining already consumed; skipped.
        max_doc_tokens: conversations longer than this are dropped (never truncated).
        val_fraction: share of conversations routed to ``sft_val`` instead of ``sft_train``.
        checkpoint_docs: fsync + persist the resume sidecar every N committed conversations.
        encode_batch_size: conversations per tokenizer call (the fast tokenizer parallelizes
            across a batch on its own threads; per-conversation calls are mostly FFI overhead).

    Returns:
        The final resume state, including per-source realized token counts and drop reasons.
    """
    state = load_state(state_path)
    state.setdefault("sources", {})
    state.setdefault("splits", {})
    state.setdefault("skipped", {})
    for split in ("sft_train", "sft_val"):
        state["splits"].setdefault(split, {"doc_count": 0, "tokens_written": 0})
    for entry in source_entries:
        state["sources"].setdefault(
            entry["key"], {"file_idx": 0, "row_idx": 0, "tokens": 0, "docs": 0, "done": False},
        )
        state["skipped"].setdefault(entry["key"], {"too_long": 0, "holdout": 0, "unrenderable": 0})

    writers = {split: SplitWriter(data_dir, split, state["splits"][split])
               for split in ("sft_train", "sft_val")}
    holdout_hashes = holdout_hashes or set()
    split_rng = random.Random(seed)

    active = {}
    for entry in source_entries:
        source_state = state["sources"][entry["key"]]
        active[entry["key"]] = {
            "weight": entry["weight"],
            "target": int(target_tokens * entry["weight"]),
            "gen": entry["generator_factory"](source_state["file_idx"], source_state["row_idx"]),
            "state": source_state,
        }

    swrr = {key: 0.0 for key in active}
    since_checkpoint = 0

    def total_tokens() -> int:
        return sum(s["tokens_written"] for s in state["splits"].values())

    def checkpoint() -> None:
        for writer in writers.values():
            writer.sync()
        save_state_atomic(state_path, state)

    def live_candidates() -> List[str]:
        return [k for k, v in active.items()
                if not v["state"]["done"] and v["state"]["tokens"] < v["target"]]

    while total_tokens() < target_tokens:
        candidates = live_candidates()
        if not candidates:
            logger.warning(
                f"all sources exhausted or at target after {total_tokens():,} of "
                f"{target_tokens:,} tokens -- see the manifest's per-source realized counts"
            )
            break

        picks = []  # (source_key, file_idx, row_idx, conversation, holdout_key)
        while len(picks) < encode_batch_size and candidates:
            # smooth weighted round robin over the still-live sources, same as prepare_data.py:
            # the mix ratio has to be baked into on-disk order because the dataset reads a
            # permutation of the whole corpus, and an exhausted source must not stall the rest
            total_w = sum(active[k]["weight"] for k in candidates)
            for k in candidates:
                swrr[k] += active[k]["weight"]
            pick = max(candidates, key=lambda k: swrr[k])
            swrr[pick] -= total_w
            try:
                file_idx, row_idx, conversation, key = next(active[pick]["gen"])
            except StopIteration:
                active[pick]["state"]["done"] = True
                candidates = live_candidates()
                continue
            # NOTE: file_idx/row_idx are applied to the state only once the conversation is
            # actually committed below -- see the identical note in prepare_data.run_phase for the
            # bug that comes from advancing at pick time.
            picks.append((pick, file_idx, row_idx, conversation, key))
            candidates = live_candidates()

        if not picks:
            break

        encoded = template.encode_batch([c for _, _, _, c, _ in picks])

        for (pick, file_idx, row_idx, _, key), result in zip(picks, encoded):
            source_state = active[pick]["state"]
            # the source position advances even for a rejected conversation: it WAS consumed, and
            # not recording that would replay it on every resume
            source_state["file_idx"] = file_idx
            source_state["row_idx"] = row_idx + 1

            if result is None:
                # unsupported roles (tool/ipython turns), or nothing supervised to train on
                state["skipped"][pick]["unrenderable"] += 1
                continue
            if key is not None and key in holdout_hashes:
                # phase-2 pretraining already trained on this exact conversation
                state["skipped"][pick]["holdout"] += 1
                continue
            ids, mask = result
            if len(ids) > max_doc_tokens:
                state["skipped"][pick]["too_long"] += 1
                continue

            split = "sft_val" if split_rng.random() < val_fraction else "sft_train"
            writers[split].write(ids, mask)
            source_state["tokens"] += len(ids)
            source_state["docs"] += 1
            since_checkpoint += 1
            if total_tokens() >= target_tokens:
                break

        if since_checkpoint >= checkpoint_docs:
            checkpoint()
            since_checkpoint = 0

    checkpoint()
    for writer in writers.values():
        writer.close()
    return state


def main():
    parser = argparse.ArgumentParser(description="Build the SFT corpus (PLAN.md Step 12)")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    parser.add_argument("--target-tokens", type=int, default=300_000_000,
                        help="total tokens across sft_train + sft_val")
    parser.add_argument("--max-doc-tokens", type=int, default=4094,
                        help="drop conversations longer than this (default: seq_length - mtp separator)")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-docs", type=int, default=5000)
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--ignore-holdout", action="store_true",
                        help="build even if manifest.json has no smoltalk2 holdout hashes (unsafe: "
                             "phase-2 pretraining conversations may leak into SFT)")
    parser.add_argument("--pull-manifest", action="store_true",
                        help="download manifest.json from the pretraining mirror repo first. Needed "
                             "on any box that did not run pretraining itself: manifest.json is "
                             "gitignored (*.json), so a fresh clone has no holdout hashes at all")
    parser.add_argument("--manifest-repo", default=None,
                        help="repo to pull the manifest from (default: config.yaml's "
                             "training.hf_upload_repo)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    scratch_dir = os.path.join(args.data_dir, "_sft_scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert len(tokenizer) <= 65536, f"tokenizer vocab {len(tokenizer)} must fit a uint16 bin file"
    template = ChatTemplate(tokenizer)

    hf_token = args.hf_token or get_hf_token()

    if args.pull_manifest:
        repo = args.manifest_repo or TrainingConfig.upload_repo(HF_UPLOAD_REPO)
        if not repo:
            raise SystemExit("--pull-manifest needs a repo: set training.hf_upload_repo or pass "
                             "--manifest-repo")
        logger.info(f"downloading manifest.json from {repo}")
        hf_hub_download(repo_id=repo, filename="manifest.json", local_dir=BASE_DIR, token=hf_token)

    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    holdout_hashes = set(manifest.get("data_prep", {}).get("smoltalk2_holdout_hashes", []))
    if not holdout_hashes and not args.ignore_holdout:
        raise SystemExit(
            "manifest.json has no data_prep.smoltalk2_holdout_hashes -- the pretraining run "
            "uploads the manifest alongside every checkpoint, so pass --pull-manifest to fetch it "
            "(manifest.json is gitignored, a fresh clone never has it), or --ignore-holdout to "
            "build without the exclusion."
        )
    logger.info(f"smoltalk2 holdout: {len(holdout_hashes):,} conversations excluded")

    hf_api = HfApi(token=hf_token)

    files_by_source, revision_by_source = {}, {}
    for spec in SOURCES:
        try:
            info = hf_api.dataset_info(spec.repo_id)
            all_files = hf_api.list_repo_files(spec.repo_id, repo_type="dataset", revision=info.sha)
        except Exception as e:
            raise RuntimeError(f"could not list files for {spec.repo_id} -- check connectivity: {e}") from e
        filtered = sorted(
            f for f in all_files
            if f.startswith(spec.file_prefix) and f.endswith(spec.file_suffix)
            and (spec.file_filter is None or spec.file_filter(f))
        )
        if not filtered:
            raise RuntimeError(
                f"no files matched source {spec.key} under prefix={spec.file_prefix!r} "
                f"suffix={spec.file_suffix!r} -- the Hub layout may have changed, update SFTSource"
            )
        files_by_source[spec.key] = filtered
        revision_by_source[spec.key] = info.sha
        logger.info(f"source {spec.key}: {len(filtered)} files (revision {info.sha[:10]})")

    total_w = sum(s.weight for s in SOURCES)
    source_entries = [
        {
            "key": spec.key,
            "weight": spec.weight / total_w,
            "generator_factory": make_generator_factory(
                spec, files_by_source[spec.key], scratch_dir, hf_token, args.seed,
                revision_by_source[spec.key],
            ),
        }
        for spec in SOURCES
    ]

    state_path = os.path.join(args.data_dir, "_prepare_state_sft.json")
    logger.info(f"=== SFT: target {args.target_tokens:,} tokens across {len(source_entries)} sources ===")
    t0 = time.time()
    final_state = build_corpus(
        source_entries, args.target_tokens, template, args.data_dir, state_path,
        holdout_hashes=holdout_hashes, max_doc_tokens=args.max_doc_tokens,
        val_fraction=args.val_fraction, seed=args.seed,
        checkpoint_docs=args.checkpoint_docs, encode_batch_size=args.encode_batch,
    )
    elapsed = time.time() - t0

    per_source = {}
    for spec in SOURCES:
        source_state = final_state["sources"][spec.key]
        target_k = int(args.target_tokens * (spec.weight / total_w))
        pct = (source_state["tokens"] / target_k - 1) * 100 if target_k else 0.0
        skipped = final_state["skipped"][spec.key]
        logger.info(
            f"[sft] {spec.key}: {source_state['tokens']:,} / {target_k:,} tokens ({pct:+.1f}%), "
            f"{source_state['docs']:,} conversations, skipped "
            f"{skipped['too_long']} too long / {skipped['holdout']} holdout / "
            f"{skipped['unrenderable']} unrenderable"
        )
        per_source[spec.key] = {
            "repo_id": spec.repo_id,
            "revision": revision_by_source[spec.key],
            "target_tokens": target_k,
            "realized_tokens": source_state["tokens"],
            "conversations": source_state["docs"],
            "skipped": skipped,
        }

    # acceptance checks, mirroring prepare_data.py's: the idx must be monotone and end exactly at
    # the bin's length, and the mask must be exactly one byte per token
    splits = {}
    for split in ("sft_train", "sft_val"):
        bin_path = os.path.join(args.data_dir, f"{split}.bin")
        idx_path = os.path.join(args.data_dir, f"{split}.idx")
        mask_path = os.path.join(args.data_dir, f"{split}.mask")
        idx = np.fromfile(idx_path, dtype=np.uint64)
        assert np.all(np.diff(idx) >= 0), f"{idx_path} is not monotonically non-decreasing"
        assert int(idx[-1]) * 2 == os.path.getsize(bin_path), f"{idx_path}'s last entry != len({bin_path})"
        assert int(idx[-1]) == os.path.getsize(mask_path), f"{mask_path} is not one byte per token"
        mask = np.fromfile(mask_path, dtype=np.uint8)
        supervised = int(mask.sum())
        splits[split] = {
            "conversations": len(idx) - 1,
            "tokens": int(idx[-1]),
            "supervised_tokens": supervised,
            "supervised_fraction": supervised / max(int(idx[-1]), 1),
        }
        logger.info(
            f"{split}: {len(idx) - 1:,} conversations, {int(idx[-1]):,} tokens, "
            f"{supervised:,} supervised ({100 * supervised / max(int(idx[-1]), 1):.1f}%)"
        )

    manifest["sft_prep"] = {
        "tokenizer_dir": args.tokenizer_dir,
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "max_doc_tokens": args.max_doc_tokens,
        "val_fraction": args.val_fraction,
        "holdout_hashes_applied": len(holdout_hashes),
        "elapsed_seconds": round(elapsed),
        "per_source": per_source,
        "splits": splits,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"recorded SFT prep manifest in {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
