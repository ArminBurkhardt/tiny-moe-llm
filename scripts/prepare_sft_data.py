"""Build an SFT corpus: the original Step 12 mix, or Phase 2's abstention-repair mix.

Writes ``{data_dir}/{prefix}_train.{bin,idx,mask}`` and ``{data_dir}/{prefix}_val.{bin,idx,mask}``:
the same flat uint16 token stream + uint64 document-offset layout the pretraining corpus uses, plus
a third uint8-per-token file marking which tokens are supervised (assistant content and its
terminating EOS -- see ``modules/data/chat.py``). ``modules/data/sft_dataset.py`` reads the triple.

``--profile sft`` (default, prefix ``sft``) builds Step 12's mix:

| dataset | role |
|---|---|
| `HuggingFaceTB/smoltalk2` (no-think SFT splits) | general instruction following |
| `HuggingFaceH4/ultrachat_200k` (train_sft) | general multi-turn chat |
| `rajpurkar/squad_v2` | **primary abstention supervision** -- the unanswerable half is the point |
| `allenai/tulu-3-sft-personas-math` | short worked solutions |
| `openai/gsm8k` (socratic) | short numbered steps |
| `HuggingFaceH4/no_robots` | human-written; tone and refusal style |

``--profile repair`` (prefix ``repair``) builds Phase 2's ~50M-token repair corpus, which exists
because that mix produced a model that refuses 78.4% of *answerable* questions. Same machinery,
three differences, one per item on NEXT.md's Phase 2 list:

  * **SQuAD v2's unanswerable rows are down-sampled** (``--squad-unanswerable-fraction``, 0.40 for
    this profile), taking refusals from ~37% of that source's realized rows to ~11% of all QA
    conversations. The flag scales one source; the corpus-wide share it produces is printed at the
    end of the run and is the number NEXT.md's 10-15% target is stated against.
  * **Answerable extractive QA is added** under the *same* instruction string: `rajpurkar/squad`
    (SQuAD 1.1) and `hotpotqa/hotpot_qa` (distractor). Reusing ``SQUAD_INSTRUCTION`` verbatim is the
    whole point -- what has to change is P(answer | this exact prompt shape), and the eval measures
    that shape.
  * **General chat is kept in the mix** at half the token budget, which is ~30% of conversations --
    the unit that matters once the loss is weighted per conversation. A QA-only repair pass at
    lr=1e-5 would trade the false-abstention rate for instruction following.

Two sources NEXT.md names are deliberately **not** here: NQ-open and TriviaQA's ``nocontext``
configs are closed-book (no passage at all), so they neither share the prompt shape Gate P2 measures
nor teach extraction -- at 332M params they teach guessing, which is the opposite of the calibration
target. Adding them later is a one-line ``SFTSource``; see ``REPAIR_SOURCES``.

Phase 2's remaining item, per-conversation loss weighting, is not a data change: it lives in
``modules/data/sft_dataset.py`` + ``modules/model/mtp.py``.

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

Run from the repo root:

    python scripts/prepare_sft_data.py                    # Step 12's mix -> sft_train/sft_val
    python scripts/prepare_sft_data.py --profile repair    # Phase 2's mix -> repair_train/repair_val
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
from dataclasses import dataclass, replace
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
    # fraction of this source's *unanswerable* rows to keep (squad_v2 only; every other source is
    # answerable by construction). 1.0 keeps all of them, which is what the original Step 12 mix
    # did and what the 78.4% false-abstention rate came out of.
    unanswerable_keep: float = 1.0
    # counted into the realized "unanswerable share of QA conversations" the manifest reports.
    # Purely bookkeeping -- it does not change what gets written.
    qa: bool = False


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
              render="squad_v2", weight=0.20, qa=True),
    SFTSource("tulu_math", "allenai/tulu-3-sft-personas-math", "data/train", (".parquet",),
              render="messages", weight=0.10),
    SFTSource("no_robots", "HuggingFaceH4/no_robots", "data/train", (".parquet",),
              render="messages", weight=0.10),
    SFTSource("gsm8k", "openai/gsm8k", "socratic/train", (".parquet",),
              render="gsm8k", weight=0.05),
]

# Phase 2's repair mix (see the module docstring). QA-shaped sources take half the token budget, of
# which only squad_v2 contributes refusals and only a down-sampled ~40% of its unanswerable rows at
# that; the other half keeps general instruction following alive.
#
# **These weights are token-budget shares, but the repair run weights its loss per CONVERSATION**
# (``RepairConfig.conversation_loss_weighting``), so what actually determines a source's influence
# on the gradient is its share of *conversations*, and the two are wildly different here: a SQuAD
# row is ~206 tokens and a HotpotQA row ~1,340. The 50/50 token split below is what produces a
# roughly 70/30 conversation split between QA and chat -- measured, not assumed, from a 2M-token
# trial build. Retune the weights against the realized conversation counts the run prints, not
# against these numbers.
#
# Note what the two answerable QA sources each buy, because they are not interchangeable:
# `rajpurkar/squad` (SQuAD 1.1) is largely the same passages and questions as squad_v2's answerable
# half, so it does not add *new* evidence -- it adds answerable volume under an identical prompt,
# which is the same lever as down-sampling the refusals and is listed separately in NEXT.md for that
# reason. `hotpot_qa`'s distractor split is the one that adds genuinely new passages, and its ten
# paragraphs per question (only two of them relevant) also teach reading past distractors, which is
# what Phase 4's distractor-evidence condition will want anyway -- but at ~6.5x the tokens per
# conversation it buys the least gradient weight per token of anything here, hence the small share.
REPAIR_SOURCES = [
    SFTSource("squad_v2", "rajpurkar/squad_v2", "squad_v2/train", (".parquet",),
              render="squad_v2", weight=0.25, unanswerable_keep=0.40, qa=True),
    SFTSource("squad_v1", "rajpurkar/squad", "plain_text/train", (".parquet",),
              render="squad_v2", weight=0.17, qa=True),
    SFTSource("hotpot_qa", "hotpotqa/hotpot_qa", "distractor/train", (".parquet",),
              render="hotpot_qa", weight=0.08, qa=True),
    SFTSource("smoltalk2", "HuggingFaceTB/smoltalk2", "SFT/", (".parquet",),
              render="messages", weight=0.28, file_filter=_no_think_sft_split, holdout=True),
    SFTSource("ultrachat", "HuggingFaceH4/ultrachat_200k", "data/train_sft", (".parquet",),
              render="messages", weight=0.14),
    SFTSource("no_robots", "HuggingFaceH4/no_robots", "data/train", (".parquet",),
              render="messages", weight=0.08),
]

PROFILES = {"sft": SOURCES, "repair": REPAIR_SOURCES}
# defaults per profile: the Step 12 corpus is a full post-training run, the repair corpus is a short
# targeted finetune (NEXT.md Phase 2: "~20-50M tokens, lr=1e-5, 1 epoch"). The top of that range,
# because only ~28% of the repair corpus is supervised -- QA passages are long prompts -- so 50M
# corpus tokens is ~14M tokens of actual supervision.
PROFILE_TARGET_TOKENS = {"sft": 300_000_000, "repair": 50_000_000}
# 1.0 = keep every unanswerable row, i.e. exactly what the original Step 12 build did. 0.40 is
# calibrated, not guessed: squad_v2's realized unanswerable share is ~37% of its rows (a little above
# the source's 33.4% because refusals are short and more of them fit per token), and squad_v2 is
# ~58% of QA conversations here, so keeping 40% of them lands the QA-wide share near 11%. The run
# prints the realized number -- adjust against that.
PROFILE_UNANSWERABLE_FRACTION = {"sft": 1.0, "repair": 0.40}

SQUAD_INSTRUCTION = (
    "Answer the question using only the passage below. If the passage does not contain the "
    "answer, say so."
)


def is_unanswerable_squad(row: dict) -> bool:
    """Whether a SQuAD-schema row has no reference answer (SQuAD v2's unanswerable third).

    Shared by the renderer and the down-sampling filter so "unanswerable" means one thing. The
    numpy-array length check is the same one ``eval_abstention.squad_references`` makes: pandas
    hands back an array whose truthiness is ambiguous.
    """
    answers = row.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    return not [str(t).strip() for t in (texts if texts is not None else []) if str(t).strip()]


def render_squad_v2(row: dict, rng: random.Random,
                    phrasings: Sequence[str] = abstention.ABSTENTIONS_PASSAGE_TRAIN) -> Optional[List[dict]]:
    """One SQuAD-schema row -> a single-turn conversation. Also renders SQuAD 1.1 unchanged.

    The unanswerable half carries an empty reference answer, so an abstention has to be supplied;
    it comes from ``modules.data.abstention``'s closed set rather than free text, for the reasons in
    that module's docstring (the acceptance metric has to be able to *detect* an abstention, and a
    closed set makes that exact rather than a classification problem). ``phrasings`` defaults to the
    **wide** training set -- Phase 2's fix #4, so no single refusal string is the whole target.

    The instruction explicitly licenses abstention. Without it, an unanswerable target is
    indistinguishable from a model that simply refuses on a question it could have answered --
    the prompt has to make "say so" a legitimate response for the supervision to mean anything.
    SQuAD 1.1 rows go through the *same* instruction even though none of them is unanswerable: the
    quantity Phase 2 has to move is P(answer | this exact prompt shape), and a different instruction
    would train a different shape.
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
        answer = abstention.pick(phrasings, rng)

    return [
        {"role": "user", "content": f"{SQUAD_INSTRUCTION}\n\nPassage:\n{context}\n\nQuestion: {question}"},
        {"role": "assistant", "content": answer},
    ]


def render_hotpot_qa(row: dict, rng: random.Random) -> Optional[List[dict]]:
    """One HotpotQA distractor row -> a single-turn conversation under ``SQUAD_INSTRUCTION``.

    ``context`` is ``{"title": [str], "sentences": [[str]]}`` -- ten paragraphs, of which two hold
    the supporting facts and eight are retrieved distractors. All ten are rendered, in the order the
    dataset gives them, as ``"Title: sentence sentence ..."`` blocks: dropping the distractors would
    turn a multi-hop reading task into a one-paragraph lookup, and the distractors are the part that
    teaches reading *past* irrelevant evidence.

    Titles are kept because they carry the entity the hop is about, and paragraphs are joined with
    blank lines so the passage has visible structure rather than one run-on wall of text. The answer
    is a short span (or the literal "yes"/"no" for comparison questions), which is the same output
    shape SQuAD supervises.
    """
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    context = row.get("context")
    if not question or not answer or context is None:
        return None

    titles = context.get("title") if isinstance(context, dict) else None
    sentence_lists = context.get("sentences") if isinstance(context, dict) else None
    if titles is None or sentence_lists is None:
        return None

    paragraphs = []
    for title, sentences in zip(titles, sentence_lists):
        body = "".join(str(s) for s in (sentences if sentences is not None else [])).strip()
        if not body:
            continue
        paragraphs.append(f"{str(title).strip()}: {body}")
    if not paragraphs:
        return None

    passage = "\n\n".join(paragraphs)
    return [
        {"role": "user", "content": f"{SQUAD_INSTRUCTION}\n\nPassage:\n{passage}\n\nQuestion: {question}"},
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

    renderer = {
        "squad_v2": render_squad_v2, "gsm8k": render_gsm8k, "hotpot_qa": render_hotpot_qa,
    }[spec.render]
    for row in frame.to_dict("records"):
        if spec.unanswerable_keep < 1.0 and is_unanswerable_squad(row) and rng.random() >= spec.unanswerable_keep:
            # Phase 2's fix #1. Dropped *before* the renderer and therefore before this shard's
            # conversation list exists, so it never occupies a row_idx -- the resume state stays a
            # plain position in the kept list and no "skipped" counter has to be threaded through
            # the writer. That also means this filter has to be reproducible across processes, which
            # is what make_generator_factory's stable per-file seed is for.
            continue
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

            # sha1, not the builtin hash(): str hashing is salted per process (PYTHONHASHSEED), so
            # hash() gave this shard a different shuffle, different abstention phrasings and -- now
            # that unanswerable_keep draws from the same rng -- a different kept subset on every
            # restart, while the resume state points at a row_idx in the *old* ordering.
            digest = hashlib.sha1(filename.encode("utf-8")).digest()
            rng = random.Random(seed ^ int.from_bytes(digest[:4], "big"))
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
    split_prefix: str = "sft",
) -> dict:
    """Interleave sources, encode with the chat template, write both splits.

    Deliberately free of Hub calls so ``tests/test_sft_dataset.py`` can drive it with synthetic
    in-memory sources, exactly as ``prepare_data.run_phase`` is driven by
    ``tests/test_prepare_data.py``.

    Args:
        source_entries: ``[{"key", "weight", "generator_factory"}]``, the factory taking
            ``(start_file_idx, start_row_idx)`` and yielding ``(file_idx, row_idx, conversation, key)``.
            An optional ``"qa"`` flag marks a source as QA-shaped, which only affects the realized
            "unanswerable share of QA conversations" this returns.
        target_tokens: total tokens to write across BOTH splits.
        template: the chat template doing the rendering and masking.
        holdout_hashes: conversation hashes phase-2 pretraining already consumed; skipped.
        max_doc_tokens: conversations longer than this are dropped (never truncated).
        val_fraction: share of conversations routed to the val split instead of the train split.
        checkpoint_docs: fsync + persist the resume sidecar every N committed conversations.
        encode_batch_size: conversations per tokenizer call (the fast tokenizer parallelizes
            across a batch on its own threads; per-conversation calls are mostly FFI overhead).
        split_prefix: writes ``{prefix}_train`` / ``{prefix}_val``. "sft" for Step 12's corpus,
            "repair" for Phase 2's.

    Returns:
        The final resume state, including per-source realized token counts, drop reasons and
        committed abstention counts.
    """
    train_split, val_split = f"{split_prefix}_train", f"{split_prefix}_val"
    state = load_state(state_path)
    state.setdefault("sources", {})
    state.setdefault("splits", {})
    state.setdefault("skipped", {})
    for split in (train_split, val_split):
        state["splits"].setdefault(split, {"doc_count": 0, "tokens_written": 0})
    for entry in source_entries:
        source_state = state["sources"].setdefault(
            entry["key"], {"file_idx": 0, "row_idx": 0, "tokens": 0, "docs": 0, "done": False},
        )
        # counted on commit rather than at render time: what matters for Phase 2's ratio is what
        # actually landed in the corpus, after the length filter and the target-token cutoff
        source_state.setdefault("abstentions", 0)
        state["skipped"].setdefault(entry["key"], {"too_long": 0, "holdout": 0, "unrenderable": 0})

    writers = {split: SplitWriter(data_dir, split, state["splits"][split])
               for split in (train_split, val_split)}
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

        for (pick, file_idx, row_idx, conversation, key), result in zip(picks, encoded):
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

            split = val_split if split_rng.random() < val_fraction else train_split
            writers[split].write(ids, mask)
            source_state["tokens"] += len(ids)
            source_state["docs"] += 1
            if abstention.is_abstention(str(conversation[-1].get("content") or "")):
                source_state["abstentions"] += 1
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
    parser = argparse.ArgumentParser(description="Build an SFT corpus (Step 12's mix or Phase 2's repair mix)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="sft",
                        help="which source mix to build. 'sft' -> sft_train/sft_val (PLAN.md Step "
                             "12); 'repair' -> repair_train/repair_val (NEXT.md Phase 2)")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="total tokens across both splits (default: 300M for --profile sft, "
                             "50M for --profile repair)")
    parser.add_argument("--squad-unanswerable-fraction", type=float, default=None,
                        help="fraction of squad_v2's unanswerable rows to KEEP (default: 1.0 for "
                             "--profile sft, 0.55 for --profile repair). This is a fraction of that "
                             "one source's rows -- the realized share across all QA sources is "
                             "reported at the end and is what NEXT.md's 10-15%% target refers to")
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

    target_tokens = args.target_tokens or PROFILE_TARGET_TOKENS[args.profile]
    unanswerable_keep = (
        args.squad_unanswerable_fraction if args.squad_unanswerable_fraction is not None
        else PROFILE_UNANSWERABLE_FRACTION[args.profile]
    )
    # dataclasses.replace rather than mutating the module-level specs: they are shared state, and a
    # test (or a second call) reading a spec someone else's --squad-unanswerable-fraction had
    # rewritten is the kind of bug that only shows up in the realized counts
    sources = [
        replace(spec, unanswerable_keep=unanswerable_keep) if spec.render == "squad_v2" else spec
        for spec in PROFILES[args.profile]
    ]

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
    for spec in sources:
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

    total_w = sum(s.weight for s in sources)
    source_entries = [
        {
            "key": spec.key,
            "weight": spec.weight / total_w,
            "qa": spec.qa,
            "generator_factory": make_generator_factory(
                spec, files_by_source[spec.key], scratch_dir, hf_token, args.seed,
                revision_by_source[spec.key],
            ),
        }
        for spec in sources
    ]

    state_path = os.path.join(args.data_dir, f"_prepare_state_{args.profile}.json")
    logger.info(
        f"=== {args.profile}: target {target_tokens:,} tokens across {len(source_entries)} sources "
        f"(squad_v2 unanswerable rows kept: {unanswerable_keep:.0%}) ==="
    )
    t0 = time.time()
    final_state = build_corpus(
        source_entries, target_tokens, template, args.data_dir, state_path,
        holdout_hashes=holdout_hashes, max_doc_tokens=args.max_doc_tokens,
        val_fraction=args.val_fraction, seed=args.seed,
        checkpoint_docs=args.checkpoint_docs, encode_batch_size=args.encode_batch,
        split_prefix=args.profile,
    )
    elapsed = time.time() - t0

    per_source = {}
    qa_conversations, qa_abstentions = 0, 0
    for spec in sources:
        source_state = final_state["sources"][spec.key]
        target_k = int(target_tokens * (spec.weight / total_w))
        pct = (source_state["tokens"] / target_k - 1) * 100 if target_k else 0.0
        skipped = final_state["skipped"][spec.key]
        logger.info(
            f"[{args.profile}] {spec.key}: {source_state['tokens']:,} / {target_k:,} tokens "
            f"({pct:+.1f}%), {source_state['docs']:,} conversations "
            f"({source_state['abstentions']:,} abstentions), skipped "
            f"{skipped['too_long']} too long / {skipped['holdout']} holdout / "
            f"{skipped['unrenderable']} unrenderable"
        )
        if spec.qa:
            qa_conversations += source_state["docs"]
            qa_abstentions += source_state["abstentions"]
        per_source[spec.key] = {
            "repo_id": spec.repo_id,
            "revision": revision_by_source[spec.key],
            "target_tokens": target_k,
            "realized_tokens": source_state["tokens"],
            "conversations": source_state["docs"],
            "abstentions": source_state["abstentions"],
            "unanswerable_keep": spec.unanswerable_keep,
            "skipped": skipped,
        }

    # the number NEXT.md's Phase 2 target is stated against. --squad-unanswerable-fraction controls
    # one source's rows; this is what that turns into once the other QA sources, the length filter
    # and the token cutoff have had their say, and it is the figure to adjust the flag against.
    qa_share = qa_abstentions / qa_conversations if qa_conversations else float("nan")
    logger.info(
        f"[{args.profile}] unanswerable share of QA conversations: {qa_share:.1%} "
        f"({qa_abstentions:,} / {qa_conversations:,}) -- NEXT.md Phase 2 targets 10-15%"
    )

    # acceptance checks, mirroring prepare_data.py's: the idx must be monotone and end exactly at
    # the bin's length, and the mask must be exactly one byte per token
    splits = {}
    for split in (f"{args.profile}_train", f"{args.profile}_val"):
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

    # one manifest key per profile, so building the repair corpus never overwrites the record of
    # what the SFT corpus was built from
    manifest[f"{args.profile}_prep"] = {
        "tokenizer_dir": args.tokenizer_dir,
        "profile": args.profile,
        "seed": args.seed,
        "target_tokens": target_tokens,
        "squad_unanswerable_fraction": unanswerable_keep,
        "qa_conversations": qa_conversations,
        "qa_abstentions": qa_abstentions,
        "qa_unanswerable_share": qa_share,
        "max_doc_tokens": args.max_doc_tokens,
        "val_fraction": args.val_fraction,
        "holdout_hashes_applied": len(holdout_hashes),
        "elapsed_seconds": round(elapsed),
        "per_source": per_source,
        "splits": splits,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"recorded {args.profile} prep manifest in {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
