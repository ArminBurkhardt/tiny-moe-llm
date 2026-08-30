"""The fixed benchmark suite: log-likelihood multiple choice plus closed-book generation, scored
identically for this model and for the Hugging Face peers it is compared against.

CE on a held-out slice of the training corpus is a health check, not a quality claim -- it cannot
see a capability regression that leaves average CE flat, and it says nothing about where a 332M
model sits against its class. This script is the instrument that can. **One scoring path serves
every model**, which is the whole point: a peer number and our number differ because the models
differ, not because two harnesses disagree about what "accuracy" means.

What it measures
----------------

  * **Multiple choice, by log-likelihood.** For every option the conditional log-probability of the
    option's tokens given the prompt; the prediction is the argmax. Two aggregations are reported
    because the published convention differs per task: ``acc`` (raw summed log-probability) and
    ``acc_norm`` (summed log-probability divided by the option's length **in bytes**, which stops
    the scoring from simply preferring short strings). Each task declares which of the two is its
    headline metric.
  * **Generative, greedy.** Closed-book exact match on TriviaQA and NQ-open, and GSM8K, decoded
    greedily and cut at the first stop string. ``--pass-k`` switches to sampled decoding and
    reports pass@k instead -- GSM8K's is the number that decides whether RL is worth attempting
    at this size, so it gets a standing measurement rather than a guess.

Prompt formats are the EleutherAI harness's, verbatim, task by task (see each ``render_*``). That
is not cosmetic: it is what makes ``PUBLISHED`` below a real check. ``--validate`` scores a peer
against numbers that harness published for it and prints the per-task gap -- a harness that cannot
reproduce known numbers produces unknown numbers.

Isolation from the training corpora
-----------------------------------

Benchmark shards go to their own directory (``--cache-dir``, default ``data/benchmarks``) and peer
weights to ``--peer-cache-dir`` (default ``ckpts/peers``). Both are **refused** if they resolve
inside ``data/prepared*``, ``data/datasets`` or ``data/archives``: ``prepare_data.py`` and
``prepare_sft_data.py`` delete each source shard as soon as they have appended it and
``archive_corpus.py`` packs whole split directories, so a benchmark file landing in either place is
at best noise in an archive and at worst a deleted download.

Reproducibility
---------------

Every number is measured at fixed flags, and **batch size is part of the measurement** -- the MoE's
grouped GEMM tiles by the batch's per-expert row counts, so a different ``--batch-size`` changes the
bf16 accumulation order for real tokens (~0.5-1%). Peers are meant to be measured **once** and
frozen: write each run with ``--json-out`` and fold the saved files back in with ``--compare``
rather than re-running them.

Peers are scored in fp32 by default and this model in bf16. Deliberate: the published reference
numbers were produced in fp32, and bf16 moves accuracy by a few tenths of a point, which is the
same size as the gap ``--validate`` is checking. This model is measured in the precision it was
trained and is shipped in.

Run from the repo root:

```bash
# peers, measured once and frozen
python scripts/eval_benchmarks.py --peer pythia-410m --validate \\
  --json-out docs/measurements/benchmarks/pythia-410m.json

# this model, compared against everything already frozen
python scripts/eval_benchmarks.py -c ckpts/repair/checkpoint_repair_final.pt \\
  --json-out docs/measurements/benchmarks/repair_055.json \\
  --compare docs/measurements/benchmarks/*.json
```
"""
import os
import re
import sys
import glob
import json
import math
import time
import random
import argparse
import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.data.chat import ChatTemplate
from config import ModelConfig
from scripts.eval_abstention import generate_batch, normalize_answer
from utils import BASE_DIR, BF16, TOKENIZER_DIR, get_hf_token, load_model_state, logger, model_params_for_state_dict

# directories the benchmark cache must never resolve inside. the two corpus builders delete each
# source shard the moment they have appended it, and archive_corpus.py packs whole split
# directories -- a benchmark parquet in either place is a file that silently disappears or silently
# ships inside an archive of something else.
PROTECTED_DIRS = ("data/datasets", "data/archives", "data/prepared")

DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "data", "benchmarks")
DEFAULT_PEER_CACHE_DIR = os.path.join(BASE_DIR, "ckpts", "peers")

# the four peers, chosen to bracket the token axis rather than the parameter axis: this model has
# seen 16B tokens, and the interesting question is how much of the gap is architecture and how much
# is data.
PEERS = {
    "gpt2-medium": ("openai-community/gpt2-medium", "~10B", "the nearest token-parity anchor"),
    "pythia-410m": ("EleutherAI/pythia-410m", "300B", "the scaling-trend anchor"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "4T", "the data ceiling in this size class"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "18T", "practical upper bound of the class"),
}

# Published zero-shot results for EleutherAI/pythia-410m at its final step, from the Pythia repo:
# https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1/pythia-410m/zero-shot/410m_step143000.json
# They were produced by the EleutherAI harness on the same splits and the same prompt formats
# reimplemented below, which is exactly what makes them a check on this file rather than trivia.
# The file carries no hellaswag/openbookqa/boolq entry, so those tasks have no anchor here.
PUBLISHED = {
    "pythia-410m": {
        "arc_easy": {"acc": 0.5210, "acc_norm": 0.4579},
        "arc_challenge": {"acc": 0.2133, "acc_norm": 0.2432},
        "piqa": {"acc": 0.6676, "acc_norm": 0.6714},
        "winogrande": {"acc": 0.5367},
        "sciq": {"acc": 0.8110, "acc_norm": 0.7210},
        "lambada_openai": {"acc": 0.5162, "ppl_doc": 10.828},
    },
}
PUBLISHED_SOURCE = ("EleutherAI/pythia evals/pythia-v1/pythia-410m/zero-shot/410m_step143000.json "
                    "(lm-evaluation-harness, zero-shot)")


# ------------------------------------------------------------------------ task definitions

@dataclass
class MCDoc:
    """One multiple-choice question, already rendered to text.

    ``pairs`` is a list of ``(context, continuation)`` -- a list rather than one context plus N
    continuations because WinoGrande's partial evaluation varies the *context* per option and keeps
    the continuation fixed, and folding that into the same shape means one scorer covers every task.
    """
    pairs: List[Tuple[str, str]]
    gold: int


@dataclass
class GenDoc:
    """One generative question: a prompt and every answer string that counts as correct."""
    context: str
    answers: List[str]


@dataclass
class Task:
    name: str
    kind: str                       # "mc" | "gen"
    repo: str
    file_prefix: str                # parquet paths under the repo that make up the eval split
    render: Callable[[dict], Optional[object]]
    metric: str                     # headline metric: acc | acc_norm | em
    chance: float                   # score a uniform random guesser gets, for the headroom column
    revision: Optional[str] = None  # e.g. "refs/convert/parquet" for script-only dataset repos
    shots: int = 0
    shot_prefix: Optional[str] = None   # parquet paths the few-shot examples are drawn from
    max_new_tokens: int = 32
    stop: Tuple[str, ...] = ("\n",)
    answer_style: str = "plain"     # plain | gsm8k -- how a generated answer is reduced before EM


def _clean_hellaswag(text: str) -> str:
    """The harness's HellaSwag detokenizer, verbatim -- the corpus carries ``[header]`` markup and
    doubled spaces that would otherwise be scored as content."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def render_hellaswag(row: dict) -> Optional[MCDoc]:
    ctx = f"{row['ctx_a']} {row['ctx_b'].capitalize()}"
    query = _clean_hellaswag(f"{row['activity_label']}: {ctx}")
    endings = [_clean_hellaswag(e) for e in row["endings"]]
    return MCDoc([(query, " " + e) for e in endings], int(row["label"]))


def _render_arc(row: dict) -> Optional[MCDoc]:
    choices = row["choices"]
    labels = list(choices["label"])
    texts = list(choices["text"])
    key = str(row["answerKey"])
    # a handful of rows number their options 1-4 instead of A-D; the harness normalizes both sides
    if key not in labels:
        return None
    context = f"Question: {row['question']}\nAnswer:"
    return MCDoc([(context, " " + t) for t in texts], labels.index(key))


def render_piqa(row: dict) -> Optional[MCDoc]:
    context = f"Question: {row['goal']}\nAnswer:"
    return MCDoc([(context, " " + row["sol1"]), (context, " " + row["sol2"])], int(row["label"]))


def render_winogrande(row: dict) -> Optional[MCDoc]:
    """Partial evaluation: the option is spliced into the *context* and the shared tail is scored.

    Scoring the whole sentence would compare two different strings; scoring only the tail compares
    the same continuation under two different premises, which is the quantity the task is about.
    Length normalization is therefore meaningless here and ``acc`` is the headline metric.
    """
    sentence = row["sentence"]
    blank = sentence.index("_")
    tail = " " + sentence[blank + 1:].strip()
    options = [row["option1"], row["option2"]]
    return MCDoc([(sentence[:blank] + opt, tail) for opt in options], int(row["answer"]) - 1)


def render_openbookqa(row: dict) -> Optional[MCDoc]:
    choices = row["choices"]
    labels = list(choices["label"])
    key = str(row["answerKey"])
    if key not in labels:
        return None
    return MCDoc([(row["question_stem"], " " + t) for t in choices["text"]], labels.index(key))


def render_sciq(row: dict) -> Optional[MCDoc]:
    support = str(row.get("support") or "").strip()
    context = f"{support}\nQuestion: {row['question']}\nAnswer:".strip()
    options = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
    return MCDoc([(context, " " + str(o)) for o in options], 3)


def render_boolq(row: dict) -> Optional[MCDoc]:
    context = f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:"
    return MCDoc([(context, " no"), (context, " yes")], int(bool(row["answer"])))


def render_lambada(row: dict) -> Optional[MCDoc]:
    """Last-token prediction: everything but the final word is context, the final word is the target.

    Single-option, so ``acc`` here is not an argmax over alternatives but whether greedy decoding
    reproduces the held-out word exactly. Perplexity over the same tokens is reported alongside it,
    which is the second half of the published number.
    """
    text = str(row["text"]).strip()
    context, _, last = text.rpartition(" ")
    if not context or not last:
        return None
    return MCDoc([(context, " " + last)], 0)


MMLU_LETTERS = ("A", "B", "C", "D")


def render_mmlu(row: dict) -> Optional[MCDoc]:
    subject = str(row.get("subject") or "").replace("_", " ")
    options = list(row["choices"])
    if len(options) != 4:
        return None
    body = "\n".join(f"{letter}. {text}" for letter, text in zip(MMLU_LETTERS, options))
    context = (f"The following are multiple choice questions (with answers) about {subject}.\n\n"
               f"{row['question']}\n{body}\nAnswer:")
    return MCDoc([(context, " " + letter) for letter in MMLU_LETTERS], int(row["answer"]))


def _as_list(value) -> List[str]:
    """pandas hands these fields back as numpy arrays, whose truthiness is ambiguous -- the same
    length-check dance ``eval_abstention.squad_references`` does."""
    if value is None:
        return []
    return [str(v) for v in value if str(v).strip()]


def render_triviaqa(row: dict) -> Optional[GenDoc]:
    answer = row.get("answer")
    if answer is None:
        return None
    aliases = _as_list(answer.get("normalized_aliases")) or _as_list(answer.get("aliases"))
    value = str(answer.get("value") or "").strip()
    if value:
        aliases.append(value)
    if not aliases:
        return None
    return GenDoc(f"Question: {row['question']}\nAnswer:", aliases)


def render_nq_open(row: dict) -> Optional[GenDoc]:
    answers = _as_list(row.get("answer"))
    if not answers:
        return None
    return GenDoc(f"Q: {row['question']}\n\nA:", answers)


def render_gsm8k(row: dict) -> Optional[GenDoc]:
    gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
    return GenDoc(f"Question: {row['question']}\nAnswer:", [gold])


TASKS: Dict[str, Task] = {
    "hellaswag": Task("hellaswag", "mc", "Rowan/hellaswag", "data/validation",
                      render_hellaswag, "acc_norm", 0.25),
    "arc_easy": Task("arc_easy", "mc", "allenai/ai2_arc", "ARC-Easy/test",
                     _render_arc, "acc_norm", 0.25),
    "arc_challenge": Task("arc_challenge", "mc", "allenai/ai2_arc", "ARC-Challenge/test",
                          _render_arc, "acc_norm", 0.25),
    "piqa": Task("piqa", "mc", "ybisk/piqa", "plain_text/validation",
                 render_piqa, "acc_norm", 0.50, revision="refs/convert/parquet"),
    "winogrande": Task("winogrande", "mc", "allenai/winogrande", "winogrande_xl/validation",
                       render_winogrande, "acc", 0.50),
    "openbookqa": Task("openbookqa", "mc", "allenai/openbookqa", "main/test",
                       render_openbookqa, "acc_norm", 0.25),
    "sciq": Task("sciq", "mc", "allenai/sciq", "data/test", render_sciq, "acc", 0.25),
    "boolq": Task("boolq", "mc", "google/boolq", "data/validation", render_boolq, "acc", 0.50),
    "lambada_openai": Task("lambada_openai", "mc", "EleutherAI/lambada_openai", "en/test",
                           render_lambada, "acc", 0.0),
    "mmlu": Task("mmlu", "mc", "cais/mmlu", "all/test", render_mmlu, "acc", 0.25),
    "triviaqa": Task("triviaqa", "gen", "mandarjoshi/trivia_qa", "rc.nocontext/validation",
                     render_triviaqa, "em", 0.0),
    "nq_open": Task("nq_open", "gen", "google-research-datasets/nq_open", "nq_open/validation",
                    render_nq_open, "em", 0.0),
    "gsm8k": Task("gsm8k", "gen", "openai/gsm8k", "main/test", render_gsm8k, "em", 0.0,
                  shots=5, shot_prefix="main/train", max_new_tokens=256,
                  stop=("\nQuestion:", "\n\n"), answer_style="gsm8k"),
}

MC_TASKS = [name for name, task in TASKS.items() if task.kind == "mc"]
GEN_TASKS = [name for name, task in TASKS.items() if task.kind == "gen"]


# ---------------------------------------------------------------------------- data loading


def check_cache_dir(path: str, label: str) -> str:
    """Refuse a cache directory that would land inside a corpus or archive directory."""
    resolved = os.path.abspath(path)
    for protected in PROTECTED_DIRS:
        root = os.path.abspath(os.path.join(BASE_DIR, protected))
        # startswith on the prefix rather than the exact path so data/prepared_frac040 is caught too
        if resolved == root or resolved.startswith(root):
            raise SystemExit(
                f"--{label} resolves to {resolved}, inside {protected}. The corpus builders delete "
                "shards from there and archive_corpus.py packs it wholesale; pick a directory "
                "outside data/prepared*, data/datasets and data/archives."
            )
    return resolved


def load_parquet_split(task_repo: str, prefix: str, revision: Optional[str], cache_dir: str,
                       hf_token: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Download every parquet shard under ``prefix`` and concatenate it into one frame.

    Parquet is read directly with pandas/pyarrow instead of through ``datasets`` for the same reason
    the corpus builders do: ``datasets`` is not a dependency of this repo, and a builder script per
    dataset is exactly the layer that would let a peer and this model see different data.

    Returns:
        ``(frame, revision_sha)`` -- the sha is recorded in the results file so a suite that was
        frozen against one dataset revision can be told apart from one frozen against another.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=hf_token)
    if revision is None:
        revision = api.dataset_info(task_repo).sha
    names = sorted(
        f for f in api.list_repo_files(task_repo, repo_type="dataset", revision=revision)
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not names:
        raise SystemExit(f"no parquet files under {prefix!r} in {task_repo} @ {revision}")
    local_dir = os.path.join(cache_dir, task_repo.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    frames = []
    for name in names:
        path = hf_hub_download(repo_id=task_repo, filename=name, repo_type="dataset",
                               local_dir=local_dir, token=hf_token, revision=revision)
        frames.append(pd.read_parquet(path, engine="pyarrow"))
    return pd.concat(frames, ignore_index=True), revision


def build_docs(task: Task, cache_dir: str, hf_token: Optional[str],
               limit: Optional[int], seed: int) -> Tuple[List[object], str, str]:
    """Render a task's eval split into docs, plus its few-shot prefix if it has one.

    The subsample is a **seeded shuffle then truncate**, not a head slice: several of these splits
    are ordered by subject or by source article, so taking the first N would measure one corner of
    the task. Every model sees the same subsample because the seed and the shuffle are the same.
    """
    frame, revision = load_parquet_split(task.repo, task.file_prefix, task.revision, cache_dir, hf_token)
    rows = frame.to_dict("records")
    if limit is not None and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    docs = [d for d in (task.render(row) for row in rows) if d is not None]

    prefix = ""
    if task.shots and task.shot_prefix:
        shot_frame, _ = load_parquet_split(task.repo, task.shot_prefix, task.revision, cache_dir, hf_token)
        shot_rows = shot_frame.to_dict("records")
        random.Random(seed).shuffle(shot_rows)
        parts = []
        for row in shot_rows:
            doc = task.render(row)
            if doc is None:
                continue
            if isinstance(doc, GenDoc):
                # the few-shot target is the full worked answer, not the extracted number -- the
                # point of the prefix is to demonstrate the format the model should produce
                answer = str(row["answer"]).replace("####", "The answer is").strip()
                parts.append(f"{doc.context} {answer}")
            else:
                parts.append(doc.pairs[doc.gold][0] + doc.pairs[doc.gold][1])
            if len(parts) >= task.shots:
                break
        prefix = "\n\n".join(parts) + "\n\n"
    return docs, revision, prefix


# ---------------------------------------------------------------------------- backends


class Backend:
    """Everything the scorers need from a model: tokenize, score continuations, generate.

    Two implementations (this model and a Hugging Face causal LM) behind one interface, because a
    peer comparison in which the two sides ran different scoring code is not a comparison.
    """

    name: str
    info: Dict[str, object]

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def encode_many(self, texts: Sequence[str]) -> List[List[int]]:
        """Batch tokenization. Not a convenience: the fast tokenizers parallelize this in Rust, and
        calling ``encode`` once per string is what made the scoring pass CPU-bound at ~16% GPU."""
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def score(self, batch: List[Tuple[List[int], List[int]]]) -> List[Tuple[float, bool]]:
        """Per (context_ids, continuation_ids): summed continuation log-probability, and whether
        greedy decoding from that context would have produced the continuation exactly."""
        raise NotImplementedError

    def generate(self, prompts: List[List[int]], max_new_tokens: int, temperature: float) -> List[List[int]]:
        raise NotImplementedError

    def wrap_prompt(self, text: str) -> List[int]:
        """Prompt ids for a generative task, including whatever prefix the model expects."""
        raise NotImplementedError


def _pack_batch(batch: Sequence[Tuple[List[int], List[int]]], pad_id: int, device):
    """Assemble one padded ``[B, S]`` batch plus the tensors every backend needs off it.

    Built in numpy and moved across in **one** host-to-device copy. Filling the batch row by row
    with per-row ``torch.tensor(...).to(device)`` is a separate small transfer per row, and at these
    sequence lengths that dominates the actual forward pass.

    Returns:
        ``(ids, real, predict_at, target_at)`` -- the token ids, a ``[B, S]`` bool marking real
        (non-pad) positions, and the flat indices of the positions that PREDICT each continuation
        token and of the target tokens themselves. Position ``t`` predicts token ``t + 1``, so
        continuation token ``i`` of a row with a ``c``-token context is predicted from ``c + i - 1``.
        Gathering those before applying the LM head is what keeps this affordable: the alternative
        materializes ``[B, S, vocab]`` (2GB in fp32 at B=16/S=512/vocab=65536) to read a few dozen
        positions out of it.
    """
    context_lengths = np.array([len(c) for c, _ in batch], dtype=np.int64)
    continuation_lengths = np.array([len(k) for _, k in batch], dtype=np.int64)
    lengths = context_lengths + continuation_lengths
    width = int(lengths.max())

    packed = np.full((len(batch), width), pad_id, dtype=np.int64)
    for row, (context, continuation) in enumerate(batch):
        packed[row, :lengths[row]] = context + continuation

    row_starts = np.arange(len(batch), dtype=np.int64) * width
    base = np.repeat(row_starts + context_lengths, continuation_lengths)
    within = np.concatenate([np.arange(n, dtype=np.int64) for n in continuation_lengths])

    ids = torch.from_numpy(packed).to(device, non_blocking=True)
    real = (torch.arange(width, device=device)[None, :]
            < torch.from_numpy(lengths).to(device, non_blocking=True)[:, None])
    predict_at = torch.from_numpy(base + within - 1).to(device, non_blocking=True)
    target_at = torch.from_numpy(base + within).to(device, non_blocking=True)
    return ids, real, predict_at, target_at


def _reduce_scores(logits: torch.Tensor, targets: torch.Tensor,
                   continuation_lengths: Sequence[int]) -> List[Tuple[float, bool]]:
    """Summed log-probability and greedy-exactness per row, from gathered ``[N, vocab]`` logits."""
    logprobs = logits.float().log_softmax(-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    greedy = logprobs.argmax(-1) == targets
    out, cursor = [], 0
    for k_len in continuation_lengths:
        span = slice(cursor, cursor + k_len)
        out.append((float(token_logprobs[span].sum()), bool(greedy[span].all())))
        cursor += k_len
    return out


class TinyBackend(Backend):
    """This repo's model. Right-padded, with the pad run given its own attention segment.

    The segment is what makes padding invisible: flash's block-diagonal mask keeps real tokens from
    attending to a pad exactly as it keeps packed documents apart during training. Under causal
    attention right-padding would already be harmless, but the varlen path needs a segmentation
    either way and this one is the same rule the trainer uses.
    """

    def __init__(self, model, tokenizer, checkpoint_path: str, device: str, max_seq_len: int,
                 chat_template: Optional[ChatTemplate] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.chat_template = chat_template
        self.name = os.path.basename(checkpoint_path)
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.info = {}

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_many(self, texts: Sequence[str]) -> List[List[int]]:
        return self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def _final_hidden(self, ids: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        # skip_mtp: the drafted tokens are discarded here, and the head is paid over the whole
        # prefix of every scored continuation
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(doc)
        out = self.model(input_ids=ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                         return_hidden=True, skip_mtp=True)
        hidden_all = out[0] if isinstance(out, tuple) else out
        return hidden_all[-1]

    @torch.inference_mode()
    def score(self, batch):
        ids, real, predict_at, target_at = _pack_batch(batch, self.pad_id, self.device)
        hidden = self._final_hidden(ids, real.long())
        gathered = hidden.reshape(-1, hidden.size(-1)).index_select(0, predict_at)
        logits = self.model.lm_head(gathered)
        targets = ids.reshape(-1).index_select(0, target_at)
        return _reduce_scores(logits, targets, [len(k) for _, k in batch])

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens, temperature):
        # reuse the abstention eval's decoder rather than writing a second one: it is the same
        # left-padded, varlen-segmented batched greedy path, already checked for pad isolation
        generated, _, _ = generate_batch(
            self.model, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=50, eos_id=self.eos_id, pad_id=self.pad_id, device=self.device,
            max_seq_len=self.max_seq_len,
        )
        return generated

    def wrap_prompt(self, text: str) -> List[int]:
        """Prompt ids for generation. ``--chat`` routes through the SFT chat template instead."""
        if self.chat_template is not None:
            return self.chat_template.encode_prompt([{"role": "user", "content": text}])
        return [self.bos_id] + self.encode(text)


class HFBackend(Backend):
    """A Hugging Face causal LM, scored through the identical gather-then-head path.

    ``output_hidden_states=True`` plus the model's own output embedding is used rather than
    ``.logits`` for the same memory reason as above; every architecture used here (GPT-2, GPT-NeoX,
    Llama, Qwen2) appends its final hidden state to that tuple *after* the final norm, which is what
    the output embedding expects to consume.
    """

    def __init__(self, repo: str, alias: str, device: str, dtype: torch.dtype, cache_dir: str,
                 hf_token: Optional[str]):
        from transformers import AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=dtype, cache_dir=cache_dir, token=hf_token).to(device).eval()
        self.device = device
        self.name = alias
        self.head = self.model.get_output_embeddings()
        self.pad_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                       else self.tokenizer.eos_token_id)
        self.bos_id = (self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                       else self.pad_id)
        self.max_seq_len = min(int(getattr(self.model.config, "max_position_embeddings", 2048)), 4096)
        params = sum(p.numel() for p in self.model.parameters())
        self.info = {
            "repo": repo,
            "params_total": params,
            "dtype": str(dtype).replace("torch.", ""),
            "vocab_size": int(self.model.config.vocab_size),
            "max_position_embeddings": int(getattr(self.model.config, "max_position_embeddings", 0)),
            "architecture": type(self.model).__name__,
        }

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_many(self, texts: Sequence[str]) -> List[List[int]]:
        return self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    @torch.inference_mode()
    def score(self, batch):
        ids, real, predict_at, target_at = _pack_batch(batch, self.pad_id, self.device)
        # base_model resolves to whatever the architecture calls its trunk (transformer / gpt_neox /
        # model), so one call covers GPT-2, GPT-NeoX, Llama and Qwen2 without a per-family branch
        hidden = self.model.base_model(input_ids=ids, attention_mask=real.long()).last_hidden_state
        gathered = hidden.reshape(-1, hidden.size(-1)).index_select(0, predict_at)
        logits = self.head(gathered)
        targets = ids.reshape(-1).index_select(0, target_at)
        return _reduce_scores(logits, targets, [len(k) for _, k in batch])

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens, temperature):
        width = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), width), self.pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(prompts), width), dtype=torch.long, device=self.device)
        for row, prompt in enumerate(prompts):
            # left-padded, so every row's last real token sits at the same index
            ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=self.device)
            mask[row, width - len(prompt):] = 1
        out = self.model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
            top_k=50 if temperature > 0 else None,
            pad_token_id=self.pad_id,
        )
        return [row[width:].tolist() for row in out]

    def wrap_prompt(self, text: str) -> List[int]:
        return self.encode(text)


# ---------------------------------------------------------------------------- scoring


def truncate_context(context_ids: List[int], continuation_ids: List[int], budget: int) -> List[int]:
    """Left-truncate an over-long context so the pair fits. The continuation is never cut -- it is
    the thing being scored, and a truncated one would change what the question asks."""
    room = budget - len(continuation_ids)
    return context_ids[-room:] if room > 0 and len(context_ids) > room else context_ids


def score_mc_task(backend: Backend, docs: List[MCDoc], batch_size: int, max_len: int,
                  progress: str) -> dict:
    """Score every option of every doc and reduce to acc / acc_norm / perplexity.

    Requests from all docs are pooled and **length-sorted into batches**: options within a doc vary
    little in length but the tasks' documents vary by an order of magnitude, and padding is what a
    batched scorer wastes most of its compute on.
    """
    # tokenize the whole task in two batched calls rather than one call per string. the fast
    # tokenizers thread this in Rust, and several tasks repeat one context across every option
    # (HellaSwag encodes 10k contexts 4x each), so the unique-context pass is most of the saving
    flat = [(doc_idx, option_idx, context, continuation)
            for doc_idx, doc in enumerate(docs)
            for option_idx, (context, continuation) in enumerate(doc.pairs)]
    unique_contexts = list(dict.fromkeys(context for _, _, context, _ in flat))
    context_lookup = dict(zip(unique_contexts, backend.encode_many(unique_contexts)))
    continuation_ids_all = backend.encode_many([continuation for _, _, _, continuation in flat])

    requests, owners = [], []
    for (doc_idx, option_idx, context, continuation), continuation_ids in zip(flat, continuation_ids_all):
        if not continuation_ids:
            continue
        context_ids = context_lookup[context]
        if isinstance(backend, TinyBackend):
            # the corpus stores documents BOS-less and the dataset prepends one at pack time, so
            # scoring without it would put the model on an input shape it never trained on
            context_ids = [backend.bos_id] + context_ids
        elif not context_ids:
            # WinoGrande's blank can be the first character of the sentence, leaving an empty
            # context -- there has to be one token to predict the continuation's first token from
            context_ids = [backend.bos_id]
        context_ids = truncate_context(context_ids, continuation_ids, max_len)
        requests.append((context_ids, continuation_ids))
        # the length normalization divides by the bytes of the *option*, with the separating space
        # that joins it to the prompt excluded. One byte out of a two-word SciQ answer is not a
        # rounding detail -- it moved acc_norm by 2 points against the published reference
        owners.append((doc_idx, option_idx, len(continuation.lstrip().encode("utf-8")),
                       len(continuation_ids)))

    results: List[Optional[Tuple[float, bool]]] = [None] * len(requests)
    order = sorted(range(len(requests)), key=lambda i: -(len(requests[i][0]) + len(requests[i][1])))
    tokens_seen, started = 0, time.time()
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        batch = [requests[i] for i in index]
        tokens_seen += sum(len(c) + len(k) for c, k in batch)
        for i, value in zip(index, backend.score(batch)):
            results[i] = value
        if start and start % (batch_size * 100) == 0:
            logger.info(f"[{progress}] {start:,}/{len(order):,} continuations")

    per_doc: Dict[int, List[Tuple[int, float, float, bool, int]]] = {}
    for (doc_idx, option_idx, n_bytes, n_tokens), value in zip(owners, results):
        logprob, greedy = value
        per_doc.setdefault(doc_idx, []).append(
            (option_idx, logprob, logprob / max(n_bytes, 1), greedy, n_tokens))

    acc, acc_norm, greedy_hits, ce_sum, gold_tokens = [], [], [], 0.0, 0
    for doc_idx, doc in enumerate(docs):
        options = per_doc.get(doc_idx)
        if not options or len(options) != len(doc.pairs):
            continue
        options.sort()
        acc.append(float(max(options, key=lambda o: o[1])[0] == doc.gold))
        acc_norm.append(float(max(options, key=lambda o: o[2])[0] == doc.gold))
        greedy_hits.append(float(options[doc.gold][3]))
        # perplexity over the gold continuation, accumulated only over docs that actually scored, so
        # the numerator and the token count can never come from different sets of documents
        ce_sum += -options[doc.gold][1]
        gold_tokens += options[doc.gold][4]

    out = {
        "n": len(acc),
        "acc": float(np.mean(acc)) if acc else float("nan"),
        "acc_norm": float(np.mean(acc_norm)) if acc_norm else float("nan"),
        "acc_stderr": float(np.std(acc, ddof=1) / math.sqrt(len(acc))) if len(acc) > 1 else float("nan"),
        "acc_norm_stderr": float(np.std(acc_norm, ddof=1) / math.sqrt(len(acc_norm))) if len(acc_norm) > 1 else float("nan"),
        # two perplexities, because LAMBADA's published one is per DOCUMENT, not per token: its
        # target is a whole word, which several tokenizers split into ~1.4 tokens, and dividing by
        # tokens instead of documents moves the number by nearly 2x. `ppl` (per token) is the
        # generally meaningful one and the column the report prints; `ppl_doc` is what the
        # published LAMBADA figure means, and is what --validate compares against. On a task whose
        # target is a whole sentence `ppl_doc` is an astronomically large number by construction --
        # it is a per-25-token quantity there, not a broken one. Hence the roomy clamp: a tight one
        # would make every such task report the same value and look like a sentinel.
        "ppl": math.exp(min(ce_sum / max(gold_tokens, 1), 20.0)),
        "ppl_doc": math.exp(min(ce_sum / max(len(acc), 1), 60.0)),
        "greedy_exact": float(np.mean(greedy_hits)) if greedy_hits else float("nan"),
        "continuations": len(requests),
        "tokens": tokens_seen,
        "seconds": time.time() - started,
    }
    # LAMBADA's single option makes the argmax vacuous; its published "acc" is greedy exactness
    if len(docs) and len(docs[0].pairs) == 1:
        out["acc"] = out["greedy_exact"]
        out["acc_stderr"] = (float(np.std(greedy_hits, ddof=1) / math.sqrt(len(greedy_hits)))
                             if len(greedy_hits) > 1 else float("nan"))
        out["acc_norm"] = out["greedy_exact"]
    return out


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def reduce_answer(text: str, style: str) -> str:
    """Cut a generated answer down to the thing EM compares."""
    if style != "gsm8k":
        return text.strip()
    cleaned = text.replace(",", "")
    numbers = NUMBER_RE.findall(cleaned)
    return numbers[-1].rstrip(".") if numbers else cleaned.strip()


def cut_at_stop(text: str, stops: Sequence[str]) -> str:
    """Truncate a generation at the first stop string.

    Leading whitespace is stripped **before** the search, not after: several of these prompts end
    on a bare ``A:`` and the models answer on the next line, so a literal search for ``"\\n"`` over
    the raw continuation would cut every answer to the empty string and report EM 0 for a model
    that answered correctly.
    """
    text = text.lstrip()
    cut = len(text)
    for stop in stops:
        found = text.find(stop)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


def score_gen_task(backend: Backend, task: Task, docs: List[GenDoc], prefix: str, batch_size: int,
                   max_len: int, pass_k: int, temperature: float, progress: str) -> dict:
    """Greedy (or sampled, for pass@k) closed-book generation scored by exact match.

    ``pass_k > 1`` runs the whole set ``k`` times at ``temperature`` and reports the fraction of
    questions solved at least once. That is the quantity a "should we attempt RL at this size"
    decision needs -- a 0/1 reward on a task the model never samples correctly produces no gradient
    at all, so pass@k, not greedy EM, is the threshold to watch. Note that under ``pass_k > 1``
    every attempt is sampled, so the reported ``em`` is the first *sample*, not the greedy answer;
    ``greedy`` in the returned dict says which of the two it is.
    """
    prompts, references = [], []
    for doc in docs:
        ids = backend.wrap_prompt(prefix + doc.context)
        prompts.append(ids[-max_len:])
        references.append(doc.answers)

    solved = np.zeros(len(docs), dtype=bool)
    em_first = np.zeros(len(docs), dtype=np.float64)
    samples: List[str] = [""] * len(docs)
    started, tokens_seen = time.time(), 0

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for attempt in range(max(pass_k, 1)):
        temp = temperature if pass_k > 1 else 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            batch = [prompts[i] for i in index]
            tokens_seen += sum(len(p) for p in batch) * task.max_new_tokens
            generated = backend.generate(batch, task.max_new_tokens, temp)
            for i, token_ids in zip(index, generated):
                text = cut_at_stop(backend.decode(token_ids), task.stop)
                prediction = reduce_answer(text, task.answer_style)
                hit = any(normalize_answer(prediction) == normalize_answer(r) for r in references[i])
                if attempt == 0:
                    em_first[i] = float(hit)
                    samples[i] = text.strip()
                solved[i] |= hit
            if start and start % (batch_size * 20) == 0:
                logger.info(f"[{progress}] attempt {attempt + 1}/{max(pass_k, 1)}: "
                            f"{start:,}/{len(order):,} prompts")

    return {
        "n": len(docs),
        "em": float(em_first.mean()) if len(em_first) else float("nan"),
        "em_stderr": (float(np.std(em_first, ddof=1) / math.sqrt(len(em_first)))
                      if len(em_first) > 1 else float("nan")),
        f"pass@{max(pass_k, 1)}": float(solved.mean()) if len(solved) else float("nan"),
        "greedy": pass_k <= 1,
        "shots": task.shots,
        "tokens": tokens_seen,
        "seconds": time.time() - started,
        "examples": samples[:5],
    }


# ---------------------------------------------------------------------------- pretty output


def rule(title: str, width: int = 100) -> str:
    body = f"══ {title} " if title else ""
    return "\n" + body + "═" * max(width - len(body), 0)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns: Optional[str] = None) -> str:
    """Aligned box-drawn table. ``aligns`` is one char per column: ``l`` or ``r``."""
    columns = len(headers)
    aligns = aligns or "l" + "r" * (columns - 1)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def render(cells, pad=" "):
        parts = []
        for cell, width, align in zip(cells, widths, aligns):
            text = str(cell)
            parts.append(f"{pad}{text:<{width}}{pad}" if align == "l" else f"{pad}{text:>{width}}{pad}")
        return "│" + "│".join(parts) + "│"

    out = [line("┌", "┬", "┐"), render(headers), line("├", "┼", "┤")]
    out += [render(row) for row in rows]
    out.append(line("└", "┴", "┘"))
    return "\n".join(out)


def fmt(value, digits: int = 4, dash: str = "--") -> str:
    if value is None:
        return dash
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return dash
    return f"{value:.{digits}f}"


def headline(result: dict, task: Task) -> Optional[float]:
    value = result.get(task.metric)
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def print_model_diagnostic(payload: dict) -> None:
    """Everything about the model under test that a later reader would need to reproduce the run."""
    info = payload["model"]
    print(rule(f"MODEL — {info['name']}"))
    rows = [(k.replace("_", " "), str(v)) for k, v in info.items() if k != "name"]
    print(table(["property", "value"], rows, aligns="ll"))

    env = payload["environment"]
    print(table(["environment", "value"], [(k.replace("_", " "), str(v)) for k, v in env.items()],
                aligns="ll"))


def print_task_report(payload: dict) -> None:
    """Per-task results with the random-chance baseline beside them.

    ``headroom`` is ``(score - chance) / (1 - chance)``: the share of the *available* range above
    guessing that the model actually covered. At this size several tasks sit within noise of chance,
    where a raw accuracy column reads like a real number and a headroom column reads like the 0 it
    is.
    """
    print(rule(f"RESULTS — {payload['model']['name']}"))
    rows = []
    for name, result in payload["results"].items():
        task = TASKS[name]
        score = headline(result, task)
        headroom = None if score is None or task.chance >= 1 else (score - task.chance) / (1 - task.chance)
        stderr = result.get(f"{task.metric}_stderr")
        rows.append([
            name, task.metric, f"{result.get('n', 0):,}",
            fmt(score), fmt(stderr), fmt(task.chance, 2), fmt(headroom),
            fmt(result.get("acc"), 4), fmt(result.get("acc_norm"), 4),
            fmt(result.get("ppl"), 2), f"{result.get('seconds', 0):.0f}s",
        ])
    print(table(["task", "metric", "n", "score", "±", "chance", "headroom",
                 "acc", "acc_norm", "ppl", "time"], rows))

    mc_headroom = [
        (headline(result, TASKS[name]) - TASKS[name].chance) / (1 - TASKS[name].chance)
        for name, result in payload["results"].items()
        if TASKS[name].kind == "mc" and TASKS[name].chance > 0 and headline(result, TASKS[name]) is not None
    ]
    if mc_headroom:
        print(f"  mean multiple-choice headroom over chance: {np.mean(mc_headroom):+.4f} "
              f"over {len(mc_headroom)} tasks")

    for name, result in payload["results"].items():
        if TASKS[name].kind == "gen" and result.get("examples"):
            print(f"\n  {name} — first generations (greedy):")
            for text in result["examples"]:
                print(f"    {text[:160]!r}")


def print_comparison(payloads: List[dict]) -> None:
    """The peer table: one row per task, one column per model, best in each row marked."""
    print(rule("BENCHMARK COMPARISON"))
    names = [p["model"]["name"] for p in payloads]
    task_names = [n for n in TASKS if any(n in p["results"] for p in payloads)]
    rows = []
    for name in task_names:
        task = TASKS[name]
        scores = [headline(p["results"].get(name, {}), task) for p in payloads]
        best = max((s for s in scores if s is not None), default=None)
        cells = []
        for score in scores:
            if score is None:
                cells.append("--")
            else:
                cells.append(f"{score:.4f}" + ("*" if best is not None and score == best else " "))
        rows.append([name, task.metric, fmt(task.chance, 2)] + cells)
    print(table(["task", "metric", "chance"] + names, rows))
    print("  * best in row. every column was scored by this file, on the same subsample, at the "
          "same flags.")

    print(rule("HEADROOM OVER CHANCE (multiple choice)"))
    rows = []
    for name in task_names:
        task = TASKS[name]
        if task.kind != "mc" or task.chance <= 0:
            continue
        cells = []
        for payload in payloads:
            score = headline(payload["results"].get(name, {}), task)
            cells.append("--" if score is None else f"{(score - task.chance) / (1 - task.chance):+.4f}")
        rows.append([name] + cells)
    means = []
    for payload in payloads:
        values = [
            (headline(payload["results"][n], TASKS[n]) - TASKS[n].chance) / (1 - TASKS[n].chance)
            for n in task_names
            if n in payload["results"] and TASKS[n].kind == "mc" and TASKS[n].chance > 0
            and headline(payload["results"][n], TASKS[n]) is not None
        ]
        means.append(f"{np.mean(values):+.4f}" if values else "--")
    rows.append(["MEAN"] + means)
    print(table(["task"] + names, rows))

    print(rule("MODELS IN THIS COMPARISON"))
    rows = []
    for payload in payloads:
        info = payload["model"]
        # PEERS is the authority for a peer's pretraining volume, so a results file written before
        # that column existed still renders it rather than showing a dash
        tokens = info.get("training_tokens") or (
            PEERS[info["name"]][1] if info["name"] in PEERS else "--")
        rows.append([
            info["name"],
            info.get("repo", info.get("checkpoint", "--")),
            f"{info.get('params_total', 0) / 1e6:.0f}M" if info.get("params_total") else "--",
            str(info.get("dtype", "--")),
            str(tokens),
            payload.get("timestamp", "--")[:19],
        ])
    print(table(["model", "source", "params", "dtype", "train tokens", "measured"],
                rows, aligns="llllll"))


def print_validation(payload: dict, tolerance: float) -> bool:
    """Compare a peer's measured numbers against the published ones. Returns True if all pass."""
    alias = payload["model"]["name"]
    reference = PUBLISHED.get(alias)
    print(rule(f"HARNESS VALIDATION — {alias} vs published"))
    if not reference:
        print(f"  no published reference recorded for {alias}; validation is only available for "
              f"{', '.join(sorted(PUBLISHED))}")
        return False
    print(f"  source: {PUBLISHED_SOURCE}")
    rows, ok = [], True
    for task_name, metrics in reference.items():
        result = payload["results"].get(task_name)
        if not result:
            rows.append([task_name, "--", "--", "--", "--", "not run"])
            continue
        for metric, expected in metrics.items():
            measured = result.get(metric)
            if measured is None or math.isnan(measured):
                rows.append([task_name, metric, fmt(expected), "--", "--", "not measured"])
                continue
            delta = measured - expected
            scale = 1.0 if metric.startswith("ppl") else 100.0
            passed = abs(delta) * scale <= tolerance
            ok &= passed
            rows.append([task_name, metric, fmt(expected), fmt(measured),
                         f"{delta * scale:+.2f}", "PASS" if passed else "FAIL"])
    print(table(["task", "metric", "published", "measured", "delta", "verdict"], rows))
    print(f"  tolerance: {tolerance:.1f} points of accuracy (ppl compared in absolute units).")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} — the harness "
          f"{'reproduces' if ok else 'does NOT reproduce'} the published numbers.")
    return ok


# ---------------------------------------------------------------------------- model loading


def write_payload(payload: dict, json_out: str, single: bool) -> None:
    """Persist one model's results. One file per model even when several were scored in one
    invocation -- ``--compare`` reads one payload per file, and a list would need a second reader."""
    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    stem, extension = os.path.splitext(json_out)
    path = json_out if single else f"{stem}.{payload['model']['name']}{extension}"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info(f"wrote results to {path}")


def load_tiny_backend(checkpoint_path: str, tokenizer_dir: str, device: str,
                      use_chat: bool) -> TinyBackend:
    """Load a checkpoint from any of this repo's phases and collect its diagnostic block.

    ``sft.save_sft_checkpoint`` writes a strict superset of the pretraining payload, so one reader
    covers pretraining, SFT and repair checkpoints alike.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # shape from the checkpoint, not config.yaml: the pre-reshape checkpoints are this suite's
    # frozen baseline and have to keep scoring after the IR table grows
    model = TinyMoETransformer(**model_params_for_state_dict(state_dict, ModelConfig.Params))
    model = model.to(device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    load_model_state(model, state_dict)
    model.eval()

    template = ChatTemplate(tokenizer) if use_chat else None
    backend = TinyBackend(model, tokenizer, checkpoint_path, device,
                          ModelConfig.Params["max_seq_len"], template)

    losses = checkpoint.get("losses") or []
    params = ModelConfig.Params
    total = sum(p.numel() for p in model.parameters())
    moe_params = sum(p.numel() for p in model.moe.parameters())
    mlp_expert_params = sum(p.numel() for p in model.moe.parallel_experts.parameters())
    active = total - mlp_expert_params + int(mlp_expert_params * params["top_k"] / params["num_mlp_experts"])
    backend.info = {
        "name": os.path.basename(checkpoint_path),
        "checkpoint": os.path.relpath(checkpoint_path, BASE_DIR),
        "checkpoint_mb": round(os.path.getsize(checkpoint_path) / 1e6, 1),
        "saved": datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).isoformat(timespec="seconds"),
        "phase": checkpoint.get("phase", "--"),
        "training_tokens": f"{checkpoint.get('token_count', 0):,}",
        "global_offset": checkpoint.get("global_offset", 0),
        "last_loss": fmt(losses[-1] if losses else None),
        "params_total": total,
        "params_active": active,
        "dtype": "bfloat16",
        "n_loops": params["n_loops"],
        "hidden_size": params["hidden_size"],
        "num_layers": params["num_layers"],
        "experts": f"{params['num_mlp_experts']} mlp / {params['num_attn_experts']} attn / "
                   f"{params['num_ir_experts']} ir, top_k={params['top_k']}",
        "ir_table": f"{params['num_ir_entries']} x {params['ir_dim']}",
        "mtp_extra_tokens": params["mtp_num_extra_tokens"],
        "vocab_size": params["vocab_size"],
        "max_seq_len": params["max_seq_len"],
        "loop_scale": "[" + ", ".join(f"{v:.4f}" for v in model.moe.loop_scale.float().tolist()) + "]",
        "flops_per_token_fwd": f"{model.flops_per_token_fwd / 1e6:.0f}M",
        "body_flops_per_token": f"{model.body_flops_per_token / 1e6:.0f}M",
        "lm_head_flops_per_token": f"{model.lm_head_flops_per_token / 1e6:.0f}M (per loop)",
        "mtp_flops_per_token": f"{model.mtp_flops_per_token / 1e6:.0f}M",
        "tokenizer": os.path.relpath(tokenizer_dir, BASE_DIR),
        "prompt_format": "chat template" if use_chat else "raw completion",
    }
    return backend


# ---------------------------------------------------------------------------- main


def run_suite(backend: Backend, task_names: Sequence[str], args, hf_token: Optional[str]) -> dict:
    """Score one model on every requested task and assemble its results payload."""
    results, revisions = {}, {}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    for name in task_names:
        task = TASKS[name]
        limit = args.gen_limit if task.kind == "gen" else args.limit
        docs, revision, prefix = build_docs(task, args.cache_dir, hf_token, limit, args.seed)
        revisions[name] = revision
        if not docs:
            logger.warning(f"[{name}] no usable documents, skipped")
            continue
        logger.info(f"[{backend.name}] {name}: {len(docs):,} docs "
                    f"({task.kind}, headline metric {task.metric})")
        if task.kind == "mc":
            results[name] = score_mc_task(backend, docs, args.batch_size, args.max_context,
                                          f"{backend.name}/{name}")
        else:
            results[name] = score_gen_task(backend, task, docs, prefix, args.batch_size,
                                           args.max_context, args.pass_k, args.temperature,
                                           f"{backend.name}/{name}")
        results[name]["dataset_revision"] = revision

    peak = (torch.cuda.max_memory_allocated() / 1e9) if args.device.startswith("cuda") else 0.0
    tokens = sum(r.get("tokens", 0) for r in results.values())
    elapsed = time.time() - started
    return {
        "model": {"name": backend.name, **backend.info},
        "results": results,
        "flags": {
            "batch_size": args.batch_size, "max_context": args.max_context,
            "limit": args.limit, "gen_limit": args.gen_limit, "seed": args.seed,
            "pass_k": args.pass_k, "temperature": args.temperature,
        },
        "environment": {
            "device": (torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device),
            "torch": torch.__version__,
            "fp32_matmul": "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee",
            "peak_memory_gb": round(peak, 2),
            "wall_clock": f"{elapsed / 60:.1f} min",
            # exact for the scoring pass; for generation it is prompt length x decode budget, an
            # upper bound that a KV-cached peer never actually pays
            "tokens_forwarded": f"{tokens:,} (generation counted as an upper bound)",
            "throughput": f"{tokens / max(elapsed, 1e-6) / 1e3:.1f}k tok/s",
            "tasks": len(results),
        },
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description="fixed benchmark suite for this model and its peers")
    parser.add_argument("--checkpoint", "-c", default=None, help="local checkpoint to score")
    parser.add_argument("--peer", action="append", default=[],
                        help=f"peer to score; repeatable. one of {', '.join(PEERS)}, 'all', "
                             "or any Hugging Face causal LM repo id")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="previously written results JSONs to fold into the comparison table")
    parser.add_argument("--tasks", default="all",
                        help=f"comma-separated subset of: {', '.join(TASKS)} (also 'mc' / 'gen')")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap multiple-choice docs per task (seeded subsample; default: all)")
    parser.add_argument("--gen-limit", type=int, default=1000,
                        help="cap generative docs per task -- generation has no KV cache on this "
                             "model, so cost is quadratic in answer length")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="part of the measurement: the MoE's grouped GEMM tiles by the batch's "
                             "per-expert row counts, so results are only comparable at a fixed value")
    parser.add_argument("--max-context", type=int, default=1024,
                        help="left-truncate a context longer than this; continuations are never cut")
    parser.add_argument("--pass-k", type=int, default=1,
                        help="sample k generations per question and report pass@k instead of greedy EM")
    parser.add_argument("--temperature", type=float, default=0.8, help="only used when --pass-k > 1")
    parser.add_argument("--peer-dtype", default="float32", choices=("float32", "bfloat16", "float16"),
                        help="peers default to fp32 because the published reference numbers were "
                             "produced there; this model is always bf16")
    parser.add_argument("--chat", action="store_true",
                        help="route generative prompts through the SFT chat template (this model "
                             "only). Off by default -- the frozen suite scores raw completions so "
                             "the peers see the same thing")
    parser.add_argument("--validate", action="store_true",
                        help="check a scored peer against its published numbers")
    parser.add_argument("--tolerance", type=float, default=1.5,
                        help="points of accuracy --validate allows")
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="benchmark parquet cache; refused inside data/prepared*, "
                             "data/datasets or data/archives")
    parser.add_argument("--peer-cache-dir", default=DEFAULT_PEER_CACHE_DIR, help="peer weight cache")
    parser.add_argument("--json-out", default=None, help="write this run's payload here")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.cache_dir = check_cache_dir(args.cache_dir, "cache-dir")
    args.peer_cache_dir = check_cache_dir(args.peer_cache_dir, "peer-cache-dir")
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.tasks == "all":
        task_names = list(TASKS)
    elif args.tasks == "mc":
        task_names = list(MC_TASKS)
    elif args.tasks == "gen":
        task_names = list(GEN_TASKS)
    else:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in task_names if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}. known: {', '.join(TASKS)}")

    peers = []
    for entry in args.peer:
        peers.extend(list(PEERS) if entry == "all" else [e.strip() for e in entry.split(",") if e.strip()])
    if not args.checkpoint and not peers and not args.compare:
        raise SystemExit("nothing to do: pass --checkpoint, --peer, or --compare")

    torch.manual_seed(args.seed)
    # TF32 for the fp32 peers: 10 mantissa bits against bf16's 7, so it is strictly closer to fp32
    # than the alternative precision on the table, and it is what makes an fp32 peer affordable at
    # full-split sizes. It does not touch this model, whose matmuls are bf16.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hf_token = args.hf_token or get_hf_token()
    payloads: List[dict] = []

    # previously frozen runs first, so a comparison table reads oldest-to-newest left to right
    for pattern in args.compare:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
            logger.info(f"loaded frozen results from {path}")

    fresh: List[dict] = []

    def finish(payload: dict) -> None:
        """Report and persist one model as soon as it is scored.

        Deliberately not deferred to the end: a multi-peer run is the better part of an hour, and
        holding every report back until the last model finishes means a crash in peer four throws
        away the readable output of peers one to three along with it.
        """
        fresh.append(payload)
        print_model_diagnostic(payload)
        print_task_report(payload)
        if args.validate:
            print_validation(payload, args.tolerance)
        if args.json_out:
            write_payload(payload, args.json_out, single=len(peers) + bool(args.checkpoint) == 1)

    for alias in peers:
        repo, tokens, description = PEERS.get(alias, (alias, "--", ""))
        logger.info(f"loading peer {alias} ({repo}) into {args.peer_cache_dir}")
        backend = HFBackend(repo, alias, args.device,
                            getattr(torch, args.peer_dtype), args.peer_cache_dir, hf_token)
        backend.info["training_tokens"] = tokens
        backend.info["note"] = description
        finish(run_suite(backend, task_names, args, hf_token))
        del backend
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.checkpoint:
        logger.info(f"loading {args.checkpoint}")
        backend = load_tiny_backend(args.checkpoint, args.tokenizer, args.device, args.chat)
        finish(run_suite(backend, task_names, args, hf_token))

    payloads.extend(fresh)
    if len(payloads) > 1:
        print_comparison(payloads)


if __name__ == "__main__":
    main()
