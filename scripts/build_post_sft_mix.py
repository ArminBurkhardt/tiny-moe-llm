"""Build the post-training (consolidation SFT) mix: chat, evidence QA with every condition, replay.

After the main run the model has to hold a chat turn: answer, abstain when the evidence lacks the
answer, and stop at the end of its turn. R5 (``docs/measurements/r5_bridge_ladder.md``) showed that a
single-format SFT erases every other ability, so the mix carries the pretraining slices beside the
new rows. One evidence-format split (``post_sft_train``, ``post_sft_val``) is merged from slices by
prompt token share, by the existing tools, no builder changed:

    chat     plain conversations through the SFT chat template (``ChatTemplate``: only the assistant
             text and its EOS are supervised): smoltalk2 (the pretraining holdout honoured), ultrachat,
             no_robots. Built here with ``prepare_sft_data.build_corpus``; no evidence, so the merge
             gives each row a zero length evidence segment. default share 0.40
    qa       evidence QA rows with the conditions gold, mixed, distractors and none (the abstention
             rows), the pilot's ``evidence_nomany`` build; the evidence-mode instruction is the
             builder's. default share 0.25
    replay   the pretraining slices at their own structure so nothing is forgotten: the biography
             build with its filler windows (every tier of exposure), the plain chains. default share
             0.35 (pilot profile: bios 0.28 and chains 0.07)

Profiles:

    pilot    the rehearsal on arm P's final: replay from ``data/prepared_pilot/inject_retrieval_train``
             (bios plus filler windows, 600M tokens) and ``data/prepared/chains_train``, QA from
             ``data/prepared/evidence_nomany_*``; output ``data/prepared_post_pilot``
    run2     the real run's consolidation on the PC: QA from a seeded sample of the run 2 corpus's own QA
             build (``qa`` stage: ``prepared_run2/run2_qa_shuf_train``, else
             ``prepared/evidence_run2_train``, val ``prepared/evidence_run2_val``; the sample is cut
             before the driver's ``prune_sources`` deletes the train build, all conditions, web text
             and chat replay arms included, so no new QA build), replay from ``post_replay_{train,val}`` which the ``replay`` stage
             cuts on the machine that holds the run's corpus: a seeded random sample of
             ``run2_main_train`` taken as one slice (the merge tool reads a merged split like any
             builder output, so bios, filler, QA and chains arrive in the pretraining proportions).
             copy that output and ``inject_val`` to the PC; output ``data/prepared_post_run2``

Stages (``python scripts/build_post_sft_mix.py STAGE --profile P``, one command per call, each
resumable by its tool or idempotent): ``plan`` prints the slice table and the merge command; ``chat``
builds the chat slice (network, the Hub; ``--pull-manifest [--manifest-repo R]`` or ``--ignore-holdout`` are forwarded to
the builder, which needs manifest.json for the smoltalk2 holdout hashes); ``qa`` cuts the run2 QA slice (where the corpus is);
``replay`` cuts the run2 replay (where the corpus is); ``merge`` writes the train and val splits;
``verify`` checks the realised shares against the targets and that every assistant turn ends with a
supervised EOS in the built ``.mask`` and that the QA slice's documents come from the QA sources (its ``.src``, skipped with a
note when absent); ``all`` runs chat, merge and verify (pilot).

The EOS check (``check_eos_supervised``): in every document that holds an assistant marker, each
marker is followed by an EOS whose mask byte is 1. A truncated conversation would lose it and teach
the model not to stop; the builders drop over long conversations whole (``--max-doc-tokens`` 4093
here, what the evidence reader keeps). Documents that end on a user turn are counted apart, not failed.

Not built: a ``--qa-abstain-keep`` fraction on the none and distractors conditions. ``merge_evidence_splits.py``
selects whole documents by token share with no per-document filter, so it would take either a condition
filter in the merge tool (a ``.cond`` mask applied in ``select_documents``, with the md5 and resume state
covering it) or a standalone step that rewrites the QA build's bin, idx, mask, ev, evchunk, evidx, evkeyidx,
evgold, cond, ans and src files for the kept documents; the ``qa`` share is the lever until then.

Args (shared):
    --profile: pilot or run2.
    --target-tokens: prompt tokens of the train split (default 150M; the run2 budget is about 400M).
    --val-tokens: prompt tokens of the val split (default 3M; the held-out builds are small, a slice
        that runs out leaves a gap the merge report prints).
    --share LABEL=F: override a default share (repeatable); shares are normalised.
    --out-dir: where the merged splits and the chat build go.
    --data-root: run2 only, the corpus root (``prepared_run2`` inside it).
"""
import os
import sys
import json
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

TRAIN_SPLIT = "post_sft_train"
VAL_SPLIT = "post_sft_val"
CHAT_PREFIX = "postchat"
QA_PREFIX = "postqa"
REPLAY_SPLIT = "post_replay"
MAX_DOC_TOKENS = 4093            # 4096 - 2 mtp pads - 1: what the evidence reader keeps
CHAT_VAL_FRACTION = 0.03
QA_SOURCE_MIN_SHARE = 0.15       # SQuAD plus HotpotQA are 0.35 of the QA build's arm weights; under 0.15 of the documents is far off
CHAT_SOURCE_WEIGHTS = {"smoltalk2": 0.55, "ultrachat": 0.30, "no_robots": 0.15}
DEFAULT_SHARES = {
    "pilot": {"chat": 0.40, "qa": 0.25, "bios": 0.28, "chains": 0.07},
    "run2": {"chat": 0.40, "qa": 0.25, "replay": 0.35},
}


@dataclass
class MixSlice:
    label: str
    train: str            # path prefix of the train build
    val: str              # path prefix of the val build
    share: float
    passes: int = 1


def normalise(shares: Dict[str, float]) -> Dict[str, float]:
    """Shares scaled to sum to 1; every share must be positive."""
    if not shares or any(v <= 0 for v in shares.values()):
        raise ValueError(f"shares must be positive: {shares}")
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def parse_shares(profile: str, overrides: Sequence[str]) -> Dict[str, float]:
    shares = dict(DEFAULT_SHARES[profile])
    for item in overrides:
        label, _, value = item.partition("=")
        if label not in shares:
            raise ValueError(f"--share {item!r}: labels of profile {profile} are {sorted(shares)}")
        shares[label] = float(value)
    return normalise(shares)


def profile_slices(profile: str, out_dir: str, data_root: str, shares: Dict[str, float]) -> List[MixSlice]:
    """The slice table of a profile, in merge order."""
    chat = MixSlice("chat", os.path.join(out_dir, CHAT_PREFIX + "_train"), os.path.join(out_dir, CHAT_PREFIX + "_val"),
                    shares["chat"])
    if profile == "pilot":
        return [
            chat,
            MixSlice("qa", "data/prepared/evidence_nomany_train", "data/prepared/evidence_nomany_val", shares["qa"]),
            MixSlice("bios", "data/prepared_pilot/inject_retrieval_train", "data/prepared_pilot/inject_val", shares["bios"]),
            MixSlice("chains", "data/prepared/chains_train", "data/prepared/chains_val", shares["chains"]),
        ]
    if profile == "run2":
        return [
            chat,
            MixSlice("qa", os.path.join(out_dir, QA_PREFIX + "_train"), os.path.join(out_dir, QA_PREFIX + "_val"), shares["qa"]),
            MixSlice("replay", os.path.join(out_dir, REPLAY_SPLIT + "_train"), os.path.join(out_dir, REPLAY_SPLIT + "_val"),
                     shares["replay"]),
        ]
    raise ValueError(f"unknown profile {profile!r}")


def merge_args(split: str, target_tokens: int, slices: Sequence[MixSlice], which: str, seed: int = 42) -> List[str]:
    """The ``merge_evidence_splits.py`` argument list for the train or val split (``which``)."""
    args = ["--out-dir", "{out}", "--split", split, "--target-tokens", str(int(target_tokens)), "--seed", str(seed),
            "--max-evidence-tokens", "4608", "--overwrite"]
    for s in slices:
        prefix = s.train if which == "train" else s.val
        args += ["--slice", f"{s.label}={prefix}:{s.share:.6f}:{s.passes}"]
    return args


def realised_shares(meta: dict) -> Dict[str, float]:
    """Prompt token share per slice label from a ``merge.json``."""
    total = sum(s["prompt_tokens"] for s in meta["slices"])
    return {s["label"]: s["prompt_tokens"] / max(total, 1) for s in meta["slices"]}


def supervised_token_shares(prefix: str, labels: Sequence[str], chunk_docs: int = 200_000) -> Dict[str, Tuple[int, float]]:
    """Supervised (mask 1) tokens and their share per slice, from a merged split's ``.mask``, ``.idx`` and ``.slice``.

    The share of supervised tokens, not of prompt tokens, sets how hard a slice pulls on the loss when the loss is a plain token
    mean (conversation loss weighting off).

    Args:
        prefix: path prefix of the merged split.
        labels: slice labels in merge order (the ``.slice`` byte is the index into them).
        chunk_docs: documents per pass over the mask.

    Returns:
        ``{label: (supervised tokens, share of all supervised tokens)}``.
    """
    idx = np.fromfile(prefix + ".idx", dtype=np.uint64).astype(np.int64)
    which = np.fromfile(prefix + ".slice", dtype=np.uint8)
    mask = np.memmap(prefix + ".mask", dtype=np.uint8, mode="r")
    counts = np.zeros(len(labels), dtype=np.int64)
    for d0 in range(0, len(which), chunk_docs):
        d1 = min(d0 + chunk_docs, len(which))
        base = idx[d0]
        cs = np.concatenate([[0], np.cumsum(np.asarray(mask[base:idx[d1]]) != 0, dtype=np.int64)])
        per_doc = cs[idx[d0 + 1:d1 + 1] - base] - cs[idx[d0:d1] - base]
        counts += np.bincount(which[d0:d1], weights=per_doc, minlength=len(labels)).astype(np.int64)
    total = max(int(counts.sum()), 1)
    return {label: (int(counts[i]), counts[i] / total) for i, label in enumerate(labels)}


def share_problems(meta: dict, targets: Dict[str, float], tol: float = 0.03) -> List[str]:
    """Slices whose realised share is more than ``tol`` (absolute) from the target."""
    real = realised_shares(meta)
    return [f"{k}: realised {real.get(k, 0.0):.3f}, target {v:.3f}" for k, v in targets.items()
            if abs(real.get(k, 0.0) - v) > tol]


def check_eos_supervised(prefix: str, assistant_id: int, eos_id: int, doc_ids: Optional[Sequence[int]] = None) -> dict:
    """Every assistant turn of the documents ends with an EOS whose loss mask is 1.

    For each document holding at least one assistant marker: every marker is followed by an EOS token
    and that EOS has mask 1 (a truncated turn has none). A document that ends on a user turn (its last
    assistant turn is not the last thing in it) is fine and only counted, in ``ends_non_assistant``.

    Args:
        prefix: path prefix of a ``bin idx mask`` split.
        assistant_id: the ``<assistant>`` marker id (``ChatTemplate.assistant_id``).
        eos_id: the EOS id.
        doc_ids: documents to check (default all).

    Returns:
        ``{"documents", "with_assistant", "bad", "turns", "bad_examples", "ends_non_assistant"}``.
    """
    idx = np.fromfile(prefix + ".idx", dtype=np.uint64).astype(np.int64)
    ids = np.memmap(prefix + ".bin", dtype=np.uint16, mode="r")
    mask = np.memmap(prefix + ".mask", dtype=np.uint8, mode="r")
    docs = range(len(idx) - 1) if doc_ids is None else doc_ids
    n = with_asst = bad = turns = open_end = 0
    examples = []
    for d in docs:
        s, e = int(idx[d]), int(idx[d + 1])
        seg = np.asarray(ids[s:e])
        n += 1
        marks = np.flatnonzero(seg == assistant_id)
        if len(marks) == 0:
            continue
        with_asst += 1
        eos = np.flatnonzero(seg == eos_id)
        ok = True
        last_close = -1
        for p in marks:
            turns += 1
            k = np.searchsorted(eos, p)
            if k >= len(eos) or mask[s + int(eos[k])] != 1:
                ok = False
            else:
                last_close = int(eos[k])
        if not ok:
            bad += 1
            if len(examples) < 5:
                examples.append(int(d))
        elif last_close != len(seg) - 1:
            open_end += 1
    return {"documents": n, "with_assistant": with_asst, "bad": bad, "turns": turns, "bad_examples": examples,
            "ends_non_assistant": open_end}


def qa_source_share(prefix: str, qa_sources: Sequence[int] = (0, 1)) -> Optional[float]:
    """Share of documents of a build whose ``.src`` index is a QA source (``SOURCE_KEYS`` 0 and 1: SQuAD v2, HotpotQA).

    Returns None when the build has no ``.src`` sidecar.
    """
    path = prefix + ".src"
    if not os.path.exists(path):
        return None
    src = np.fromfile(path, dtype=np.uint8)
    return float(np.isin(src, list(qa_sources)).mean()) if len(src) else 0.0


def _run(cmd: List[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


def build_chat(out_dir: str, target_tokens: int, seed: int, extra: Sequence[str] = ()) -> int:
    """Chat slice through ``prepare_sft_data.main`` with a profile of its own (no builder changed).

    ``target_tokens`` counts both splits of the chat build (the builder's rule); the val split takes
    ``CHAT_VAL_FRACTION`` of the conversations. ``extra`` is forwarded to the builder (``--pull-manifest``,
    ``--manifest-repo R``, ``--ignore-holdout``); without one of the first and the last, the builder needs
    ``manifest.json`` with the smoltalk2 holdout hashes in the repo root.
    """
    import scripts.prepare_sft_data as psd
    if "--pull-manifest" not in extra and "--ignore-holdout" not in extra and not os.path.exists(psd.MANIFEST_PATH):
        print(f"manifest.json not found at {psd.MANIFEST_PATH}: the chat build needs its smoltalk2 holdout hashes. "
              "pass --pull-manifest (with --manifest-repo REPO if config has no upload repo) to download it, copy "
              "manifest.json from the pretraining box, or pass --ignore-holdout (unsafe: the chat slice may then hold "
              "conversations the pretraining mix used)", file=sys.stderr)
        return 1
    sources = [s for s in psd.SOURCES if s.key in CHAT_SOURCE_WEIGHTS]
    assert {s.key for s in sources} == set(CHAT_SOURCE_WEIGHTS), "a chat source left prepare_sft_data.SOURCES"
    sources = [psd.replace(s, weight=CHAT_SOURCE_WEIGHTS[s.key]) for s in sources]
    psd.PROFILES[CHAT_PREFIX] = sources
    psd.PROFILE_TARGET_TOKENS[CHAT_PREFIX] = int(target_tokens)
    psd.PROFILE_UNANSWERABLE_FRACTION[CHAT_PREFIX] = 1.0
    sys.argv = ["prepare_sft_data.py", "--profile", CHAT_PREFIX, "--data-dir", out_dir,
                "--target-tokens", str(int(target_tokens)), "--max-doc-tokens", str(MAX_DOC_TOKENS),
                "--val-fraction", str(CHAT_VAL_FRACTION), "--seed", str(seed)] + list(extra)
    psd.main()
    return 0


def cmd_plan(a, slices, shares):
    chat_need = int((a.target_tokens + a.val_tokens) * shares["chat"] * 1.08 / (1 - CHAT_VAL_FRACTION))
    print(f"profile {a.profile}  out {a.out_dir}  train {a.target_tokens:,} val {a.val_tokens:,} prompt tokens")
    for s in slices:
        print(f"  {s.label:<7} share {s.share:.3f}  {s.share * a.target_tokens / 1e6:7.1f}M  train {s.train}  val {s.val}")
    print(f"chat build target (both splits, 8% spare for the drops): {chat_need:,}")
    for which, split, t in (("train", TRAIN_SPLIT, a.target_tokens), ("val", VAL_SPLIT, a.val_tokens)):
        print("python scripts/merge_evidence_splits.py " + " ".join(merge_args(split, t, slices, which, a.seed)).replace("{out}", a.out_dir))
    return 0


def cmd_chat(a, slices, shares):
    need = int((a.target_tokens + a.val_tokens) * shares["chat"] * 1.08 / (1 - CHAT_VAL_FRACTION))
    os.makedirs(a.out_dir, exist_ok=True)
    extra = []
    if a.pull_manifest:
        extra.append("--pull-manifest")
    if a.manifest_repo:
        extra += ["--manifest-repo", a.manifest_repo]
    if a.ignore_holdout:
        extra.append("--ignore-holdout")
    return build_chat(a.out_dir, need, a.seed, extra)


def run2_qa_sources(data_root: str):
    """``(train prefix, val prefix)`` of the run 2 corpus's own QA build, or None when the train build is gone.

    The train side is the seeded permutation ``run2_qa_shuf_train`` (a prefix of it is a random sample)
    or, without it, the builder's ``evidence_run2_train``; both are deleted by the corpus driver's
    ``prune_sources`` stage, so this stage runs on the machine that holds the corpus before that stage.
    The val side is ``evidence_run2_val``, which stays.
    """
    P, R = os.path.join(data_root, "prepared"), os.path.join(data_root, "prepared_run2")
    for train in (os.path.join(R, "run2_qa_shuf_train"), os.path.join(P, "evidence_run2_train")):
        if os.path.exists(train + ".bin") and os.path.exists(train + ".evidx"):
            return train, os.path.join(P, "evidence_run2_val")
    return None


def cmd_qa(a, slices, shares):
    """The QA slice. pilot: nothing to build (``evidence_nomany``). run2: a seeded sample of the run 2 corpus's own QA
    build (all conditions, web text and chat replay arms included), taken by the merge tool like ``cmd_replay``."""
    if a.profile != "run2":
        print("pilot profile: the QA slice is data/prepared/evidence_nomany_*, nothing to build")
        return 0
    src = run2_qa_sources(a.data_root)
    if src is None:
        print(f"no run 2 QA build under {a.data_root}: neither prepared_run2/run2_qa_shuf_train nor prepared/evidence_run2_train "
              "has bin and evidx (the driver's prune_sources stage deleted them, or the corpus is not here). "
              "run this stage on the machine that holds the corpus before prune_sources", file=sys.stderr)
        return 1
    train, val = src
    qa = next(s for s in slices if s.label == "qa")
    n = int(a.target_tokens * qa.share * 1.05)
    nv = int(a.val_tokens * qa.share * 1.05)
    os.makedirs(a.out_dir, exist_ok=True)
    rc = _run([sys.executable, "scripts/merge_evidence_splits.py", "--out-dir", a.out_dir, "--split", QA_PREFIX + "_train",
               "--target-tokens", str(n), "--seed", str(a.seed), "--max-evidence-tokens", "4608", "--overwrite",
               "--slice", f"qa={train}:1.0:1"])
    if rc:
        return rc
    return _run([sys.executable, "scripts/merge_evidence_splits.py", "--out-dir", a.out_dir, "--split", QA_PREFIX + "_val",
                 "--target-tokens", str(nv), "--seed", str(a.seed), "--max-evidence-tokens", "4608", "--overwrite",
                 "--slice", f"qa={val}:1.0:1"])


def cmd_replay(a, slices, shares):
    """run2 only, on the machine that holds the corpus: a seeded sample of the main split as one slice, plus its val."""
    R = os.path.join(a.data_root, "prepared_run2")
    replay = next(s for s in slices if s.label == "replay")
    n = int(a.target_tokens * replay.share * 1.05)
    nv = int(a.val_tokens * replay.share * 1.05)
    os.makedirs(a.out_dir, exist_ok=True)
    rc = _run([sys.executable, "scripts/merge_evidence_splits.py", "--out-dir", a.out_dir, "--split", REPLAY_SPLIT + "_train",
               "--target-tokens", str(n), "--seed", str(a.seed), "--max-evidence-tokens", "4608", "--overwrite",
               "--slice", f"replay={os.path.join(R, 'run2_main_train')}:1.0:1"])
    if rc:
        return rc
    return _run([sys.executable, "scripts/merge_evidence_splits.py", "--out-dir", a.out_dir, "--split", REPLAY_SPLIT + "_val",
                 "--target-tokens", str(nv), "--seed", str(a.seed), "--overwrite", "--no-evidence",
                 "--slice", f"replay={os.path.join(R, 'inject_val')}:1.0:1"])


def cmd_merge(a, slices, shares):
    os.makedirs(a.out_dir, exist_ok=True)
    for which, split, t in (("train", TRAIN_SPLIT, a.target_tokens), ("val", VAL_SPLIT, a.val_tokens)):
        cmd = [sys.executable, "scripts/merge_evidence_splits.py"] + [x.replace("{out}", a.out_dir) for x in merge_args(split, t, slices, which, a.seed)]
        rc = _run(cmd)
        if rc:
            return rc
    return 0


def cmd_verify(a, slices, shares):
    from transformers import AutoTokenizer
    from modules.data.chat import ChatTemplate
    from utils import TOKENIZER_DIR
    tpl = ChatTemplate(AutoTokenizer.from_pretrained(TOKENIZER_DIR))
    failed = False
    for split in (TRAIN_SPLIT, VAL_SPLIT):
        prefix = os.path.join(a.out_dir, split)
        with open(prefix + ".merge.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        real = realised_shares(meta)
        print(f"{split}: {meta['documents']:,} documents, {meta['prompt_tokens']:,} prompt tokens, "
              f"{meta['evidence_tokens']:,} evidence tokens (ratio {meta['evidence_tokens'] / max(meta['prompt_tokens'], 1):.3f})")
        for s in meta["slices"]:
            print(f"  {s['label']:<7} {s['documents']:>9,} docs {s['prompt_tokens']:>12,} tokens  share {real[s['label']]:.3f}"
                  f"  evidence ratio {s['evidence_ratio']:.3f}")
        if os.path.exists(prefix + ".slice") and os.path.exists(prefix + ".mask"):
            sup = supervised_token_shares(prefix, [s["label"] for s in meta["slices"]])
            for s in meta["slices"]:
                n_sup, sup_share = sup[s["label"]]
                print(f"  {s['label']:<7} supervised tokens {n_sup:>12,}  share {sup_share:.3f} (prompt token share {real[s['label']]:.3f}; "
                      f"the supervised share sets the pull on a token mean loss)")
        tgt = normalise({s.label: s.share for s in slices})
        if split == TRAIN_SPLIT:
            problems = share_problems(meta, tgt)
            for p in problems:
                print("  SHARE OFF: " + p)
            failed |= bool(problems)
        r = check_eos_supervised(prefix, tpl.assistant_id, tpl.eos_id)
        print(f"  EOS check: {r['with_assistant']:,} conversations, {r['turns']:,} assistant turns, {r['bad']} without a supervised closing EOS {r['bad_examples']}; "
              f"{r['ends_non_assistant']:,} end on a non-assistant turn (reported, not a failure)")
        failed |= r["bad"] > 0 or r["turns"] == 0
        qa_slice = next((s for s in slices if s.label == "qa"), None)
        if qa_slice is not None and split == TRAIN_SPLIT:
            qa_prefix = qa_slice.train
            qs = qa_source_share(qa_prefix)
            if qs is None:
                print(f"  QA source check skipped: {qa_prefix}.src not found")
            else:
                low = qs < QA_SOURCE_MIN_SHARE
                print(f"  QA slice: {qs:.3f} of its documents come from SQuAD or HotpotQA (floor {QA_SOURCE_MIN_SHARE})"
                      + ("  QA SOURCE SHARE LOW: the slice is mostly chat or web text" if low else ""))
                failed |= low
    print("verify: " + ("FAILED" if failed else "ok"))
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "chat", "qa", "replay", "merge", "verify", "all"])
    ap.add_argument("--profile", choices=sorted(DEFAULT_SHARES), default="pilot")
    ap.add_argument("--target-tokens", type=int, default=150_000_000)
    ap.add_argument("--val-tokens", type=int, default=3_000_000)
    ap.add_argument("--share", action="append", default=[], metavar="LABEL=F")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--data-root", default="data/run2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pull-manifest", action="store_true", help="chat: download manifest.json (smoltalk2 holdout hashes) first")
    ap.add_argument("--manifest-repo", default=None, help="chat: repo for --pull-manifest")
    ap.add_argument("--ignore-holdout", action="store_true", help="chat: build without the holdout hashes (unsafe)")
    a = ap.parse_args()
    a.out_dir = a.out_dir or {"pilot": "data/prepared_post_pilot", "run2": "data/prepared_post_run2"}[a.profile]
    shares = parse_shares(a.profile, a.share)
    slices = profile_slices(a.profile, a.out_dir, a.data_root, shares)
    stages = {"plan": cmd_plan, "chat": cmd_chat, "qa": cmd_qa, "replay": cmd_replay, "merge": cmd_merge, "verify": cmd_verify}
    if a.stage == "all":
        for st in ("chat", "merge", "verify"):
            rc = stages[st](a, slices, shares)
            if rc:
                sys.exit(rc)
        return
    sys.exit(stages[a.stage](a, slices, shares))


if __name__ == "__main__":
    main()
