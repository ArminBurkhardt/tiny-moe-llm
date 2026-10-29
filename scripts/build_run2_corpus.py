"""Build the Run 2 corpus with the existing builders: the pilot's three merges at scale.

The recipe is the pilot's (``ckpts/inject/build_pilot_bios.sh``, ``merge_pilot.sh``,
``merge_pilot2.sh``) scaled to the token target of ``docs/plans/RUN2.md``, with
``--support-gold-only`` on the biography builds, the written bridge chains beside the plain ones, QA
at its finite passes and the filler from ``prepare_data.py``. No builder is changed here: the things
the builders lack (a fineweb prefix covering more than one crawl, source weights that respect the
supply, a hop mix with 4-hop training rows, files already used by an earlier phase, a mirror for a
cluster without internet) are done by patching the imported module in ``wrap-prepare-data``,
``wrap-chains`` and ``wrap-hub`` below, in memory, before its ``main`` runs.

The derivation of every number is ``derive()``; ``python scripts/build_run2_corpus.py plan`` prints
it as tables (shares, the phase 1 mix, the disk schedule with its peak, the hours per stage, the
transfer estimate) and ``docs/plans/RUN2_corpus.md`` carries them with their sources.

The tail split (``run2_tail_train``, docs/plans/RUN2_corpus.md section 2b, RUN2.md row 16) is the run's last 5%: 42B prompt
tokens for the 40B tokens from 760B, on a higher quality mix, ended by the trainer at the plan's last step
(``sft.py --end-at-plan``), so the cosine, fitted to the 800B plan on the main split, is unchanged.

The warm-up and control splits are prefixes of the main split in merge order: the main merge defines
the document order and ``--order-prefix-from`` replays it until the smaller split reaches its token
count, so the biography builds for the warm-up and the control are byte prefixes of a build (the
injection builder's ``--stop-after-tokens``). A switch from the warm-up to the main split then
continues the same document stream (``run2_main_train.order.json`` tells the trainer how many
documents the warm-up covered).

Stages, in order (``--stage NAME[,NAME]`` runs some, default all):

    filler         prepare_data: phase1, phase2, ir (the manifest key rule: ir needs its own key), then the tail's
                   plain text as a fourth build into DATA_ROOT/prepared_tail (fresh files only, the higher quality mix)
    qa             prepare_evidence_data: SQuAD v2, HotpotQA, web text, chat replay
    bios_main      prepare_injection_data: arm retrieval, swap 0.15, gold-only support; writes the
                   shared outputs (inject_val, the card store)
    val_clean      prepare_clean_val: inject_val_clean (needs inject_val, so after bios_main)
    bios_full      prepare_injection_data: arm full, a prefix for the control
    bios_warm      prepare_injection_data: arm retrieval, swap 1.0, a prefix for the warm-up
    tail_text      the tail's plain text cut into windows of at most 1000 tokens with BOS, EOS and a mask (numpy)
    prune_filler   delete the phase1, phase2, ir and tail phase bins once qa, val_clean, the three bios builds and
                   tail_text are done
    chains_plain   prepare_chain_data: hops 1 / 2 / 3 at 25 / 50 / 25
    chains_bridge  prepare_chain_data --bridge: hops 1 to 4 at 25 each
    qa_shuffle     merge_evidence_splits over the qa build alone: its documents in a seeded permutation
                   (the builder's storage order is not random), the copy every later stage reads
    order          the document order of the whole run, one uint8 per document (run2_order.slice), numpy only
    merge_main     merge_evidence_splits --order-prefix-from the order file, over the whole order
    merge_warm     merge_evidence_splits --order-prefix-from main until warm_max tokens; then the epoch 0 order sidecar
    merge_control  merge_evidence_splits --no-evidence --order-prefix-from main until the control's tokens
    prune_bios     delete the three biography builds and the raw QA build once the main, warm and control merges and the
                   QA copy exist (the disk peak: the tail merge is written after this)
    merge_tail     merge_evidence_splits, five slices (tail text, a seeded sample of the main split, the QA copy with its
                   chat arms, both chain builds), its own seed: the tail split read for the run's last 40B tokens
    rows           sft.py --estimate-rows-only on the main, control and tail splits (CPU): the packed row counts
                   that set total_steps (main) and check the tail; the main one needs the order sidecar merge_warm writes
    prune_sources  delete the chain train builds, the shuffled QA copy and the tail text once all four merges pass
    heldout_qa     prepare_evidence_data --heldout: evidence_dev, evidence_fixed
    heldout_chains prepare_chain_data: chains_eval, chains_heldout_tmpl, chains_hop4
    store_openqa   build_store: the A4 / A7 store; store_edit: its edit twin

A stage prints each command, runs it through bash (``cd ROOT && source ENV_INIT && ...``, the way
``tests/run_tests.sh`` does, with ``TINY_LLM_ROOT`` and ``TINY_LLM_ENV_INIT`` read from the
environment) in its own process group, tees it to ``DATA_ROOT/run2_logs/STAGE.log``, checks its
expected outputs and writes a marker ``DATA_ROOT/run2_done/STAGE.json``. A stage with a marker and
its outputs present (or pruned by a later prune stage) is skipped (``--force`` reruns it), so a rerun
after a failure resumes at the first unfinished stage.

What a rerun does to a stage without a marker depends on its tool (``RESUMABLE``): a tool that resumes
from its own state (filler, qa, the biography builds, the two chain builds, the three merges) is run
again as it is (a finished split prints "already complete" and exits 0), the
merges with ``--overwrite`` (a valid merge state resumes, none starts clean), and a stage whose tool
finished earlier (outputs there, state file gone) is only checked; a tool that cannot resume (val_clean,
order, rows, the held out builds, the stores) has its partial outputs deleted first
(``Stage.clean``). A stop that lands after a stage's work is done (the child exited 0, or its outputs and
completion records are there and its state is gone) still gets the checks and the marker before the stop
is honoured. A stage that prints nothing for ``HEARTBEAT_SECONDS`` prints a heartbeat line.

Exit codes of ``build``: 0 the requested stages are done, 1 a stage failed, 2 refused before running
(disk plan over budget, filler preflight), 20 interrupted by SIGTERM (the signal is forwarded to the
running child's process group, the child is waited for, no marker is written; rerun the same command).

Usage:

    python scripts/build_run2_corpus.py plan [--scale S]
    python scripts/build_run2_corpus.py build --data-root DIR [--dry-run] [--scale S] [--stage A,B]
    python scripts/build_run2_corpus.py wrap-prepare-data [opts] -- PREPARE_DATA_ARGS
    python scripts/build_run2_corpus.py wrap-chains [--hops 4] -- PREPARE_CHAIN_DATA_ARGS
    python scripts/build_run2_corpus.py wrap-hub --stage-dir DIR -- scripts/SCRIPT.py ARGS

Cluster (day 1, ``TINY_LLM_ENV_INIT=/dev/null`` as the rented box has no env_init; with no internet
on the compute nodes add ``--mirror-inbox`` / ``--mirror-listing`` / ``--offline-stage``, see
``scripts/mirror_sources.py``):

    TINY_LLM_ROOT=/workspace/tiny-llm TINY_LLM_ENV_INIT=/dev/null HF_TOKEN=... \\
        python scripts/build_run2_corpus.py build --data-root /data/run2 --device cuda
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

REPO = os.environ.get("TINY_LLM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_INIT = os.environ.get("TINY_LLM_ENV_INIT", "env_init")

# the file set an evidence format split owns (scripts/prepare_evidence_data.EVIDENCE_SUFFIXES)
EVIDENCE_FILES = ("bin", "idx", "mask", "ev", "evidx", "evchunk", "evkey", "evkeyidx", "evgold", "cond", "ans")
PLAIN_FILES = ("bin", "idx", "mask")

# measured on the pilot builds (ckpts/inject build and merge logs); every use below says which
PLAIN_TOKENS_PER_QUESTION = 104_663_146 / 2_000_000      # chains_train.mask bytes over its 2M questions
# the bridge at hops 1 to 4 at 25 each, measured on the smoke build (31,180 questions of
# chains_bridge_run2_train, 2026-10-26); the pilot's hops 1 to 3 form was 67.35 tokens and 12.03 chunks
BRIDGE_TOKENS_PER_QUESTION = 2_351_504 / 31_180
PLAIN_CHUNKS_PER_TOKEN = 22_987_422 / 99_999_982         # pilot_warm_train.merge.json, chains slice
PLAIN_EVTOK_PER_TOKEN = 419_556_724 / 99_999_982
BRIDGE_CHUNKS_PER_QUESTION = 393_040 / 31_180
BRIDGE_EVTOK_PER_QUESTION = 7_269_687 / 31_180
# the biography evidence is on the renderings only (162,000 of them at the pilot tier structure), not
# on every token: 648,000 chunk occurrences and 30.0M evidence tokens in the pilot's bios slice
BIOS_CHUNKS_PER_RENDER = 648_000 / 162_000
BIOS_EVTOK_PER_RENDER = 0.05006 * 599_638_781 / 162_000
QA_CHUNKS_PER_TOKEN = 1_480_692 / 210_607_046
QA_EVTOK_PER_TOKEN = 0.9624
KEY_BYTES = 768          # fp16 bge-small key per chunk occurrence
TOK_BYTES = 2            # uint16 per token, prompt or evidence (ev and evchunk each)
PLAIN_DOC_BYTES = 8      # idx
EVIDENCE_DOC_BYTES = 8 + 8 + 1 + 1       # evidx, evkeyidx, cond, ans
MERGE_DOC_BYTES = 1      # the .slice file
TOKENS_PER_DOC = {"bios": 616.0, "qa": 349.0, "plain": PLAIN_TOKENS_PER_QUESTION,
                  "bridge": BRIDGE_TOKENS_PER_QUESTION}   # pilot documents per token
# rates (update these when a builder is timed at scale)
# the merge tool's prefix mode reads every slice sequentially: 37k documents per second with evidence, 118k
# without (measured on a local disk with a warm cache; a parallel file system may be slower, update when timed)
MERGE_DOCS_PER_SECOND = 37_000
MERGE_DOCS_PER_SECOND_PLAIN = 118_000
TAIL_TEXT_TOKENS_PER_SECOND = 10e6                      # assumed: the windowing copy of the tail's plain text, numpy, one process, sequential disk
ORDER_DOCS_PER_SECOND = 400_000                          # assumed: one stable argsort of float64 keys, about 20 minutes at 480M documents
INJECT_HOURS_PER_B = 1e9 / 2.07e6 / 3600                 # the landed builder: about 2.07M tokens per second, one process (plus 10 to 15 minutes of setup per start)
CHAIN_HOURS_PER_B = (55 / 60) / 0.10466                  # 2M questions in about 55 minutes on cuda
FILLER_SCRATCH_BYTES = 30e9    # shards in flight during the filler build, one per source
SMALL_STAGE_BYTES = {"val_clean": 0.1e9, "heldout_qa": 1e9, "heldout_chains": 0.5e9, "store_openqa": 0.2e9,
                     "store_edit": 0.1e9, "bios_shared": 0.5e9}   # assumed, small
BYTES_PER_FILLER_TOKEN = 1.5   # parquet and zstd shard bytes per token; an estimate until mirror_sources list reads sizes
FILLER_IDX_BYTES_PER_TOKEN = 0.01
# Per stage: does the tool continue from its own state after a kill (True: the driver reruns the command and
# the tool picks up; the merges always get --overwrite, which resumes with a valid state and starts clean
# without one), or must the driver delete the stage's partial outputs and run it again from the start (False).
RESUMABLE = {"filler": True, "qa": True, "qa_shuffle": True, "bios_main": True, "bios_full": True, "bios_warm": True,
             "merge_main": True, "merge_warm": True, "merge_control": True, "merge_tail": True, "tail_text": False,
             "chains_plain": True, "chains_bridge": True,
             "val_clean": False, "order": False, "rows": False,
             "heldout_qa": False, "heldout_chains": False, "store_openqa": False, "store_edit": False}
CHAINS_JSONL_BYTES = 1400      # per question, the .chains.jsonl sidecar (about 40 GB at the full plan); not measured
HEARTBEAT_SECONDS = 300.0      # a silent stage prints a line this often so the job's no growth watchdog stays quiet
LONG_STAGE_HOURS = 60.0
PREFIX_MARGIN = 0.05
SUPPLY_CAP = 0.90
# per source share of the Hub's supply the plan may use; the shortfall goes to the receivers in order
SOURCE_CAPS = {"code_edu": 0.95, "wikipedia": 0.90}
PHASE1_RECEIVERS = (("nemotron_math", 0.65), ("fineweb", SUPPLY_CAP))
WEBTEXT_PROMPT_PER_IR_TOKEN = 60_092_611 / 149_321_284   # the pilot QA build: ir tokens read per prompt token
QA_ARM_WEIGHTS = {"squad_v2": 0.20, "hotpot_qa": 0.15, "webtext": 0.40, "smoltalk2": 0.12,
                  "ultrachat": 0.08, "no_robots": 0.05}
QA_ARM_PER_PASS = {"squad_v2": 5_862_565, "hotpot_qa": 4_861_098}     # prompt tokens, pilot state file
QA_ARM_SUPPLY = {"ultrachat": 242e6, "no_robots": 5.3e6}               # ultrachat: 1,163 tokens x 207k rows
WINDOW_TOKENS = 721.0         # filler tokens per 4096 window after the short tail windows are dropped (the pilot's count)
SMOLTALK2_SUPPLY = 1.8e9     # 24 no_think SFT files at about 75M tokens (the 16B manifest: 675M in 9 files)

# the tail phase (decided 2026-10-28, docs/plans/RUN2.md row 16): the last 5% of the run, 40B tokens from 760B, on a higher
# quality mix that is built with the corpus. The split is sized 5% over the 40B it is read for, so the run reaches its
# planned step count on the cosine's floor before the split ends (sft.py --end-at-plan stops it there).
TAIL_TOKENS = 42e9
TAIL_READ = 40e9              # what the run reads of it: 800B minus the 760B stop
# prompt token shares of the tail merge, by slice. "main" is a seeded sample of the main split's documents (biographies in
# their main form: swap 0.15, gold only support, with the filler windows they were built with); "qa" is the evidence
# QA build, whose chat arms (smoltalk2, ultrachat, no_robots) are a third of its tokens, so 0.15 of the tail is about 0.10
# QA and 0.05 chat; the chains are the plain and bridge builds
TAIL_SHARES = {"text": 0.65, "main": 0.12, "qa": 0.15, "chains": 0.04, "bridge": 0.04}
TAIL_QA_PASSES = 5            # the qa build is read up to this many times (it holds 1.5B tokens, the tail takes 6.3B: 4.1)
TAIL_CHAIN_PASSES = 2         # the chain builds (1.4B each) are read up to this many times (the tail takes 1.7B each: 1.2)
TAIL_SEED = 43                # the tail merge's seed: its permutations differ from the main run's
# the plain text of the tail by source, higher quality first. Fresh files only: the tail phase skips every file the filler's
# three phases used (--exclude-files-of), so a source is held at its remaining supply (TAIL_CAP of the Hub's files) and the
# shortfall goes to the receivers in order. code_edu and wikipedia are almost used up by the main filler, so they get what
# is left, not what they weigh here
TAIL_WEIGHTS = {"dclm": 0.38, "finepdfs": 0.25, "nemotron_math": 0.14, "code_edu": 0.08, "wikipedia": 0.03,
                "fineweb": 0.12}
TAIL_RECEIVERS = (("dclm", 0.99), ("finepdfs", 0.99), ("fineweb", 0.99))
TAIL_CAP = 0.99
TAIL_TEXT_FLOOR_MARGIN = 0.005   # the tail-text stage fails only below the merge's need times this (document granularity of the merge's slice)
TAIL_TEXT_MARGIN = 0.05       # the plain text slice is built this much over its share (the merge takes a slice only up to its target; prepare_data does not move an exhausted source's share)

# phase source weights as in scripts/prepare_data.SOURCES (raw, renormalized per phase at run time)
PHASE_WEIGHTS = {
    "phase1": {"fineweb": 0.55, "dclm": 0.10, "finepdfs": 0.07, "code_edu": 0.12, "nemotron_math": 0.03,
               "wikipedia": 0.03},
    "phase2": {"fineweb": 0.15, "finepdfs": 0.10, "code_edu": 0.22, "nemotron_math": 0.30,
               "wikipedia": 0.08, "smoltalk2": 0.15},
    "ir": {"fineweb": 0.124, "finepdfs": 0.082, "code_edu": 0.181, "nemotron_math": 0.247,
           "wikipedia": 0.066, "smoltalk2": 0.30},
}
# tokens per file (the 16B manifest: realized tokens over files_used) and files on the Hub
SOURCE_SUPPLY = {
    "fineweb": {"per_file": 296e6, "files_default": 50, "files_all_crawls": 2410},
    "dclm": {"per_file": 109e6, "files_default": 2790, "files_all_crawls": 2790},
    "finepdfs": {"per_file": 0.99e9, "files_default": 100, "files_all_crawls": 100},
    "code_edu": {"per_file": 283e6, "files_default": 95, "files_all_crawls": 95},
    "nemotron_math": {"per_file": 0.7e9, "files_default": 46, "files_all_crawls": 46},
    "wikipedia": {"per_file": 121e6, "files_default": 41, "files_all_crawls": 41},
    "smoltalk2": {"per_file": 75e6, "files_default": 24, "files_all_crawls": 24},
}


# the evidence trainer stops at num_epochs times the split's rows and ignores target_tokens, so the
# split is sized as the token target over the epoch count: one derivation
TOKEN_TARGET = 800e9
EPOCHS = 3


@dataclass
class Plan:
    """Every free number of the corpus, each with its source in docs/plans/RUN2_corpus.md.

    Args:
        total: prompt tokens of the main split, TOKEN_TARGET / EPOCHS (the trainer runs whole epochs).
        run_tokens: tokens the run trains on, so epochs = run_tokens / total.
        warm_max: the most the warm-up is read for before the stop rule (28B).
        control: prompt tokens of the short matched control (10B).
        chain_tokens: absolute prompt tokens of the chains, plain and bridge together (2.8B: every
            chain prompt token carries about 177 bytes of chunk keys, so a share of the split would
            grow the disk with the corpus).
        chain_share: when set, the chains are this share of the split instead of ``chain_tokens``.
        bridge_fraction: the bridge's part of the chains (assumption).
        qa_target: the QA builder's prompt token target (assumption, see the sheet).
        phase2_tokens: prepare_data phase 2 target; ir_tokens: its ir target (assumptions, smoltalk2
            supply is the limit, see the sheet).
        margin: extra filler built over the need (windows set aside for validation, the filler repeat).
        persons_per_tier: the biography tier structure (the pilot's tiers at 3x the persons: 10,800 persons, 486,000 renderings, about 1.7x lower sigma on the leak read).
        tail_tokens: prompt tokens of the tail split (42B); tail_read: what the run reads of it (40B).
        tail_shares: the tail's slice shares (TAIL_SHARES); tail_weights: its plain text source weights (TAIL_WEIGHTS).
        tail_qa_passes, tail_chain_passes: the most times the tail merge reads the QA and the chain builds.
        scale: multiplies every token quantity above (smoke runs).
    """
    total: float = TOKEN_TARGET / EPOCHS
    run_tokens: float = TOKEN_TARGET
    warm_max: float = 28e9
    control: float = 10e9
    chain_tokens: float = 2.8e9
    chain_share: Optional[float] = None
    bridge_fraction: float = 0.5
    qa_target: float = 2.5e9
    phase2_tokens: float = 5e9
    ir_tokens: float = 2.5e9
    margin: float = 0.01
    persons_per_tier: str = "1000:300,100:1500,10:3000,1:6000"
    tail_tokens: float = TAIL_TOKENS
    tail_read: float = TAIL_READ
    tail_shares: Dict[str, float] = field(default_factory=lambda: dict(TAIL_SHARES))
    tail_weights: Dict[str, float] = field(default_factory=lambda: dict(TAIL_WEIGHTS))
    tail_qa_passes: int = TAIL_QA_PASSES
    tail_chain_passes: int = TAIL_CHAIN_PASSES
    scale: float = 1.0
    support_gold_only: bool = True
    val_windows: int = 2000
    seed: int = 42

    def scaled(self, name: str) -> float:
        return getattr(self, name) * self.scale


def parse_tiers(text: str) -> Dict[int, int]:
    return {int(a): int(b) for a, b in (part.split(":") for part in text.split(","))}


def qa_realized(plan: Plan) -> Dict[str, float]:
    """Prompt tokens each QA arm can supply at the builder's target, an estimate from the pilot state.

    Each arm stops at ``weight * target`` or at its supply, whichever is first; the build ends when
    every arm is done. SQuAD and HotpotQA repeat up to ``--max-source-epochs`` (2).
    """
    n = plan.scaled("qa_target")
    s = plan.scale
    ir = plan.scaled("ir_tokens")
    p2 = plan.scaled("phase2_tokens")
    smol_used = PHASE_WEIGHTS["phase2"]["smoltalk2"] / sum(PHASE_WEIGHTS["phase2"].values()) * p2
    smol_used += PHASE_WEIGHTS["ir"]["smoltalk2"] / sum(PHASE_WEIGHTS["ir"].values()) * ir
    supply = {
        "squad_v2": QA_ARM_PER_PASS["squad_v2"] * 2,
        "hotpot_qa": QA_ARM_PER_PASS["hotpot_qa"] * 2,
        "webtext": ir * WEBTEXT_PROMPT_PER_IR_TOKEN,
        "smoltalk2": max(0.0, SMOLTALK2_SUPPLY * s - smol_used),
        "ultrachat": QA_ARM_SUPPLY["ultrachat"] * s,
        "no_robots": QA_ARM_SUPPLY["no_robots"] * s,
    }
    return {k: min(QA_ARM_WEIGHTS[k] * n, supply[k]) for k in QA_ARM_WEIGHTS}


def physical_supply(src: str) -> float:
    """Tokens the Hub holds for a source with fineweb's prefix widened to every crawl (not scaled)."""
    sup = SOURCE_SUPPLY[src]
    return sup["files_all_crawls"] * sup["per_file"]


def cap_phase1(phase1: float, p2_need: Dict[str, float], ir_need: Dict[str, float],
               caps: Optional[Dict[str, float]] = None, receivers: Sequence = PHASE1_RECEIVERS):
    """Phase 1 token allocation per source with each source held at its share of the supply.

    A source whose need over the three phases exceeds its cap of the tokens the Hub holds (``caps``,
    ``SUPPLY_CAP`` by default) is capped there (phase 1 gets what phase 2 and ir leave under the cap).
    The shortfall goes to the ``receivers`` in order, each up to its own cap (nemotron_math to 65%
    of its supply, then fineweb), within a receiver in no case past its cap.

    Returns:
        ``(weights, rows)``: normalized phase 1 weights, and a report row per source.
    """
    caps = dict(SOURCE_CAPS if caps is None else caps)
    raw = PHASE_WEIGHTS["phase1"]
    tot = sum(raw.values())
    w0 = {s: v / tot for s, v in raw.items()}
    alloc = {s: w0[s] * phase1 for s in w0}
    names = [r[0] for r in receivers]
    limit = {s: max(0.0, caps.get(s, SUPPLY_CAP) * physical_supply(s) - p2_need.get(s, 0.0) - ir_need.get(s, 0.0))
             for s in w0}
    capped = set()
    excess = 0.0
    ceilings = {s: min(limit[s], cap * physical_supply(s) - p2_need.get(s, 0.0) - ir_need.get(s, 0.0))
                for s, cap in receivers}
    for s in alloc:
        top = ceilings.get(s, limit[s])
        if alloc[s] > top + 1e-6:
            # a receiver already over its own ceiling is cut back too, not only the plain sources
            excess += alloc[s] - top
            alloc[s] = max(0.0, top)
            capped.add(s)
    for s, cap in receivers:
        ceiling = ceilings[s]
        take = min(max(0.0, ceiling - alloc[s]), excess)
        alloc[s] += take
        excess -= take
        if alloc[s] >= ceiling - 1e-6:
            capped.add(s)
    if excess > 1e-6:
        raise ValueError("every receiver is at its supply cap: the plan needs more filler than the Hub holds")
    weights = {s: alloc[s] / phase1 for s in alloc}
    rows = [{"source": s, "before_share": w0[s], "after_share": weights[s], "before_tokens": w0[s] * phase1,
             "after_tokens": alloc[s], "limit": limit[s], "capped": s in capped} for s in w0]
    return weights, rows


def cap_tail(tail_filler: float, used: Dict[str, float], weights: Optional[Dict[str, float]] = None,
             cap: float = TAIL_CAP, receivers: Sequence = TAIL_RECEIVERS):
    """The tail phase's token allocation per source, each held at what the earlier three phases left of the Hub.

    The tail phase skips every file the filler's phases used, so a source can only give the part of its supply
    (``cap`` of the files the Hub holds) that phase 1, phase 2 and ir did not take (``used``). The shortfall of a
    capped source goes to the ``receivers`` in order, each up to its own remaining supply.

    Returns:
        ``(weights, rows)``: normalized weights for the tail phase, and a report row per source.
    """
    weights = dict(TAIL_WEIGHTS if weights is None else weights)
    tot = sum(weights.values())
    alloc = {s: v / tot * tail_filler for s, v in weights.items()}
    limit = {s: max(0.0, cap * physical_supply(s) - used.get(s, 0.0)) for s in alloc}
    before = dict(alloc)
    excess, capped = 0.0, set()
    for s in alloc:
        if alloc[s] > limit[s] + 1e-6:
            excess += alloc[s] - limit[s]
            alloc[s] = limit[s]
            capped.add(s)
    for s, _ in receivers:
        if s not in alloc:
            continue
        take = min(max(0.0, limit[s] - alloc[s]), excess)
        alloc[s] += take
        excess -= take
    if excess > 1e-6:
        raise ValueError("every receiver is at its supply cap: the tail's plain text needs more fresh files than the Hub holds")
    out = {s: (v / tail_filler if tail_filler else 0.0) for s, v in alloc.items()}
    rows = [{"source": s, "before_tokens": before[s], "after_tokens": alloc[s], "limit": limit[s], "capped": s in capped}
            for s in alloc]
    return out, rows


def derive(plan: Plan) -> dict:
    """Every target of the pipeline from the plan, as one dict of plain numbers."""
    total = plan.scaled("total")
    chains = plan.chain_share * total if plan.chain_share is not None else plan.scaled("chain_tokens")
    bridge = chains * plan.bridge_fraction
    plain = chains - bridge
    qa_arms = qa_realized(plan)
    qa = sum(qa_arms.values())
    bios = total - qa - chains
    margin = 1.0 + plan.margin
    phase2 = plan.scaled("phase2_tokens")
    ir = plan.scaled("ir_tokens")
    phase1 = bios * margin - phase2
    if phase1 <= 0:
        raise ValueError("phase2 is larger than the filler need; lower phase2_tokens")
    p2w, irw = PHASE_WEIGHTS["phase2"], PHASE_WEIGHTS["ir"]
    p2_need = {s: v / sum(p2w.values()) * phase2 for s, v in p2w.items()}
    ir_need = {s: v / sum(irw.values()) * ir for s, v in irw.items()}
    p1_weights, p1_rows = cap_phase1(phase1, p2_need, ir_need)
    plain_q = plain / PLAIN_TOKENS_PER_QUESTION
    bridge_q = bridge / BRIDGE_TOKENS_PER_QUESTION
    tiers = parse_tiers(plan.persons_per_tier)
    bio_renders = sum(t * c for t, c in tiers.items())
    out = {
        "total": total, "epochs": plan.run_tokens / plan.total, "warm_max": plan.scaled("warm_max"),
        "control": plan.scaled("control"), "bios": bios, "qa": qa, "qa_arms": qa_arms,
        "qa_target": plan.scaled("qa_target"), "chains": chains, "plain": plain, "bridge": bridge,
        "phase1": phase1, "phase2": phase2, "ir": ir,
        "phase1_weights": p1_weights, "phase1_rows": p1_rows,
        "plain_questions": int(plain_q * 1.05), "bridge_questions": int(bridge_q * 1.05),
        "bio_renders": bio_renders, "persons": sum(tiers.values()),
        "shares": {"bios": bios / total, "qa": qa / total, "chains": plain / total,
                   "bridge": bridge / total},
    }
    # the injection builder keeps a window when crc32 % 1e6 < int(fraction * 1e6): the fraction is a whole
    # number of millionths, rounded so the realized count is closest to val_windows (it cannot be finer), and
    # the half millionth keeps int() off the float rounding edge. The filler windows are the phase 1 and 2 bins.
    windows = max(1.0, (phase1 + phase2) / WINDOW_TOKENS)
    cutoff = max(1, min(5000, int(round(plan.val_windows / windows * 1e6))))
    out["val_cutoff"] = cutoff
    out["val_fraction"] = (cutoff + 0.5) / 1e6
    out["val_windows_expected"] = cutoff / 1e6 * windows
    # the bios slice is taken whole (share above its size, one pass): the warm-up and main builds differ
    # in token length (swaps), and the prefix merges replay the main order document for document
    out["merge_shares"] = dict(out["shares"], bios=out["shares"]["bios"] * 1.02)
    out["injection_target"] = int(bios)
    # a prefix of the merge order that holds X tokens holds at most X biography tokens; the margin
    # covers the other arm's token length (swaps, the full form)
    out["bios_stop_warm"] = int(min(out["injection_target"], out["warm_max"] * (1 + PREFIX_MARGIN)))
    out["bios_stop_control"] = int(min(out["injection_target"], out["control"] * (1 + PREFIX_MARGIN)))
    out["qa_val_fraction"] = 0.001
    # the tail split: slice tokens, the plain text build behind its share and its source allocation
    tail = plan.scaled("tail_tokens")
    sh = dict(plan.tail_shares)
    if abs(sum(sh.values()) - 1.0) > 1e-6:
        raise ValueError(f"tail_shares sum to {sum(sh.values()):.4f}, not 1")
    text_target = int(sh["text"] * tail * (1.0 + TAIL_TEXT_MARGIN))
    tail_filler = text_target * margin
    text_need = int(sh["text"] * tail * (1.0 + TAIL_TEXT_FLOOR_MARGIN))
    # every earlier phase marks its partly used last file as used and the tail phase skips it, so each phase's use
    # counts in whole files
    def whole_files(src: str, tokens: float) -> float:
        per_file = SOURCE_SUPPLY[src]["per_file"] * plan.scale
        return math.ceil(tokens / per_file - 1e-9) * per_file if tokens > 0 else 0.0

    used = {r["source"]: whole_files(r["source"], r["after_tokens"]) + whole_files(r["source"], p2_need.get(r["source"], 0.0))
            + whole_files(r["source"], ir_need.get(r["source"], 0.0)) for r in p1_rows}
    tail_w, tail_rows = cap_tail(tail_filler, used, plan.tail_weights)
    out["tail"] = {
        "tokens": tail, "read": plan.scaled("tail_read"), "shares": sh,
        "slice_tokens": {k: sh[k] * tail for k in sh},
        "text_target": text_target, "text_need": text_need, "filler": tail_filler, "weights": tail_w, "rows": tail_rows,
        "qa_passes": sh["qa"] * tail / qa if qa else 0.0, "plain_passes": sh["chains"] * tail / plain if plain else 0.0,
        "bridge_passes": sh["bridge"] * tail / bridge if bridge else 0.0,
        "main_fraction": sh["main"] * tail / total,
    }
    return out


def filler_need_by_phase(d: dict) -> Dict[str, Dict[str, float]]:
    """Tokens each source must supply in each filler phase, phase 1 at the capped weights, then the tail phase."""
    out = {}
    for phase in ("phase1", "phase2", "ir"):
        w = d["phase1_weights"] if phase == "phase1" else PHASE_WEIGHTS[phase]
        tot = sum(w.values())
        out[phase] = {s: v / tot * d[phase] for s, v in w.items()}
    tail = d["tail"]["weights"]
    out["late"] = {s: v * d["tail"]["filler"] for s, v in tail.items()}      # the tail phase, after ir (listing phase name "late")
    return out


def filler_supply(plan: Plan, d: dict) -> List[dict]:
    """Per source, tokens needed across the three phases against what the Hub holds."""
    rows = []
    by_phase = filler_need_by_phase(d)
    for src, sup in SOURCE_SUPPLY.items():
        need = sum(by_phase[ph].get(src, 0.0) for ph in by_phase)
        s = plan.scale
        rows.append({"source": src, "need": need, "per_file": sup["per_file"],
                     "files_needed": int(-(-need // sup["per_file"])) if need else 0,
                     "supply_default": sup["files_default"] * sup["per_file"] * s,
                     "supply_patched": sup["files_all_crawls"] * sup["per_file"] * s})
    return rows


def filler_file_need(plan: Plan, d: dict) -> Dict[str, dict]:
    """The input of ``mirror_sources list``: per source the tokens per file and the need by phase."""
    by_phase = filler_need_by_phase(d)
    return {src: {"per_file": sup["per_file"], "need": {ph: by_phase[ph].get(src, 0.0) for ph in by_phase}}
            for src, sup in SOURCE_SUPPLY.items()}


def transfer_estimate(plan: Plan, d: dict, margin: float = 0.10) -> Dict[str, dict]:
    """Files and bytes the mirror would push, from the per-file token yields (bytes are an estimate)."""
    out = {}
    for src, row in filler_file_need(plan, d).items():
        need = sum(row["need"].values())
        files = min(SOURCE_SUPPLY[src]["files_all_crawls"], int(math.ceil(need * (1 + margin) / row["per_file"])) + 1) if need else 0
        out[src] = {"files": files, "bytes": files * row["per_file"] * BYTES_PER_FILLER_TOKEN}
    return out


# --------------------------------------------------------------------------- disk and hours

def build_sizes(plan: Plan, d: dict) -> Dict[str, float]:
    """Bytes of every build and merge. Biography evidence is per rendering, everything else per token."""
    def plain(tokens, docs):
        return tokens * (TOK_BYTES + 1) + docs * PLAIN_DOC_BYTES

    def evidence(docs, chunks, evtok):
        return docs * EVIDENCE_DOC_BYTES + chunks * (KEY_BYTES + 1) + evtok * 2 * TOK_BYTES

    renders = d["bio_renders"]
    bios_t = float(d["injection_target"])
    bios_docs = bios_t / TOKENS_PER_DOC["bios"]
    bios_chunks, bios_evtok = renders * BIOS_CHUNKS_PER_RENDER, renders * BIOS_EVTOK_PER_RENDER
    qa_t = d["qa"]
    qa_docs = qa_t / TOKENS_PER_DOC["qa"]
    pl_t, br_t = d["plain"], d["bridge"]
    pl_docs, br_docs = d["plain_questions"], d["bridge_questions"]
    pl_ch, pl_ev = pl_t * PLAIN_CHUNKS_PER_TOKEN, pl_t * PLAIN_EVTOK_PER_TOKEN
    br_ch, br_ev = br_docs * BRIDGE_CHUNKS_PER_QUESTION, br_docs * BRIDGE_EVTOK_PER_QUESTION
    qa_ch, qa_ev = qa_t * QA_CHUNKS_PER_TOKEN, qa_t * QA_EVTOK_PER_TOKEN
    s: Dict[str, float] = {}
    s["filler"] = (d["phase1"] + d["phase2"] + d["ir"]) * (TOK_BYTES + FILLER_IDX_BYTES_PER_TOKEN)
    s["bios_main"] = plain(bios_t, bios_docs) + evidence(bios_docs, bios_chunks, bios_evtok)
    stop_c, stop_w = d["bios_stop_control"], d["bios_stop_warm"]
    s["bios_full"] = plain(stop_c, stop_c / TOKENS_PER_DOC["bios"])
    fw = stop_w / bios_t
    s["bios_warm"] = (plain(stop_w, stop_w / TOKENS_PER_DOC["bios"])
                      + evidence(stop_w / TOKENS_PER_DOC["bios"], bios_chunks * fw, bios_evtok * fw))
    s["qa"] = plain(qa_t, qa_docs) + evidence(qa_docs, qa_ch, qa_ev)
    s["chains_plain"] = plain(pl_t, pl_docs) + evidence(pl_docs, pl_ch, pl_ev) + pl_docs * CHAINS_JSONL_BYTES
    s["chains_bridge"] = plain(br_t, br_docs) + evidence(br_docs, br_ch, br_ev) + br_docs * CHAINS_JSONL_BYTES
    merged = (s["bios_main"] + s["qa"] + s["chains_plain"] + s["chains_bridge"]
              + (bios_docs + qa_docs + pl_docs + br_docs) * MERGE_DOC_BYTES)
    s["merge_main"] = merged
    s["merge_warm"] = merged * d["warm_max"] / d["total"]
    docs = bios_docs + qa_docs + pl_docs + br_docs
    s["merge_control"] = (d["control"] * (TOK_BYTES + 1) + docs * d["control"] / d["total"] * (PLAIN_DOC_BYTES + MERGE_DOC_BYTES))
    s["docs_main"] = docs
    s["order"] = docs
    s["ev_tokens"] = bios_evtok + qa_ev + pl_ev + br_ev
    s["ev_ratio"] = s["ev_tokens"] / d["total"]
    # the tail: its plain text phase (bin and idx), the windowed copy (bin, idx, mask) and the merge. Per token bytes
    # of each slice come from the main run's own estimates above
    t = d["tail"]
    tt = t["slice_tokens"]
    s["filler_tail"] = t["filler"] * (TOK_BYTES + FILLER_IDX_BYTES_PER_TOKEN)
    text_docs = t["text_target"] / WINDOW_TOKENS
    s["tail_text"] = plain(t["text_target"], text_docs)
    chain_bytes = lambda total, docs: total - docs * CHAINS_JSONL_BYTES
    per_tok = {
        "main": merged / d["total"],
        "qa": s["qa"] / qa_t if qa_t else 0.0,
        "chains": chain_bytes(s["chains_plain"], pl_docs) / pl_t if pl_t else 0.0,
        "bridge": chain_bytes(s["chains_bridge"], br_docs) / br_t if br_t else 0.0,
    }
    s["merge_tail"] = (plain(tt["text"], tt["text"] / WINDOW_TOKENS) + sum(tt[k] * per_tok[k] for k in per_tok)
                       + tt["main"] / TOKENS_PER_DOC["bios"] * MERGE_DOC_BYTES)
    tail_docs = (tt["text"] / WINDOW_TOKENS + tt["main"] / TOKENS_PER_DOC["bios"] + tt["qa"] / TOKENS_PER_DOC["qa"]
                 + tt["chains"] / TOKENS_PER_DOC["plain"] + tt["bridge"] / TOKENS_PER_DOC["bridge"])
    s["docs_tail"] = tail_docs
    tail_ev = (tt["qa"] * QA_EVTOK_PER_TOKEN + tt["chains"] * PLAIN_EVTOK_PER_TOKEN
               + tt["bridge"] / TOKENS_PER_DOC["bridge"] * BRIDGE_EVTOK_PER_QUESTION
               + tt["main"] * s["ev_tokens"] / d["total"])
    s["tail_ev_tokens"] = tail_ev
    s["tail_ev_ratio"] = tail_ev / t["tokens"] if t["tokens"] else 0.0
    return s


def disk_schedule(plan: Plan, d: dict, window_bytes: Optional[float] = None,
                  parallel: Sequence[str] = ()) -> dict:
    """Bytes written and deleted per stage in run order, the running total, the peak and the standing size."""
    s = build_sizes(plan, d)
    small = SMALL_STAGE_BYTES
    scratch = window_bytes if window_bytes else FILLER_SCRATCH_BYTES
    bios_sources = s["bios_main"] + s["bios_full"] + s["bios_warm"] + s["qa"]
    sources = s["qa"] + s["chains_plain"] + s["chains_bridge"] + s["tail_text"]
    steps = [
        ("filler", s["filler"] + s["filler_tail"], 0.0, scratch), ("qa", s["qa"], 0.0, 0.0),
        ("bios_main", s["bios_main"] + small["bios_shared"], 0.0, 0.0), ("val_clean", small["val_clean"], 0.0, 0.0),
        ("bios_full", s["bios_full"], 0.0, 0.0), ("bios_warm", s["bios_warm"], 0.0, 0.0),
        ("tail_text", s["tail_text"], 0.0, 0.0),
        ("prune_filler", 0.0, s["filler"] + s["filler_tail"], 0.0),
        ("chains_plain", s["chains_plain"], 0.0, 0.0), ("chains_bridge", s["chains_bridge"], 0.0, 0.0),
        ("qa_shuffle", s["qa"], 0.0, 0.0),
        ("order", s["order"], 0.0, 0.0), ("merge_main", s["merge_main"], 0.0, 0.0), ("merge_warm", s["merge_warm"], 0.0, 0.0),
        ("merge_control", s["merge_control"], 0.0, 0.0),
        ("prune_bios", 0.0, bios_sources, 0.0), ("merge_tail", s["merge_tail"], 0.0, 0.0), ("rows", 1e6, 0.0, 0.0),
        ("prune_sources", 0.0, sources, 0.0),
        ("heldout_qa", small["heldout_qa"], 0.0, 0.0), ("heldout_chains", small["heldout_chains"], 0.0, 0.0),
        ("store_openqa", small["store_openqa"], 0.0, 0.0), ("store_edit", small["store_edit"], 0.0, 0.0),
    ]
    group = [st for st in steps if st[0] in parallel]
    if len(group) > 1:
        first = next(i for i, st in enumerate(steps) if st[0] in parallel)
        merged = ("+".join(st[0] for st in group), sum(st[1] for st in group), sum(st[2] for st in group),
                  sum(st[3] for st in group))
        rest = [st for st in steps if st[0] not in parallel]
        steps = rest[:first] + [merged] + rest[first:]
    rows, running, peak = [], 0.0, 0.0
    for name, w, dele, transient in steps:
        at = running + w + transient
        peak = max(peak, at)
        running = running + w - dele
        rows.append({"stage": name, "writes": w, "deletes": dele, "transient": transient, "peak_during": at,
                     "running": running})
    return {"rows": rows, "peak": peak, "standing": running, "sizes": s}


def docs_qa(d: dict) -> float:
    return d["qa"] / TOKENS_PER_DOC["qa"]


def hours_plan(plan: Plan, d: dict) -> Dict[str, Optional[float]]:
    """One process hours per stage from the pilot rates above (None: not measured)."""
    s = build_sizes(plan, d)
    hours: Dict[str, Optional[float]] = {
        "filler": None, "qa": None,
        "bios_main": d["injection_target"] / 1e9 * INJECT_HOURS_PER_B,
        "val_clean": None,
        "bios_full": d["bios_stop_control"] / 1e9 * INJECT_HOURS_PER_B,
        "bios_warm": d["bios_stop_warm"] / 1e9 * INJECT_HOURS_PER_B,
        "chains_plain": d["plain"] / 1e9 * CHAIN_HOURS_PER_B,
        "chains_bridge": d["bridge"] / 1e9 * CHAIN_HOURS_PER_B,
    }
    docs = s["docs_main"]
    hours["qa_shuffle"] = docs_qa(d) / MERGE_DOCS_PER_SECOND / 3600
    hours["order"] = docs / ORDER_DOCS_PER_SECOND / 3600
    hours["merge_main"] = docs / MERGE_DOCS_PER_SECOND / 3600
    hours["merge_warm"] = hours["merge_main"] * d["warm_max"] / d["total"]
    hours["merge_control"] = docs * d["control"] / d["total"] / MERGE_DOCS_PER_SECOND_PLAIN / 3600
    hours["tail_text"] = d["tail"]["text_target"] / TAIL_TEXT_TOKENS_PER_SECOND / 3600
    hours["merge_tail"] = s["docs_tail"] / MERGE_DOCS_PER_SECOND / 3600
    return hours


def disk_problems(sched: dict, budget: float, standing_budget: float) -> List[str]:
    problems = []
    if sched["peak"] > budget:
        problems.append(f"disk peak {sched['peak'] / 1e12:.2f} TB is over the budget {budget / 1e12:.2f} TB")
    if sched["standing"] > standing_budget:
        problems.append(f"standing size {sched['standing'] / 1e12:.2f} TB is over the budget "
                        f"{standing_budget / 1e12:.2f} TB")
    return problems


def long_stage_flags(hours: Dict[str, Optional[float]]) -> List[str]:
    """Stages over LONG_STAGE_HOURS that cannot resume after a kill (a 72 hour job would lose them)."""
    return [k for k, v in hours.items() if v is not None and v > LONG_STAGE_HOURS and not RESUMABLE.get(k, False)]


# --------------------------------------------------------------------------- commands

def q(x) -> str:
    return shlex.quote(str(x))


@dataclass
class Cfg:
    data_root: str
    plan: Plan
    device: str = "cuda"
    filler_dir: Optional[str] = None
    fineweb_prefix: str = "data/"
    heldout_max_questions: Optional[int] = None
    store_max_questions: int = 3000
    chain_eval_per_hop: int = 1000
    persons_per_tier: Optional[str] = None
    link_filler_from: Optional[str] = None
    mirror_inbox: Optional[str] = None
    mirror_listing: Optional[str] = None
    offline_stage: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    mirror_window_gb: float = 120.0
    allow_filler_repeat: bool = False
    disk_budget: float = 3.4e12
    standing_budget: float = 2.6e12

    @property
    def prepared(self) -> str:
        return os.path.join(self.data_root, "prepared")

    @property
    def run2(self) -> str:
        return os.path.join(self.data_root, "prepared_run2")

    @property
    def index(self) -> str:
        return os.path.join(self.data_root, "index")

    @property
    def logs(self) -> str:
        return os.path.join(self.data_root, "run2_logs")

    @property
    def done(self) -> str:
        return os.path.join(self.data_root, "run2_done")

    @property
    def tail_dir(self) -> str:
        """Where the tail's plain text phase is built (the same phase machinery as the filler, its own state files)."""
        return os.path.join(self.data_root, "prepared_tail")

    @property
    def bios_store(self) -> str:
        return os.path.join(self.index, "run2_bios")


ORDER_NAME = "run2_order"
MAIN_SUFFIX = "s15a50g1"
WARM_SUFFIX = "s100a50g1"
QA_PREFIX = "evidence_run2"
PLAIN_PREFIX = "chains_run2"
BRIDGE_PREFIX = "chains_bridge_run2"
WARM_SPLIT = "run2_warm_train"
MAIN_SPLIT = "run2_main_train"
CTL_SPLIT = "run2_control_train"
TAIL_SPLIT = "run2_tail_train"
TAIL_TEXT = "tail_text"
TAIL_LABELS = ("text", "main", "qa", "chains", "bridge")
SLICE_LABELS = ("bios", "qa", "chains", "bridge")
# slices whose documents the order stage must not read in storage order: the QA builder draws its rows by
# smooth weighted round robin over its arms, so a storage prefix holds the finite extractive sources
# (SQuAD, HotpotQA, no_robots) at about 3x the run's share, and the warm-up and control are prefixes. Each
# listed slice is first rewritten by a one slice merge (a seeded permutation of its documents, label and
# seed fixing it) and every later stage reads the permuted copy. Bios and chains are shuffled by their
# builders already.
SHUFFLED_SLICES = {"qa": "run2_qa_shuf_train"}


def share(v: float) -> str:
    return f"{v:.6f}"


def slice_prefixes(cfg: Cfg, bios_split: str) -> List[tuple]:
    """``(label, path prefix)`` of the four slices as the order and merge stages read them."""
    if "qa" in SHUFFLED_SLICES:
        qa = os.path.join(cfg.run2, SHUFFLED_SLICES["qa"])
    else:
        qa = os.path.join(cfg.prepared, QA_PREFIX + "_train")
    return [("bios", os.path.join(cfg.run2, bios_split)), ("qa", qa),
            ("chains", os.path.join(cfg.prepared, PLAIN_PREFIX + "_train")),
            ("bridge", os.path.join(cfg.prepared, BRIDGE_PREFIX + "_train"))]


def merge_slices(cfg: Cfg, d: dict, bios_split: str) -> List[str]:
    """The four ``--slice`` arguments; the order is fixed because the slice file numbers them."""
    shares = {"bios": d["merge_shares"]["bios"], "qa": d["shares"]["qa"], "chains": d["shares"]["chains"],
              "bridge": d["shares"]["bridge"]}
    return [f"{label}={prefix}:{share(shares[label])}:1" for label, prefix in slice_prefixes(cfg, bios_split)]


def split_files(prefix: str, suffixes: Sequence[str]) -> List[str]:
    return [f"{prefix}.{s}" for s in suffixes]


@dataclass
class Stage:
    name: str
    commands: List[str]
    outputs: List[str]
    note: str = ""
    requires: List[str] = field(default_factory=list)     # stages whose markers must exist first
    pruned_by: Optional[str] = None                        # a prune stage that may delete the outputs
    prune: List[str] = field(default_factory=list)         # files this stage deletes
    check: Optional[str] = None                            # merge stage name for check_merge
    post: List[Callable] = field(default_factory=list)     # idempotent hooks, run after success and on skip
    state_file: Optional[str] = None   # a resumable tool's state file: outputs present and this gone means it finished
    restart: bool = False              # the tool cannot resume: partial outputs are deleted before a rerun (RESUMABLE)
    clean: List[str] = field(default_factory=list)   # glob patterns a restart deletes; the outputs when empty
    needs_files: List[str] = field(default_factory=list)   # files that must exist before the stage runs
    pre: List[Callable] = field(default_factory=list)      # hooks run before the commands (not on a skip)


ENV_STAGES = ("rows", "filler", "qa", "bios_main", "val_clean", "bios_full", "bios_warm", "tail_text", "chains_plain", "chains_bridge",
              "heldout_qa", "heldout_chains", "store_openqa", "store_edit")


ROWS_WORLD_SIZES = "1,2,4,8,16"  # every rank count a launch may use (the 1 and 2 GPU check, the fallbacks): the trainer exits 2 without one
ROWS_NUM_DATA_WORKERS = 4       # sft.NUM_DATA_WORKERS: the cache is keyed by workers = this times the rank count


def rows_check(cfg: Cfg) -> List[str]:
    """Each rows estimate json exists and holds an entry for every rank count in ROWS_WORLD_SIZES."""
    problems = []
    found = {}
    for split in (MAIN_SPLIT, CTL_SPLIT, TAIL_SPLIT):
        path = os.path.join(cfg.run2, split + ".rows_estimate.json")
        try:
            with open(path) as f:
                rows = json.load(f).get("rows") or {}
        except (OSError, ValueError, AttributeError) as e:
            problems.append(f"{path} unreadable: {e}")
            continue
        found[split] = rows
        for w in ROWS_WORLD_SIZES.split(","):
            if str(ROWS_NUM_DATA_WORKERS * int(w)) not in rows:
                problems.append(f"{path} has no entry for {w} ranks")
    # the tail must hold the rows of the last stretch of the run (the stop at tail_read tokens before the planned end),
    # or the split would end before the cosine reaches its floor
    d = derive(cfg.plan)
    if MAIN_SPLIT in found and TAIL_SPLIT in found:
        share = cfg.plan.tail_read / cfg.plan.run_tokens
        for key, main_rows in found[MAIN_SPLIT].items():
            tail_rows = found[TAIL_SPLIT].get(key)
            if tail_rows is None:
                continue
            need = share * d["epochs"] * float(main_rows)
            if float(tail_rows) < need:
                problems.append(f"{TAIL_SPLIT} has {float(tail_rows):,.0f} packed rows at {key} workers, the last "
                                f"{share:.1%} of the run needs about {need:,.0f}: the split would end before the plan does")
    return problems


def offline_tokenizer(cfg: Cfg) -> Optional[str]:
    """The tokenizer directory the builders read offline: ``--tokenizer-dir``, else the staged copy.

    The builders take it from ``utils.TOKENIZER_DIR`` (``TINY_LLM_TOKENIZER``), so ``offline_env`` exports
    exactly this path and ``check-offline`` checks exactly it.
    """
    if cfg.tokenizer_dir:
        return cfg.tokenizer_dir
    return os.path.join(cfg.offline_stage, "tokenizer") if cfg.offline_stage else None


def offline_env(cfg: Cfg) -> str:
    """Environment for every stage that could reach the Hub: no network, models from the staged cache."""
    if not (cfg.offline_stage or cfg.mirror_inbox):
        return ""
    home = f"HF_HOME={q(cfg.offline_stage)} " if cfg.offline_stage else ""
    tok = offline_tokenizer(cfg)
    tok_env = f"TINY_LLM_TOKENIZER={q(tok)} " if tok else ""
    return f"{home}HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 {tok_env}"


def offline_report(cfg: Cfg, names: Sequence[str], tokenizer_dir: Optional[str] = None) -> List[str]:
    """Print what each requested stage needs from the offline cache and return the problems."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mirror_sources as ms
    tok = tokenizer_dir or offline_tokenizer(cfg)
    rows = ms.check_stages(names, cfg.offline_stage, tok, cfg.mirror_listing, cfg.mirror_inbox)
    problems = []
    for name, needs in rows.items():
        print(f"  {name:16s} " + "  ".join(f"{n}:{'ok' if ok else 'MISSING'}" for n, ok, _ in needs))
        for n, ok, why in needs:
            if not ok:
                problems.append(f"stage {name} needs {n}: {why}")
    return problems


def build_stages(cfg: Cfg) -> List[Stage]:
    plan = cfg.plan
    d = derive(plan)
    P, R, I = cfg.prepared, cfg.run2, cfg.index
    py = "python scripts/"
    tiers = cfg.persons_per_tier or plan.persons_per_tier
    filler_dir = cfg.filler_dir or P
    gold = " --support-gold-only" if plan.support_gold_only else ""
    me = "python scripts/build_run2_corpus.py"
    stages: List[Stage] = []

    mirror = ""
    if cfg.mirror_inbox:
        mirror = f" --mirror-inbox {q(cfg.mirror_inbox)} --mirror-listing {q(cfg.mirror_listing or '')}"
    weights = ",".join(f"{k}={v:.6f}" for k, v in d["phase1_weights"].items())
    base = f"{me} wrap-prepare-data --fineweb-prefix {q(cfg.fineweb_prefix)} --data-root {q(cfg.data_root)}{mirror}"
    p1 = (f"{base} --phase1-weights {weights} -- "
          f"--data-dir {q(P)} --phases phase1 --phase1-tokens {int(d['phase1'])} "
          f"--manifest-key data_prep --seed {plan.seed}")
    p2 = (f"{base} --exclude-files-of phase1 -- "
          f"--data-dir {q(P)} --phases phase2 --phase2-tokens {int(d['phase2'])} "
          f"--manifest-key data_prep --seed {plan.seed}")
    ir = (f"{base} --exclude-files-of phase1,phase2 -- "
          f"--data-dir {q(P)} --phases ir --ir-tokens {int(d['ir'])} --manifest-key ir_prep --seed {plan.seed}")
    # the tail's plain text: one more phase 1 style build into its own directory (own state files, own bins), from files
    # none of the three phases used, at the higher quality source weights of cap_tail. It runs as the filler's fourth
    # command so the mirror's listing, its consumption order and its end mark cover it with no other change
    tail_w = ",".join(f"{k}={v:.6f}" for k, v in d["tail"]["weights"].items())
    late = (f"{base} --exclude-files-of phase1,phase2,ir --phase1-weights {tail_w} -- "
            f"--data-dir {q(cfg.tail_dir)} --phases phase1 --phase1-tokens {int(d['tail']['filler'])} "
            f"--manifest-key tail_prep --seed {plan.seed}")
    filler_outputs = [os.path.join(P, f"{ph}.{s}") for ph in ("phase1", "phase2", "ir") for s in ("bin", "idx")]
    tail_filler_outputs = [os.path.join(cfg.tail_dir, f"phase1.{s}") for s in ("bin", "idx")]
    filler_outputs += tail_filler_outputs
    stages.append(Stage("filler", [p1, p2, ir, late], filler_outputs,
                        "prepare_data x4: phase 1, 2, ir, then the tail's plain text (fresh files, higher quality mix); "
                        "HF_TOKEN needed for the gated nemotron_math (or the mirror)",
                        pruned_by="prune_filler",
                        pre=[clear_inbox_done] if cfg.mirror_inbox else [],
                        post=[write_inbox_done] if cfg.mirror_inbox else []))

    def offline(script_and_args: str) -> str:
        if not cfg.offline_stage:
            return f"{py}{script_and_args}"
        stage_dir = os.path.join(cfg.offline_stage, "datasets")
        return f"{me} wrap-hub --stage-dir {q(stage_dir)} -- scripts/{script_and_args}"

    held = f" --heldout-max-questions {cfg.heldout_max_questions}" if cfg.heldout_max_questions else ""
    qa = offline(f"prepare_evidence_data.py --data-dir {q(P)} --target-tokens {int(d['qa_target'])} "
                 f"--max-evidence-tokens 4608 --max-source-epochs 2 --split-prefix {QA_PREFIX} --webtext-phase ir "
                 f"--val-fraction {d['qa_val_fraction']} --seed {plan.seed} --device {cfg.device}")
    stages.append(Stage("qa", [qa], split_files(os.path.join(P, QA_PREFIX + "_train"), EVIDENCE_FILES),
                        "reads the ir bin (web text) and manifest.json data_prep.smoltalk2_holdout_hashes (written by the "
                        "phase2 build)", requires=["filler"], pruned_by="prune_bios", post=[qa_arm_check]))

    def inject(arms: str, swap: float, suffix: str, stop: Optional[int] = None, shared: bool = True) -> str:
        extra = f" --stop-after-tokens {stop}" if stop else ""
        extra += "" if shared else " --no-shared-outputs"
        return (f"{py}prepare_injection_data.py --out-dir {q(R)} --filler-dir {q(filler_dir)} "
                f"--filler-phases phase1,phase2 --store-dir {q(cfg.bios_store)} "
                f"--target-tokens {d['injection_target']} --seq-length 4096 --swap-rate {swap} --anon-rate 0.5 "
                f"--gold-drop-rate 0.2 --distractors 3 --persons-per-tier {tiers} "
                f"--val-fraction {d['val_fraction']:.8f} --seed {plan.seed} --arms {arms} "
                f"--suffix {suffix}{gold} --no-train-factspan{extra} --device {cfg.device}")
    full_name = f"inject_full_{MAIN_SUFFIX}_train"
    main_name = f"inject_retrieval_{MAIN_SUFFIX}_train"
    warm_name = f"inject_retrieval_{WARM_SUFFIX}_train"
    val = [os.path.join(R, f"inject_val.{s}") for s in ("bin", "idx", "mask")]
    stages.append(Stage("bios_main", [inject("retrieval", 0.15, MAIN_SUFFIX)],
                        split_files(os.path.join(R, main_name), EVIDENCE_FILES) + val,
                        "swap 0.15, anonymization 0.5, gold card only supports a span; writes inject_val and the card store",
                        requires=["filler"], pruned_by="prune_bios", post=[filler_repeat_check]))

    val_state = os.path.join(P, "_prepare_state_" + QA_PREFIX + ".json")
    stages.append(Stage("val_clean", [f"{py}prepare_clean_val.py --out-dir {q(R)} --filler-dir {q(filler_dir)} "
                                      f"--filler-phases phase1,phase2 --seed {plan.seed} "
                                      f"--val-fraction {d['val_fraction']:.8f} "
                                      f"--exclude-evidence-state {q(val_state)}"],
                        split_files(os.path.join(R, "inject_val_clean"), ("bin", "idx", "mask", "factspan")),
                        "inject_val windows whose documents the QA builder never read; reads the filler bins, the QA "
                        "state file and inject_val.idx (the size to match)", requires=["filler", "qa", "bios_main"]))
    stages.append(Stage("bios_full", [inject("full", 0.15, MAIN_SUFFIX, d["bios_stop_control"], False)],
                        split_files(os.path.join(R, full_name), PLAIN_FILES),
                        f"a prefix of the full form build, {d['bios_stop_control']:,} tokens: the control's biographies",
                        requires=["filler", "bios_main"], pruned_by="prune_bios", post=[filler_repeat_check]))
    stages.append(Stage("bios_warm", [inject("retrieval", 1.0, WARM_SUFFIX, d["bios_stop_warm"], False)],
                        split_files(os.path.join(R, warm_name), EVIDENCE_FILES),
                        f"swap 1.0, a prefix of {d['bios_stop_warm']:,} tokens: the warm-up form",
                        requires=["filler", "bios_main"], pruned_by="prune_bios", post=[filler_repeat_check]))

    tail_text_prefix = os.path.join(cfg.tail_dir, TAIL_TEXT)
    repeat = " --allow-repeat" if (cfg.allow_filler_repeat or cfg.link_filler_from) else ""
    stages.append(Stage("tail_text", [f"{me} tail-text --src {q(os.path.join(cfg.tail_dir, 'phase1'))} "
                                      f"--out {q(tail_text_prefix)} --target {d['tail']['text_target']} "
                                      f"--min-tokens {d['tail']['text_need']}{repeat}"],
                        split_files(tail_text_prefix, PLAIN_FILES),
                        "the tail's plain text cut into windows of at most 1000 tokens with BOS and EOS, the way the "
                        "biography build cuts its filler (numpy; the merge permutes the documents); the source of the tail "
                        "merge's text slice", requires=["filler"], pruned_by="prune_sources"))

    pruned_filler = list(filler_outputs)
    stages.append(Stage("prune_filler", [f"rm -f {' '.join(q(p) for p in pruned_filler)}"], [],
                        "the filler bins are only read by qa, val_clean, the bios builds and tail_text; the manifest and the "
                        "state files stay", requires=["filler", "qa", "val_clean", "bios_main", "bios_full", "bios_warm",
                                                      "tail_text"],
                        prune=pruned_filler))

    def chain(prefix: str, questions: int, bridge: bool) -> str:
        args = (f"--out-dir {q(P)} --prefix {prefix} --splits train,val --train-questions {questions} "
                f"--val-questions 2000 --seed {plan.seed} --device {cfg.device}")
        if bridge:
            return f"{me} wrap-chains --hops 4 -- {args} --bridge"
        return f"{py}prepare_chain_data.py {args}"
    stages.append(Stage("chains_plain", [chain(PLAIN_PREFIX, d["plain_questions"], False)],
                        split_files(os.path.join(P, PLAIN_PREFIX + "_train"), EVIDENCE_FILES), "hops 1 / 2 / 3 at 25 / 50 / 25",
                        pruned_by="prune_sources"))
    stages.append(Stage("chains_bridge", [chain(BRIDGE_PREFIX, d["bridge_questions"], True)],
                        split_files(os.path.join(P, BRIDGE_PREFIX + "_train"), EVIDENCE_FILES),
                        "hops 1 to 4 at 25 each, the written bridge", pruned_by="prune_sources"))

    def merge(split: str, target, slices: List[str], extra: str) -> str:
        sl = " ".join(f"--slice {q(s)}" for s in slices)
        target = target if isinstance(target, str) else int(target)
        # --overwrite always: with a valid merge state the tool resumes, without one it starts clean, and a
        # complete merge never reaches here (the driver sees no state file and only checks it)
        return (f"{py}merge_evidence_splits.py --out-dir {q(R)} --split {split} --target-tokens {target} "
                f"--seed {plan.seed} --max-evidence-tokens 4608 --overwrite {extra}{sl}").replace("  ", " ")
    main_slice = os.path.join(R, MAIN_SPLIT + ".slice")
    prefix_flag = f"--order-prefix-from {q(main_slice)} "
    shuf = {label: os.path.join(R, split) for label, split in SHUFFLED_SLICES.items()}
    qa_train = os.path.join(P, QA_PREFIX + "_train")
    if "qa" in shuf:
        # one slice at share 1 with a token target above its size takes every document in the merge tool's
        # seeded permutation of (seed, label, count); a prefix of the result is a random sample of the build
        stages.append(Stage("qa_shuffle", [
            f"{py}merge_evidence_splits.py --out-dir {q(R)} --split {SHUFFLED_SLICES['qa']} "
            f"--target-tokens $({me} split-tokens --prefix {q(qa_train)} --plus 1) --seed {plan.seed} "
            f"--max-evidence-tokens 4608 --overwrite --slice {q('qa=' + qa_train + ':1.0:1')}"],
            split_files(shuf["qa"], EVIDENCE_FILES + ("slice", "merge.json")),
            "the QA build rewritten in a seeded document permutation (the builder's storage order is not random: "
            "its finite sources run out in the first third); every later stage reads this copy",
            requires=["qa"], pruned_by="prune_sources", post=[qa_shuffle_check],
            state_file=os.path.join(R, SHUFFLED_SLICES["qa"] + ".merge_state.json")))
    merge_in = ["bios_main", "qa_shuffle" if "qa" in shuf else "qa", "chains_plain", "chains_bridge"]
    order_slice, order_info = os.path.join(R, ORDER_NAME + ".slice"), os.path.join(R, ORDER_NAME + ".json")
    order_args = " ".join(f"--slice {q(l + '=' + pth)}" for l, pth in slice_prefixes(cfg, main_name))
    stages.append(Stage("order", [f"{me} order --out {q(order_slice)} --info {q(order_info)} {order_args}"],
                        [order_slice, order_info],
                        "every document of every slice once, slices interleaved in proportion to their size, storage "
                        "order within a slice (bios and chains are shuffled by their builders, qa by qa_shuffle); "
                        "numpy, about 9 GB of memory",
                        requires=merge_in))
    total = f"$({me} order-total --info {q(order_info)})"
    stages.append(Stage("merge_main", [merge(MAIN_SPLIT, total, merge_slices(cfg, d, main_name),
                                             f"--order-prefix-from {q(order_slice)} ")],
                        split_files(os.path.join(R, MAIN_SPLIT), EVIDENCE_FILES + ("slice", "merge.json")),
                        "the whole order file: the run's document order, the same one every prefix merge replays",
                        requires=["order"], check="merge_main", state_file=os.path.join(R, MAIN_SPLIT + ".merge_state.json")))
    stages.append(Stage("merge_warm", [merge(WARM_SPLIT, d["warm_max"], merge_slices(cfg, d, warm_name), prefix_flag)],
                        split_files(os.path.join(R, WARM_SPLIT), EVIDENCE_FILES + ("slice", "merge.json")),
                        "the first documents of the main order up to warm_max tokens; writes the main split's order sidecar",
                        requires=["merge_main", "bios_warm"], check="merge_warm",
                        post=[write_order_sidecar], state_file=os.path.join(R, WARM_SPLIT + ".merge_state.json")))
    stages.append(Stage("merge_control", [merge(CTL_SPLIT, d["control"], merge_slices(cfg, d, full_name),
                                                prefix_flag + "--no-evidence ")],
                        split_files(os.path.join(R, CTL_SPLIT), PLAIN_FILES + ("slice", "merge.json")),
                        "no evidence files, the biographies in the full form, the first 10B of the main order",
                        requires=["merge_main", "bios_full"], check="merge_control",
                        state_file=os.path.join(R, CTL_SPLIT + ".merge_state.json")))

    # the tail merge reads a seeded sample of the main split (the biographies in their main form with the filler windows they
    # were built with), the permuted QA copy (its chat arms are the tail's chat), both chain builds and the tail's plain
    # text. The biography builds and the raw QA build are not needed any more once the three first merges are done, so
    # they are deleted before the tail merge writes (the disk peak: docs/plans/RUN2_corpus.md section 3)
    pruned_bios = []
    for name in (main_name, full_name, warm_name):
        pruned_bios += split_files(os.path.join(R, name), EVIDENCE_FILES + ("factspan",))
    pruned_bios += split_files(os.path.join(P, QA_PREFIX + "_train"), EVIDENCE_FILES + ("evhop", "chains.jsonl"))
    stages.append(Stage("prune_bios", [f"rm -f {' '.join(q(p) for p in pruned_bios)}"], [],
                        "the three biography builds and the raw QA build: the main, warm and control merges and the QA copy "
                        "hold them; the tail merge samples the main split instead",
                        requires=["merge_main", "merge_warm", "merge_control", "qa_shuffle"], prune=pruned_bios))
    tail_in = {"text": tail_text_prefix, "main": os.path.join(R, MAIN_SPLIT), "qa": os.path.join(R, SHUFFLED_SLICES["qa"]),
               "chains": os.path.join(P, PLAIN_PREFIX + "_train"), "bridge": os.path.join(P, BRIDGE_PREFIX + "_train")}
    tail_passes = {"text": 1, "main": 1, "qa": plan.tail_qa_passes, "chains": plan.tail_chain_passes,
                   "bridge": plan.tail_chain_passes}
    tail_slices = [f"{l}={tail_in[l]}:{share(d['tail']['shares'][l])}:{tail_passes[l]}" for l in TAIL_LABELS]
    tail_cmd = (f"{py}merge_evidence_splits.py --out-dir {q(R)} --split {TAIL_SPLIT} --target-tokens {int(d['tail']['tokens'])} "
                f"--seed {TAIL_SEED} --max-evidence-tokens 4608 --overwrite "
                + " ".join(f"--slice {q(x)}" for x in tail_slices))
    stages.append(Stage("merge_tail", [tail_cmd],
                        split_files(os.path.join(R, TAIL_SPLIT), EVIDENCE_FILES + ("slice", "merge.json")),
                        "the tail split (the last 5% of the run, read from the 760B stop): plain text from the higher "
                        "quality tail phase, a sample of the main split, the QA build with its chat arms, both chain builds; "
                        "its own seed, so its permutations differ from the main run's",
                        requires=["merge_main", "tail_text", "qa_shuffle", "chains_plain", "chains_bridge", "prune_bios"],
                        check="merge_tail", state_file=os.path.join(R, TAIL_SPLIT + ".merge_state.json")))

    main_rows = os.path.join(R, MAIN_SPLIT + ".rows_estimate.json")
    ctl_rows = os.path.join(R, CTL_SPLIT + ".rows_estimate.json")
    tail_rows = os.path.join(R, TAIL_SPLIT + ".rows_estimate.json")
    estimate = (f"{py}sft.py --evidence --estimate-rows-only --estimate-world-sizes {ROWS_WORLD_SIZES} "
                f"--data-dir {q(R)} --train-split ")
    stages.append(Stage("rows", [f"TINY_LLM_CONFIG=config_run2.yaml {estimate}{MAIN_SPLIT} --plan-split {MAIN_SPLIT}",
                                 f"TINY_LLM_CONFIG=config_run2_control.yaml {estimate}{CTL_SPLIT}",
                                 f"TINY_LLM_CONFIG=config_run2.yaml {estimate}{TAIL_SPLIT}"],
                        [main_rows, ctl_rows, tail_rows],
                        "the packed row estimate of the main, control and tail splits (the main one sets total_steps and "
                        "the cosine; the tail's is checked against the rows the last stretch of the run needs), CPU only; "
                        "the main estimate depends on the order sidecar the warm merge writes",
                        requires=["merge_main", "merge_warm", "merge_control", "merge_tail"],
                        needs_files=[os.path.join(R, MAIN_SPLIT + ".order.json")], post=[rows_check]))

    pruned_sources = []
    for prefix in (PLAIN_PREFIX, BRIDGE_PREFIX):
        pruned_sources += split_files(os.path.join(P, prefix + "_train"), EVIDENCE_FILES + ("evhop", "chains.jsonl"))
    for path in shuf.values():
        pruned_sources += split_files(path, EVIDENCE_FILES + ("slice", "merge.json"))
    pruned_sources += split_files(tail_text_prefix, PLAIN_FILES + ("json",))
    stages.append(Stage("prune_sources", [f"rm -f {' '.join(q(p) for p in pruned_sources)}"], [],
                        "the chain builds, the permuted QA copy and the tail's plain text, which the four merges copied; "
                        "inject_val, the card store and the val splits stay",
                        requires=["merge_main", "merge_warm", "merge_control", "merge_tail"], prune=pruned_sources))

    stages.append(Stage("heldout_qa", [offline(f"prepare_evidence_data.py --heldout --data-dir {q(P)} "
                                              f"--max-evidence-tokens 4608{held} --device {cfg.device}")],
                        split_files(os.path.join(P, "evidence_dev"), EVIDENCE_FILES)
                        + split_files(os.path.join(P, "evidence_fixed"), EVIDENCE_FILES), "SQuAD v2 and HotpotQA dev"))
    stages.append(Stage("heldout_chains", [f"{py}prepare_chain_data.py --out-dir {q(P)} --prefix chains "
                                           f"--splits eval,heldout_tmpl,hop4 --eval-questions-per-hop {cfg.chain_eval_per_hop} "
                                           f"--seed {plan.seed} --device cpu"],
                        sum([split_files(os.path.join(P, f"chains_{n}"), EVIDENCE_FILES + ("chains.jsonl",))
                             for n in ("eval", "heldout_tmpl", "hop4")], []), "same seed, so the held out compositions are the pilot's"))
    stages.append(Stage("store_openqa", [offline(f"build_store.py --name openqa --sources nq,triviaqa,hotpotqa "
                                                f"--max-questions {cfg.store_max_questions} --chunk-tokens 128 "
                                                f"--index-root {q(I)} --device {cfg.device} --seed {plan.seed}")],
                        [os.path.join(I, "openqa", "meta.json")], "A4 and A7 store"))
    stages.append(Stage("store_edit", [f"{py}build_store.py --edit-from openqa --name openqa_edit --edit-fraction 0.5 "
                                       f"--neighbour-cos 0.85 --index-root {q(I)} --device cpu --seed {plan.seed}"],
                        [os.path.join(I, "openqa_edit", "meta.json")], "the A4 edit twin"))
    esc = glob.escape
    restart_clean = {
        "val_clean": [esc(os.path.join(R, "inject_val_clean")) + ".*"],
        "order": [esc(order_slice) + "*", esc(order_info) + "*"],
        "tail_text": [esc(tail_text_prefix) + ".*"],
        "heldout_qa": [esc(os.path.join(P, "evidence_dev")) + ".*", esc(os.path.join(P, "evidence_fixed")) + ".*"],
        "heldout_chains": [esc(os.path.join(P, f"chains_{n}")) + ".*" for n in ("eval", "heldout_tmpl", "hop4")],
        "store_openqa": [esc(os.path.join(I, "openqa"))],
        "store_edit": [esc(os.path.join(I, "openqa_edit"))],
    }
    for st in stages:
        if not st.prune and not RESUMABLE.get(st.name, True):
            st.restart = True
            st.clean = restart_clean.get(st.name, [])
    env = offline_env(cfg)
    if env:
        for st in stages:
            if st.name in ENV_STAGES:
                st.commands = [env + c for c in st.commands]
    return stages


STAGE_ORDER = ["filler", "qa", "bios_main", "val_clean", "bios_full", "bios_warm", "tail_text", "prune_filler",
               "chains_plain", "chains_bridge", "qa_shuffle", "order", "merge_main", "merge_warm", "merge_control",
               "prune_bios", "merge_tail", "rows", "prune_sources",
               "heldout_qa", "heldout_chains", "store_openqa", "store_edit"]


# --------------------------------------------------------------------------- merge checks

def file_md5(path: str, limit: Optional[int] = None, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    left = limit
    with open(path, "rb") as f:
        while left is None or left > 0:
            buf = f.read(chunk if left is None else min(chunk, left))
            if not buf:
                break
            h.update(buf)
            if left is not None:
                left -= len(buf)
    return h.hexdigest()


def split_docs(prefix: str) -> int:
    """Documents of a split from its ``.slice`` file (one uint8 per document)."""
    return os.path.getsize(prefix + ".slice")


def split_tokens(prefix: str) -> int:
    with open(prefix + ".idx", "rb") as f:
        f.seek(-8, os.SEEK_END)
        return int.from_bytes(f.read(8), "little")


def prefix_content_problems(main_prefix: str, small_prefix: str, bios_label: int = 0, chunk_docs: int = 2_000_000) -> List[str]:
    """Non-biography documents of a prefix merge must equal the main split's, document for document.

    The biography documents differ in token length between builds, so the two ``.bin`` files are
    compared after dropping the biography documents of both: what remains must be identical, and so
    must the ``.mask`` bytes and the per document slice labels.
    """
    import numpy as np
    problems = []
    n = split_docs(small_prefix)
    sl_main = np.fromfile(main_prefix + ".slice", dtype=np.uint8, count=n)
    sl_small = np.fromfile(small_prefix + ".slice", dtype=np.uint8)
    if not np.array_equal(sl_main, sl_small):
        return [f"{os.path.basename(small_prefix)}: .slice differs from the first {n:,} bytes of the main .slice"]
    idx_m = np.memmap(main_prefix + ".idx", dtype=np.uint64, mode="r")
    idx_s = np.memmap(small_prefix + ".idx", dtype=np.uint64, mode="r")
    tok_m = np.memmap(main_prefix + ".bin", dtype=np.uint16, mode="r")
    tok_s = np.memmap(small_prefix + ".bin", dtype=np.uint16, mode="r")
    mask_m = np.memmap(main_prefix + ".mask", dtype=np.uint8, mode="r")
    mask_s = np.memmap(small_prefix + ".mask", dtype=np.uint8, mode="r")
    for a in range(0, n, chunk_docs):
        b = min(n, a + chunk_docs)
        lens_m = np.diff(idx_m[a:b + 1].astype(np.int64))
        lens_s = np.diff(idx_s[a:b + 1].astype(np.int64))
        keep = sl_main[a:b] != bios_label
        if not np.array_equal(lens_m[keep], lens_s[keep]):
            problems.append(f"{os.path.basename(small_prefix)}: non-biography document lengths differ in documents {a:,} to {b:,}")
            break
        sel_m = np.repeat(sl_main[a:b] != bios_label, lens_m)
        sel_s = np.repeat(keep, lens_s)
        for name, xm, xs in (("bin", tok_m[int(idx_m[a]):int(idx_m[b])], tok_s[int(idx_s[a]):int(idx_s[b])]),
                             ("mask", mask_m[int(idx_m[a]):int(idx_m[b])], mask_s[int(idx_s[a]):int(idx_s[b])])):
            if not np.array_equal(np.asarray(xm)[sel_m], np.asarray(xs)[sel_s]):
                problems.append(f"{os.path.basename(small_prefix)}: non-biography {name} bytes differ in documents {a:,} to {b:,}")
                return problems
    return problems


def check_merge(cfg: Cfg, stage: str, deep: bool = True) -> List[str]:
    """Problems with a merge's output: the md5 lines present, and a prefix merge agreeing with the main split.

    A prefix merge (warm, control) must have a ``.slice`` file equal to the first bytes of the main
    one (md5 compared), reach its token target, and (``deep``) hold the same non-biography
    documents as the main split's first documents.
    """
    if stage == "merge_tail":
        return check_tail_merge(cfg)
    problems = []
    split = {"merge_warm": WARM_SPLIT, "merge_main": MAIN_SPLIT, "merge_control": CTL_SPLIT}[stage]
    path = os.path.join(cfg.run2, split + ".merge.json")
    if not os.path.exists(path):
        return [f"{path} missing"]
    with open(path) as f:
        meta = json.load(f)
    md5 = meta.get("md5_by_slice", {})
    for label in SLICE_LABELS:
        if label not in md5:
            problems.append(f"{split}: md5_by_slice has no {label}")
    prefix = os.path.join(cfg.run2, split)
    if stage == "merge_main":
        order = os.path.join(cfg.run2, ORDER_NAME)
        info_path = order + ".json"
        if not os.path.exists(order + ".slice"):
            problems.append(f"{order}.slice missing")
        elif split_docs(prefix) != split_docs(order) or file_md5(prefix + ".slice") != file_md5(order + ".slice"):
            problems.append(f"{split}: .slice differs from the order file")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            if split_tokens(prefix) != info["total_tokens"]:
                problems.append(f"{split}: {split_tokens(prefix):,} tokens, the order file's inputs hold {info['total_tokens']:,}")
    if stage != "merge_main":
        main = os.path.join(cfg.run2, MAIN_SPLIT)
        if not os.path.exists(main + ".slice"):
            problems.append(f"{main}.slice missing: merge_main has not run")
        else:
            n = split_docs(prefix)
            if n > split_docs(main):
                problems.append(f"{split} has more documents than the main split")
            elif file_md5(main + ".slice", n) != file_md5(prefix + ".slice"):
                problems.append(f"{split}: .slice differs from the main split's first {n:,} bytes")
            elif deep:
                problems += prefix_content_problems(main, prefix)
        target = derive(cfg.plan)["warm_max" if stage == "merge_warm" else "control"]
        if os.path.exists(prefix + ".idx") and split_tokens(prefix) < 0.999 * target:
            problems.append(f"{split}: {split_tokens(prefix):,} tokens, under the target {int(target):,}")
        n_meta = meta.get("num_docs")
        if n_meta is not None and os.path.exists(prefix + ".slice") and n_meta != split_docs(prefix):
            problems.append(f"{split}: merge.json num_docs {n_meta} differs from the .slice file {split_docs(prefix)}")
    for label, value in md5.items():
        print(f"  md5_by_slice {split} {label} {value}")
    return problems


TAIL_GAP_TOLERANCE = 0.01


def check_tail_merge(cfg: Cfg) -> List[str]:
    """The tail split holds its five slices, reached its token target and read no slice past its pass cap.

    A slice that ran out of documents leaves a gap to the target (the merge prints it and writes the tokens it got): a gap
    over 1% means the run would reach the end of the split with the cosine not yet at its floor, so it is refused here.
    """
    path = os.path.join(cfg.run2, TAIL_SPLIT + ".merge.json")
    if not os.path.exists(path):
        return [f"{path} missing"]
    with open(path) as f:
        meta = json.load(f)
    problems = []
    md5 = meta.get("md5_by_slice", {})
    for label in TAIL_LABELS:
        if label not in md5:
            problems.append(f"{TAIL_SPLIT}: md5_by_slice has no {label}")
    target = derive(cfg.plan)["tail"]["tokens"]
    tokens = meta.get("prompt_tokens", 0)
    if tokens < (1.0 - TAIL_GAP_TOLERANCE) * target:
        passes = [(st["label"], round(st["passes"], 2), st["max_passes"]) for st in meta.get("slices", [])]
        problems.append(f"{TAIL_SPLIT}: {tokens:,} tokens, {target - tokens:,.0f} under the target {int(target):,} "
                        f"(a slice ran out of documents or of passes: {passes})")
    for st in meta.get("slices", []):
        print(f"  {TAIL_SPLIT} {st['label']}: {st['prompt_tokens']:,} tokens, share {st['share_realised']:.3f} "
              f"(target {st['share_target']:.3f}), {st['passes']:.2f} passes of its input (cap {st['max_passes']}), "
              f"evidence ratio {st['evidence_ratio']:.3f}")
    for label, value in md5.items():
        print(f"  md5_by_slice {TAIL_SPLIT} {label} {value}")
    return problems


QA_SMOLTALK2_FAIL = 0.90


def qa_arm_check(cfg: Cfg) -> List[str]:
    """The smoltalk2 replay arm reached its target: a short arm is not an error in the builder (the round robin gives its
    share to the other arms), so the stage checks the realized tokens in the builder's state file.

    Fails under 90% of the arm's target and warns under 100% (smoke scales only warn: the supply scales with the plan, the
    real files do not).
    """
    path = os.path.join(cfg.prepared, "_prepare_state_" + QA_PREFIX + ".json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        state = json.load(f)
    got = state.get("sources", {}).get("smoltalk2", {}).get("tokens")
    if got is None:
        return []
    target = QA_ARM_WEIGHTS["smoltalk2"] * cfg.plan.scaled("qa_target")
    frac = got / target if target else 1.0
    if frac >= 1.0:
        return []
    msg = (f"the smoltalk2 replay arm realized {got:,} of {target:,.0f} tokens ({frac:.0%}): the supply left after phase 2 and ir "
           f"is short; lower QA_ARM_WEIGHTS['smoltalk2'] (the arm's share) or the qa target, or phase 2 / ir smoltalk2 tokens, and rebuild qa")
    if frac < QA_SMOLTALK2_FAIL and cfg.plan.scale >= 1.0:
        return [msg]
    print(f"!!! WARNING {msg}", flush=True)
    return []


def qa_shuffle_check(cfg: Cfg) -> List[str]:
    """The permuted QA copy holds every prompt token of the build it was drawn from, once."""
    src = os.path.join(cfg.prepared, QA_PREFIX + "_train")
    dst = os.path.join(cfg.run2, SHUFFLED_SLICES["qa"])
    if not os.path.exists(src + ".idx"):
        return []   # the build was pruned after the merges; the copy was checked then
    problems = []
    if split_tokens(dst) != split_tokens(src):
        problems.append(f"{dst}: {split_tokens(dst):,} tokens, the build {src} holds {split_tokens(src):,}")
    if split_docs(dst) > os.path.getsize(src + ".idx") // 8 - 1:
        problems.append(f"{dst} has more documents than {src}")
    return problems


def write_order_sidecar(cfg: Cfg) -> List[str]:
    """``run2_main_train.order.json``: how many leading main order documents the warm-up covered.

    The trainer visits those documents first in epoch 0 of the main split (in the warm split's own
    permutation) so a switch continues the stream instead of repeating or skipping it.
    """
    warm = os.path.join(cfg.run2, WARM_SPLIT)
    if not os.path.exists(warm + ".slice"):
        return [f"{warm}.slice missing"]
    head = split_docs(warm)
    meta_path = warm + ".merge.json"
    md5 = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            md5 = json.load(f).get("md5_by_slice", {})
    body = {"epoch0_head_docs": head,
            "source": {"warm_split": WARM_SPLIT, "warm_tokens": split_tokens(warm) if os.path.exists(warm + ".idx") else None,
                       "warm_slice_md5": file_md5(warm + ".slice"), "warm_md5_by_slice": md5}}
    out = os.path.join(cfg.run2, MAIN_SPLIT + ".order.json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(body, f, indent=1)
    os.replace(tmp, out)
    print(f"  wrote {out}: epoch0_head_docs {head:,}")
    return []


# --------------------------------------------------------------------------- running

class Interrupted(Exception):
    """SIGTERM arrived; the running child was told and waited for."""


_STATE = {"requested": False, "procs": set()}
_LOCK = threading.Lock()


def handle_term(signum, frame) -> None:
    _STATE["requested"] = True
    for proc in list(_STATE["procs"]):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


_LAST_OUTPUT = {"t": time.time()}
CHILD_OUTPUT_FILE = "last_child_output"
CHILD_TOUCH_SECONDS = 10.0
_CHILD_TOUCH = {"t": 0.0}


def touch_child_output(cfg: "Cfg", force: bool = False) -> None:
    """Touch ``run2_logs/last_child_output``: the corpus job's stall check reads its mtime.

    Called when a child prints a line (throttled to once per ``CHILD_TOUCH_SECONDS``) and at a stage's
    start and end (forced); the heartbeat lines do not touch it, so a hung child goes stale.
    """
    now = time.time()
    with _LOCK:
        if not force and now - _CHILD_TOUCH["t"] < CHILD_TOUCH_SECONDS:
            return
        _CHILD_TOUCH["t"] = now
    try:
        os.makedirs(cfg.logs, exist_ok=True)
        path = os.path.join(cfg.logs, CHILD_OUTPUT_FILE)
        with open(path, "a"):
            pass
        os.utime(path, None)
    except OSError:
        pass


def say(tag: str, msg: str) -> None:
    """One output line, prefixed with the stage name when stages run in parallel."""
    _LAST_OUTPUT["t"] = time.time()
    print(f"[{tag}] {msg}" if tag else msg, flush=True)


class Heartbeat:
    """Prints a line when nothing has been printed for ``interval`` seconds while a stage works.

    The corpus job's watchdog stops a job whose files stop growing; a long check (md5 over hundreds of
    GB, a merge check) or a child that writes nothing for a while is healthy, so it says so.
    """

    def __init__(self, tag: str, name: str, interval: Optional[float] = None):
        self.tag, self.name = tag, name
        self.interval = HEARTBEAT_SECONDS if interval is None else interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.t0 = time.time()

    def loop(self) -> None:
        poll = min(5.0, max(0.05, self.interval / 4))
        while not self.stop.wait(poll):
            if time.time() - _LAST_OUTPUT["t"] >= self.interval:
                say(self.tag, f"heartbeat: stage {self.name} still working, {time.time() - self.t0:.0f}s in")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        return False


def install_signal_handlers() -> None:
    _STATE["requested"] = False
    signal.signal(signal.SIGTERM, handle_term)


def shell_line(cmd: str) -> str:
    return f"cd {q(REPO)} && source {q(ENV_INIT)} && export PYTHONUNBUFFERED=1 && {cmd}"


def marker_path(cfg: Cfg, name: str) -> str:
    return os.path.join(cfg.done, name + ".json")


def finished_on_disk(stage: Stage) -> bool:
    """A resumable tool's outputs are all there and its state file is gone: it finished."""
    return bool(stage.state_file) and all(os.path.exists(p) for p in stage.outputs) \
        and not os.path.exists(stage.state_file)


def clean_partial(stage: Stage, log: Callable) -> int:
    """Delete what a tool that cannot resume left behind (the stage's ``clean`` patterns, else its outputs)."""
    removed = 0
    for pat in (stage.clean or [glob.escape(p) for p in stage.outputs]):
        for p in glob.glob(pat):
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed += 1
    log(f"  restart stage: removed {removed} partial outputs")
    return removed


def run_stage(cfg: Cfg, stage: Stage, dry: bool, force: bool, tag: str = "") -> bool:
    """Run one stage under a non-blocking lock on ``run2_done/STAGE.lock``: a second driver process that
    reaches the same stage fails it at once instead of running the same merge beside the first."""
    if dry:
        return _run_stage(cfg, stage, dry, force, tag)
    try:
        import fcntl
    except ImportError:
        return _run_stage(cfg, stage, dry, force, tag)
    os.makedirs(cfg.done, exist_ok=True)
    with open(os.path.join(cfg.done, stage.name + ".lock"), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            say(tag, f"!!! stage {stage.name}: another driver process holds run2_done/{stage.name}.lock; not waiting")
            return False
        return _run_stage(cfg, stage, dry, force, tag)


def _run_stage(cfg: Cfg, stage: Stage, dry: bool, force: bool, tag: str = "") -> bool:
    marker = marker_path(cfg, stage.name)
    say(tag, f"=== stage {stage.name}: {stage.note}")
    for c in stage.commands:
        say(tag, f"+ {c}")
    if dry:
        return True
    if _STATE["requested"]:
        raise Interrupted()
    missing_req = [r for r in stage.requires if not os.path.exists(marker_path(cfg, r))]
    if missing_req:
        say(tag, f"!!! stage {stage.name}: needs {', '.join(missing_req)} done first")
        return False
    missing_in = [p for p in stage.needs_files if not os.path.exists(p)]
    if missing_in:
        say(tag, f"!!! stage {stage.name}: needs {', '.join(missing_in)} first")
        return False
    present = all(os.path.exists(p) for p in stage.outputs)
    pruned = bool(stage.pruned_by) and os.path.exists(marker_path(cfg, stage.pruned_by))
    if os.path.exists(marker) and (present or pruned) and not force:
        say(tag, f"=== stage {stage.name}: done before ({'pruned by ' + stage.pruned_by if pruned and not present else 'marker and ' + str(len(stage.outputs)) + ' outputs present'}), skipped")
        # the merge checks run again on a skip (md5 of the .slice files, token totals; not the deep document
        # comparison, which ran when the stage finished), unless the stage's outputs were pruned away
        problems = check_merge(cfg, stage.check, deep=False) if stage.check and present else []
        problems += [p for hook in stage.post for p in hook(cfg)]
        for p in problems:
            say(tag, f"!!! stage {stage.name}: {p}")
        return not problems
    os.makedirs(cfg.logs, exist_ok=True)
    os.makedirs(cfg.done, exist_ok=True)
    if os.path.exists(marker):
        os.remove(marker)
    touch_child_output(cfg, force=True)
    try:
        with Heartbeat(tag, stage.name):
            return execute_stage(cfg, stage, force, tag, marker)
    finally:
        touch_child_output(cfg, force=True)


def execute_stage(cfg: Cfg, stage: Stage, force: bool, tag: str, marker: str) -> bool:
    present = all(os.path.exists(p) for p in stage.outputs)
    log = os.path.join(cfg.logs, stage.name + ".log")
    t0 = time.time()
    finished = bool(stage.state_file) and present and not os.path.exists(stage.state_file) and not force
    if finished:
        say(tag, f"=== stage {stage.name}: outputs present and no resume state, the tool finished earlier; checking only")
    late = None     # set when a stop arrived after the work was done: "clean" (child exit 0) or "killed"
    if stage.prune:
        removed = 0
        with open(log, "a", encoding="utf-8") as lf:
            for p in stage.prune:
                if os.path.lexists(p):
                    os.remove(p)
                    removed += 1
            lf.write(f"removed {removed} of {len(stage.prune)} files\n")
        say(tag, f"  removed {removed} of {len(stage.prune)} files")
    elif not finished:
        for hook in stage.pre:
            for p in hook(cfg):
                say(tag, f"!!! stage {stage.name}: {p}")
        if stage.restart:
            clean_partial(stage, lambda m: say(tag, m))
        with open(log, "a", encoding="utf-8") as lf:
            for n, c in enumerate(stage.commands):
                lf.write(f"\n+ {c}\n")
                lf.flush()
                proc = subprocess.Popen(["bash", "-c", shell_line(c)], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, cwd=REPO, start_new_session=True)
                with _LOCK:
                    _STATE["procs"].add(proc)
                if _STATE["requested"]:
                    handle_term(signal.SIGTERM, None)
                try:
                    for line in proc.stdout:
                        lf.write(line)
                        say(tag, line.rstrip("\n"))
                        touch_child_output(cfg)
                    code = proc.wait()
                finally:
                    with _LOCK:
                        _STATE["procs"].discard(proc)
                lf.write(f"exit {code}\n")
                if _STATE["requested"]:
                    # decide on the outputs, not on the signal: a stop that lands after the last stop check
                    # of a child that was done still gets its checks and its marker
                    if code == 0 and n == len(stage.commands) - 1:
                        late = "clean"
                    elif code != 0 and finished_on_disk(stage):
                        late = "killed"
                    else:
                        say(tag, f"=== stage {stage.name}: interrupted (child exit {code}); rerun to resume")
                        raise Interrupted()
                    say(tag, f"=== stage {stage.name}: stop requested after the work was done (child exit {code}); "
                             f"checking and marking before stopping")
                    break
                if code != 0:
                    say(tag, f"!!! stage {stage.name}: command exited {code}, see {log}")
                    return False
    missing = [p for p in stage.outputs if not os.path.exists(p)]
    problems = [f"missing output {p}" for p in missing]
    if stage.check and not missing:
        problems += check_merge(cfg, stage.check)
    if not problems:
        for hook in stage.post:
            problems += hook(cfg)
    if problems:
        if late == "killed":
            say(tag, f"=== stage {stage.name}: stop requested and the outputs do not check out; rerun to resume")
            raise Interrupted()
        for p in problems:
            say(tag, f"!!! stage {stage.name}: {p}")
        return False
    with open(marker, "w") as f:
        json.dump({"stage": stage.name, "seconds": round(time.time() - t0, 1), "commands": stage.commands,
                   "outputs": stage.outputs}, f, indent=1)
    say(tag, f"=== stage {stage.name}: ok in {time.time() - t0:.0f}s")
    return True


def run_plan(order: Sequence[str], parallel: Sequence[str]) -> List[object]:
    """The run list: stage names in order, the ``parallel`` stages gathered into one group (a list)
    at the position of the earliest of them."""
    group = [n for n in order if n in parallel]
    out: List[object] = []
    placed = False
    for n in order:
        if n in parallel:
            if not placed:
                out.append(group)
                placed = True
        else:
            out.append(n)
    return out


def parallel_problems(order: Sequence[str], parallel: Sequence[str], stages: Dict[str, Stage]) -> List[str]:
    """Why a parallel group cannot run: dependencies inside it, unmet earlier ones, shared build directories."""
    problems = []
    group = [n for n in order if n in parallel]
    for n in parallel:
        if n not in order:
            problems.append(f"--parallel names {n}, which is not among the requested stages")
    if len(group) < 2:
        return problems
    injection = [n for n in group if n.startswith("bios_")]
    if len(injection) > 1:
        problems.append(f"{', '.join(injection)} write one inject_build.json and must run one at a time")
    if any(n.startswith("prune_") for n in group):
        problems.append("a prune stage cannot run beside the stages it waits for")
    first = order.index(group[0])
    for n in group:
        for r in stages[n].requires:
            if r in group:
                problems.append(f"{n} needs {r}, which is in the same parallel group")
            elif r in order and order.index(r) > first:
                problems.append(f"{n} needs {r}, which would run after the group starts")
    return problems


def run_parallel(cfg: Cfg, names: Sequence[str], stages: Dict[str, Stage], dry: bool, force: bool) -> bool:
    """Run independent stages as concurrent processes; each keeps its own log, marker and output tag."""
    print(f"=== parallel group: {', '.join(names)}")
    if dry:
        return all(run_stage(cfg, stages[n], True, force, tag=n) for n in names)
    from concurrent.futures import ThreadPoolExecutor
    results, interrupted = {}, []

    def one(n):
        try:
            results[n] = run_stage(cfg, stages[n], False, force, tag=n)
        except Interrupted:
            interrupted.append(n)
            results[n] = False
        except Exception as e:
            say(n, f"!!! stage {n}: {type(e).__name__}: {e}")
            results[n] = False
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        list(pool.map(one, names))
    if interrupted:
        raise Interrupted()
    bad = [n for n, ok in results.items() if not ok]
    if bad:
        print(f"!!! parallel group: {', '.join(bad)} did not finish; the others' markers stand")
    return not bad


def link_filler(cfg: Cfg) -> None:
    """Smoke only: symlink small local filler bins into the prepared dir instead of building them."""
    os.makedirs(cfg.prepared, exist_ok=True)
    for phase in ("phase1", "phase2", "ir"):
        for s in ("bin", "idx"):
            dst = os.path.join(cfg.prepared, f"{phase}.{s}")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(os.path.join(cfg.link_filler_from, f"{phase}.{s}")), dst)
    # the tail's plain text phase: the local ir bins stand in (the smoke has no fresh files to read)
    os.makedirs(cfg.tail_dir, exist_ok=True)
    for s in ("bin", "idx"):
        dst = os.path.join(cfg.tail_dir, f"phase1.{s}")
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(cfg.link_filler_from, f"ir.{s}")), dst)


def inbox_done_path(cfg: Cfg) -> str:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mirror_sources import DONE_MARK
    return os.path.join(cfg.mirror_inbox, DONE_MARK)


def clear_inbox_done(cfg: Cfg) -> List[str]:
    """Before the filler runs: drop a stale end mark so the pusher does not stop early."""
    try:
        os.remove(inbox_done_path(cfg))
    except OSError:
        pass
    return []


def write_inbox_done(cfg: Cfg) -> List[str]:
    """After the filler finished: tell the pusher the consumer is done with the inbox.

    Runs on the skip path too; an inbox the operator already deleted after the filler is fine.
    """
    if not os.path.isdir(cfg.mirror_inbox):
        return []
    try:
        with open(inbox_done_path(cfg), "w") as f:
            f.write("filler finished\n")
    except OSError as e:
        return [f"could not write the inbox end mark {inbox_done_path(cfg)}: {e}"]
    return []


FILLER_REPEAT_TOLERANCE = 0.001


def filler_repeat_check(cfg: Cfg) -> List[str]:
    """The biography builds must not have repeated filler: the filler stage owes the plan's tokens.

    Reads ``common.filler_repeat_factor`` of ``inject_build.json`` (1.0 means every filler window was
    used once). A smoke run that links a small filler (``--link-filler-from``) or ``--allow-filler-repeat``
    skips the check.
    """
    if cfg.allow_filler_repeat or cfg.link_filler_from:
        return []
    path = os.path.join(cfg.run2, "inject_build.json")
    try:
        with open(path) as f:
            repeat = json.load(f).get("common", {}).get("filler_repeat_factor")
    except (OSError, ValueError):
        return []
    if repeat is not None and repeat > 1.0 + FILLER_REPEAT_TOLERANCE:
        return [f"{path}: filler repeat factor {repeat:.3f}x, the filler bins hold fewer tokens than the plan "
                f"needs and the biography build repeated windows; rebuild the filler (or pass "
                f"--allow-filler-repeat for a smoke run)"]
    return []


def filler_preflight(cfg: Cfg) -> List[str]:
    """Refuse a filler build that would overwrite the 16B run's manifest entries.

    An entry this driver wrote itself (``run2_data_root`` equal to this data root, written by
    ``wrap-prepare-data`` after each build, or a filler marker under this root) is not foreign: a rerun
    after phase 1 must go on.
    """
    path = os.path.join(REPO, "manifest.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        meta = json.load(f)
    entry = meta.get("data_prep", {})
    if "phase1" not in entry:
        return []
    root = entry.get("run2_data_root")
    if root and os.path.realpath(root) == os.path.realpath(cfg.data_root):
        return []
    if os.path.exists(marker_path(cfg, "filler")):
        return []
    # a kill between prepare_data's manifest write and record_owner leaves no stamp; this root's own phase 1
    # state file is proof the entry was written by this run
    if os.path.exists(os.path.join(cfg.prepared, "_prepare_state_phase1.json")):
        return []
    return [f"{path} already holds data_prep.phase1 and this run did not write it (no run2_data_root for "
            f"{cfg.data_root}; the 16B run's counts?); the filler build would overwrite it. "
            f"Move manifest.json away first (a fresh cluster has none)"]


def record_owner(manifest_path: str, key: str, data_root: str) -> None:
    """Stamp the manifest entry a filler build just wrote with this run's data root (atomic)."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.setdefault(key, {})["run2_data_root"] = data_root
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, manifest_path)


# --------------------------------------------------------------------------- wrappers

def split_dashes(argv: Sequence[str]):
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def parse_weights(text: str) -> Dict[str, float]:
    return {k: float(v) for k, v in (p.split("=") for p in text.split(",") if p)}


def apply_phase1_weights(specs, weights: Dict[str, float], log: Callable = print) -> None:
    """Set each source's phase 1 weight (prepare_data renormalizes) and print before and after shares."""
    old_total = sum(s.phase1_weight for s in specs)
    new_total = sum(weights.get(s.key, s.phase1_weight) if s.phase1_weight > 0 else 0.0 for s in specs)
    for s in specs:
        if s.phase1_weight > 0:
            before = s.phase1_weight / old_total
            s.phase1_weight = weights.get(s.key, s.phase1_weight)
            log(f"wrap-prepare-data: phase1 {s.key} share {before:.4f} -> {s.phase1_weight / new_total:.4f}")


def wrap_prepare_data(argv: Sequence[str]) -> None:
    """Runs ``scripts/prepare_data.py`` with fineweb's prefix widened, capped phase 1 weights, earlier
    phases' files skipped and, with ``--mirror-inbox``, the Hub replaced by a listing and an inbox."""
    own, rest = split_dashes(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--fineweb-prefix", default=None)
    p.add_argument("--exclude-files-of", default="", help="comma separated phases whose files are skipped")
    p.add_argument("--phase1-weights", default="", help="source=weight,... replaces the phase 1 weights")
    p.add_argument("--mirror-inbox", default=None)
    p.add_argument("--mirror-listing", default=None)
    p.add_argument("--data-root", default=None, help="stamped into the manifest entry as run2_data_root")
    a = p.parse_args(own)
    sys.path.insert(0, REPO)
    import scripts.prepare_data as pd
    manifest = {}
    if os.path.exists(pd.MANIFEST_PATH):
        with open(pd.MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    used: Dict[str, set] = {}
    for phase in [x for x in a.exclude_files_of.split(",") if x]:
        for key in ("data_prep", "ir_prep"):
            per = manifest.get(key, {}).get(phase, {}).get("per_source", {})
            for src, entry in per.items():
                used.setdefault(src, set()).update(entry.get("files_used", []))
    if a.phase1_weights:
        apply_phase1_weights(pd.SOURCES, parse_weights(a.phase1_weights))
    for spec in pd.SOURCES:
        if spec.key == "fineweb" and a.fineweb_prefix:
            spec.file_prefix = a.fineweb_prefix
        skip = used.get(spec.key, set())
        if skip:
            old = spec.file_filter
            spec.file_filter = (lambda f, old=old, skip=skip: f not in skip and (old is None or old(f)))
            print(f"wrap-prepare-data: {spec.key} skips {len(skip)} files an earlier phase used")
    fetcher = None
    if a.mirror_inbox:
        import scripts.mirror_sources as ms
        with open(a.mirror_listing, "r", encoding="utf-8") as f:
            listing = ms.MirrorListing(json.load(f))
        fetcher = ms.InboxFetcher(a.mirror_inbox, listing)
        pd.HfApi = lambda token=None: ms.FakeHfApi(listing, token)
        pd.hf_hub_download = lambda repo_id, filename, **kw: fetcher.download(repo_id, filename)
        pd.os = ms.NoRemoveOs(os, a.mirror_inbox)
        real_load = pd.load_document_texts

        def load(spec, local_path, seed):
            texts = real_load(spec, local_path, seed)
            fetcher.note_rows(len(texts), spec.repo_id)
            return texts
        pd.load_document_texts = load
        print(f"wrap-prepare-data: mirror inbox {a.mirror_inbox}, {len(listing.by_repo)} repos in the listing")
    sys.argv = ["prepare_data.py"] + rest
    pd.main()
    if fetcher is not None:
        fetcher.clear()
    if a.data_root and "--manifest-key" in rest:
        record_owner(pd.MANIFEST_PATH, rest[rest.index("--manifest-key") + 1], a.data_root)


def wrap_chains(argv: Sequence[str]) -> None:
    """Runs ``scripts/prepare_chain_data.py`` with the training hop mix widened to 1..N at equal weight."""
    own, rest = split_dashes(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--hops", type=int, default=3)
    a = p.parse_args(own)
    sys.path.insert(0, REPO)
    from modules.data import chains
    if a.hops != 3:
        chains.sample_hops.__defaults__ = (tuple([1.0 / a.hops] * a.hops),)
        print(f"wrap-chains: training hops 1..{a.hops} at {100.0 / a.hops:.0f}% each")
    import scripts.prepare_chain_data as pcd
    sys.argv = ["prepare_chain_data.py"] + rest
    pcd.main()


def wrap_hub(argv: Sequence[str]) -> None:
    """Runs a builder script with the Hub dataset calls answered from a staged directory (no internet)."""
    import runpy
    own, rest = split_dashes(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--stage-dir", required=True)
    a = p.parse_args(own)
    sys.path.insert(0, REPO)
    import scripts.mirror_sources as ms
    ms.install_offline_hub(a.stage_dir)
    script = os.path.join(REPO, rest[0])
    sys.argv = [script] + rest[1:]
    runpy.run_path(script, run_name="__main__")


# --------------------------------------------------------------------------- reports

def gb(x: float) -> str:
    return f"{x / 1e9:,.1f} GB" if x < 1e12 else f"{x / 1e12:,.2f} TB"


def print_plan(plan: Plan, window_bytes: Optional[float] = None, disk_budget: float = 3.4e12,
               standing_budget: float = 2.6e12, parallel: Sequence[str] = ()) -> List[str]:
    d = derive(plan)
    sched = disk_schedule(plan, d, window_bytes, parallel)
    hours = hours_plan(plan, d)
    print(f"TOKEN_TARGET {TOKEN_TARGET:,.0f}  EPOCHS {EPOCHS}  T = {TOKEN_TARGET / EPOCHS:,.0f}")
    print(f"scale {plan.scale}: {d['total']:,.0f} prompt tokens per split, {d['epochs']:.2f} epochs of it in a "
          f"{plan.run_tokens:,.0f} token run; warm_max {d['warm_max']:,.0f}, control {d['control']:,.0f}")
    print("slice shares of the merged split")
    for k in ("bios", "qa", "chains", "bridge"):
        print(f"  {k:7s} {d[k if k != 'chains' else 'plain']:>18,.0f} tokens  share {d['shares'][k]:.6f}")
    print("qa arms (estimated realised prompt tokens)")
    for k, v in d["qa_arms"].items():
        print(f"  {k:10s} {v:>16,.0f}")
    print(f"filler builds: phase1 {d['phase1']:,.0f}  phase2 {d['phase2']:,.0f}  ir {d['ir']:,.0f}")
    print(f"phase 1 mix, code_edu at 95% and wikipedia at 90% of supply, the shortfall to nemotron_math (up to 65%) then fineweb")
    print(f"  {'source':14s} {'before':>8s} {'after':>8s} {'before B':>10s} {'after B':>10s} {'limit B':>10s}")
    for r in d["phase1_rows"]:
        print(f"  {r['source']:14s} {r['before_share']:8.4f} {r['after_share']:8.4f} {r['before_tokens'] / 1e9:10.2f} "
              f"{r['after_tokens'] / 1e9:10.2f} {r['limit'] / 1e9:10.2f}{'  capped' if r['capped'] else ''}")
    print("filler need against supply (all four phases: phase 1, phase 2, ir, the tail's plain text)")
    for r in filler_supply(plan, d):
        print(f"  {r['source']:14s} need {r['need'] / 1e9:8.2f}B  files {r['files_needed']:5d}  supply "
              f"{r['supply_patched'] / 1e9:8.2f}B (one crawl for fineweb {r['supply_default'] / 1e9:.2f}B)")
    sz = build_sizes(plan, d)
    tl = d["tail"]
    print(f"tail split {TAIL_SPLIT}: {tl['tokens']:,.0f} prompt tokens, the run reads {tl['read']:,.0f} of them from the stop at "
          f"{(plan.run_tokens - plan.tail_read) / 1e9:,.0f}B (a surplus of {tl['tokens'] / max(tl['read'], 1) - 1:.1%})")
    for k in TAIL_LABELS:
        extra = {"qa": f"  {tl['qa_passes']:.2f} passes of the QA build (cap {plan.tail_qa_passes}); its chat arms are about a third",
                 "chains": f"  {tl['plain_passes']:.2f} passes of the plain chain build (cap {plan.tail_chain_passes})",
                 "bridge": f"  {tl['bridge_passes']:.2f} passes of the bridge chain build (cap {plan.tail_chain_passes})",
                 "main": f"  {tl['main_fraction']:.2%} of the main split's documents, a seeded sample"}.get(k, "")
        print(f"  {k:7s} {tl['slice_tokens'][k]:>18,.0f} tokens  share {tl['shares'][k]:.2f}{extra}")
    print(f"  tail plain text build: {tl['text_target']:,} tokens from {tl['filler']:,.0f} filler tokens of fresh files "
          f"(skipping every file of phase 1, 2 and ir); evidence ratio of the tail about {sz['tail_ev_ratio']:.3f} per prompt token "
          f"(the main run's {sz['ev_ratio']:.3f})")
    print(f"  {'source':14s} {'asked B':>9s} {'got B':>9s} {'fresh limit B':>14s}")
    for r in tl["rows"]:
        print(f"  {r['source']:14s} {r['before_tokens'] / 1e9:9.2f} {r['after_tokens'] / 1e9:9.2f} {r['limit'] / 1e9:14.2f}"
              f"{'  capped' if r['capped'] else ''}")
    print(f"evidence tokens {sz['ev_tokens']:,.0f} (bios {d['bio_renders'] * BIOS_EVTOK_PER_RENDER:,.0f} from "
          f"{d['bio_renders']:,} renderings), evidence ratio {sz['ev_ratio']:.4f} per prompt token")
    print(f"chain questions: plain {d['plain_questions']:,}  bridge {d['bridge_questions']:,}")
    print(f"prefix builds: bios full {d['bios_stop_control']:,} tokens, bios warm {d['bios_stop_warm']:,} tokens "
          f"(the token count of the split plus {PREFIX_MARGIN:.0%})")
    print(f"biography renderings {d['bio_renders']:,} for {d['persons']:,} persons")
    print(f"validation windows: cutoff {d['val_cutoff']} per million, about {d['val_windows_expected']:,.0f} expected "
          f"(asked {plan.val_windows}; the builder's hash gives whole millionths only)")
    print("disk schedule (estimate)")
    print(f"  {'stage':16s} {'writes':>10s} {'deletes':>10s} {'transient':>10s} {'running':>10s} {'peak during':>12s}")
    for r in sched["rows"]:
        print(f"  {r['stage']:16s} {gb(r['writes']):>10s} {gb(r['deletes']):>10s} {gb(r['transient']):>10s} "
              f"{gb(r['running']):>10s} {gb(r['peak_during']):>12s}")
    print(f"  peak {gb(sched['peak'])} (budget {gb(disk_budget)}), standing {gb(sched['standing'])} (budget {gb(standing_budget)})")
    print("hours, one process (estimate from the pilot rates)")
    for k in STAGE_ORDER:
        if k in hours:
            v = hours[k]
            tag = "" if k not in RESUMABLE else ("  resumable" if RESUMABLE[k] else "  not resumable")
            over = "  OVER 60 h" if v is not None and v > LONG_STAGE_HOURS else ""
            print(f"  {k:16s} {'unmeasured' if v is None else f'{v:,.1f}':>10s}{tag}{over}")
    print(f"  merge documents {sched['sizes']['docs_main']:,.0f} at {MERGE_DOCS_PER_SECOND:,.0f} per second with evidence, "
          f"{MERGE_DOCS_PER_SECOND_PLAIN:,.0f} without (the prefix mode's local disk rates)")
    t = transfer_estimate(plan, d)
    tb = sum(v["bytes"] for v in t.values())
    print(f"mirror transfer estimate {gb(tb)} in {sum(v['files'] for v in t.values())} files "
          f"(at {BYTES_PER_FILLER_TOKEN} bytes per token, an estimate; limit 2 TB): "
          + ", ".join(f"{k} {v['files']}" for k, v in t.items()))
    problems = disk_problems(sched, disk_budget, standing_budget)
    for label, need, cap in (("qa", tl["qa_passes"], plan.tail_qa_passes), ("chains", tl["plain_passes"], plan.tail_chain_passes),
                             ("bridge", tl["bridge_passes"], plan.tail_chain_passes)):
        if need > cap + 1e-9:
            problems.append(f"the tail's {label} slice needs {need:.2f} passes of its build, the cap is {cap}: raise the cap "
                            f"or lower the share")
    for k in long_stage_flags(hours):
        problems.append(f"stage {k} is estimated over {LONG_STAGE_HOURS:.0f} hours and cannot resume")
    for pr in problems:
        print(f"!!! {pr}")
    return problems


def compute_order(counts: Sequence[int]):
    """Slice index per document: every document of every slice once, the slices interleaved in
    proportion to their size (document ``k`` of a slice of ``n`` sorts at ``(k + 0.5) / n``, a tie goes
    to the lower slice index), deterministic.

    Documents stay in storage order inside a slice, so every slice must already be in random order:
    the biography layout and the chain generator's seeded questions are; the QA builder's rows are not
    (smooth weighted round robin over its arms), which is why ``qa_shuffle`` rewrites that slice in a
    seeded permutation before this runs (``SHUFFLED_SLICES``). The merge tool takes a slice's k-th
    document in storage order in prefix mode, so the permutation has to be physical.
    """
    import numpy as np
    keys = np.concatenate([(np.arange(n, dtype=np.float64) + 0.5) / n for n in counts])
    labels = np.repeat(np.arange(len(counts), dtype=np.uint8), counts)
    idx = np.argsort(keys, kind="stable")
    del keys
    return labels[idx]


def split_summary(prefix: str) -> dict:
    return {"docs": os.path.getsize(prefix + ".idx") // 8 - 1, "tokens": split_tokens(prefix)}


def cmd_order(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="build_run2_corpus.py order")
    p.add_argument("--out", required=True)
    p.add_argument("--info", required=True)
    p.add_argument("--slice", dest="slices", action="append", required=True, help="label=path_prefix")
    a = p.parse_args(argv)
    pairs = [x.split("=", 1) for x in a.slices]
    sums = [split_summary(pth) for _, pth in pairs]
    order = compute_order([x["docs"] for x in sums])
    tmp = a.out + ".tmp"
    order.tofile(tmp)
    os.replace(tmp, a.out)
    info = {"labels": [l for l, _ in pairs], "docs": [x["docs"] for x in sums], "tokens": [x["tokens"] for x in sums],
            "total_docs": int(sum(x["docs"] for x in sums)), "total_tokens": int(sum(x["tokens"] for x in sums)),
            "slice_md5": file_md5(a.out)}
    permuted = {}
    for label, pth in pairs:
        meta_path = pth + ".merge.json"
        if label in SHUFFLED_SLICES and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            permuted[label] = {"split": os.path.basename(pth), "seed": meta.get("args", {}).get("seed"),
                               "method": "merge_evidence_splits one slice seeded permutation, share 1",
                               "documents": meta.get("documents"), "md5": meta.get("md5_by_slice", {}).get(label)}
    info["permuted_slices"] = permuted
    with open(a.info + ".tmp", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(a.info + ".tmp", a.info)
    print(f"order: {info['total_docs']:,} documents, {info['total_tokens']:,} tokens, {len(order):,} bytes in {a.out}")
    return 0


TAIL_WINDOW = 1000       # the biography build's filler window: min(1000, seq_length - 24)
TAIL_MIN_WINDOW = 32     # prepare_injection_data.MIN_WINDOW: a document's last window under this is dropped
TAIL_BLOCK_WINDOWS = 400_000


def tail_text_blocks(offsets, window: int = TAIL_WINDOW, min_window: int = TAIL_MIN_WINDOW, docs_per_chunk: int = 1 << 20):
    """Yield ``(starts, lengths)`` of the windows of each run of documents, in storage order.

    A document is cut from its first token in windows of ``window`` tokens, the last one dropped when under
    ``min_window``: the biography build's rule (``load_filler_windows``), so the tail's plain text has the same shape
    as the main run's filler.
    """
    import numpy as np
    n = len(offsets) - 1
    for lo in range(0, n, docs_per_chunk):
        starts = offsets[lo:min(lo + docs_per_chunk, n)]
        ends = offsets[lo + 1:lo + 1 + len(starts)]
        counts = (np.maximum(ends - starts, 0) + window - 1) // window
        total = int(counts.sum())
        if not total:
            continue
        doc_of = np.repeat(np.arange(len(starts)), counts)
        first = np.cumsum(counts) - counts
        w = starts[doc_of] + (np.arange(total) - first[doc_of]) * window
        length = np.minimum(window, ends[doc_of] - w)
        keep = length >= min_window
        yield w[keep], length[keep]


def tail_text_assemble(tokens, starts, lengths, bos: int, eos: int):
    """``[BOS] window [EOS]`` for each window, concatenated: ``(ids, mask, document end offsets)``.

    The mask is 1 everywhere but on BOS, as the biography build writes its filler documents.
    """
    import numpy as np
    lens = lengths + 2
    offs = np.cumsum(lens) - lens
    ids = np.empty(int(lens.sum()), dtype=np.uint16)
    body_first = np.cumsum(lengths) - lengths
    within = np.arange(int(lengths.sum())) - np.repeat(body_first, lengths)
    ids[np.repeat(offs + 1, lengths) + within] = tokens[np.repeat(starts, lengths) + within]
    ids[offs] = bos
    ids[offs + lens - 1] = eos
    mask = np.ones(len(ids), dtype=np.uint8)
    mask[offs] = 0
    return ids, mask, np.cumsum(lens)


def cmd_tail_text(argv: Sequence[str]) -> int:
    """``tail-text``: the tail phase's documents cut into windows, as a plain split (``bin idx mask``).

    Reads ``SRC.bin`` and ``SRC.idx`` (a ``prepare_data`` phase: raw documents with no BOS or EOS), writes ``OUT.bin``,
    ``OUT.idx``, ``OUT.mask`` and ``OUT.json``, stopping at the first window that reaches ``--target`` tokens (BOS and
    EOS counted). The documents stay in the phase's order (its sources interleaved by weight); the tail merge permutes
    them. With fewer tokens than the target it cycles the input under ``--allow-repeat`` (smoke runs); otherwise it fails
    only below ``--min-tokens`` (what the merge needs; default the target), and between that and the target it warns
    loudly and writes what it has, recording ``shortfall`` in ``OUT.json``.
    """
    import numpy as np
    p = argparse.ArgumentParser(prog="build_run2_corpus.py tail-text")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--min-tokens", type=int, default=None)
    p.add_argument("--allow-repeat", action="store_true")
    p.add_argument("--bos-id", type=int, default=None)
    p.add_argument("--eos-id", type=int, default=None)
    a = p.parse_args(argv)
    bos, eos = a.bos_id, a.eos_id
    if bos is None or eos is None:
        sys.path.insert(0, REPO)
        from transformers import AutoTokenizer
        from utils import TOKENIZER_DIR
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        bos, eos = tok.bos_token_id, tok.eos_token_id
    tokens = np.memmap(a.src + ".bin", dtype=np.uint16, mode="r")
    offsets = np.asarray(np.memmap(a.src + ".idx", dtype=np.uint64, mode="r"), dtype=np.int64)
    min_tokens = a.target if a.min_tokens is None else min(a.min_tokens, a.target)
    short = False
    written = windows = 0
    passes = 0
    tmp = {s: a.out + f".{s}.part" for s in ("bin", "idx", "mask")}
    files = {s: open(path, "wb") for s, path in tmp.items()}
    files["idx"].write(np.array([0], dtype=np.uint64).tobytes())
    t0 = time.time()
    done = False
    while not done:
        got = 0
        for starts, lengths in tail_text_blocks(offsets):
            for lo in range(0, len(starts), TAIL_BLOCK_WINDOWS):
                st, ln = starts[lo:lo + TAIL_BLOCK_WINDOWS], lengths[lo:lo + TAIL_BLOCK_WINDOWS]
                cum = np.cumsum(ln + 2)
                if written + int(cum[-1]) >= a.target:
                    cut = int(np.searchsorted(cum, a.target - written, side="left")) + 1
                    st, ln = st[:cut], ln[:cut]
                    done = True
                ids, mask, ends = tail_text_assemble(tokens, st, ln, bos, eos)
                files["bin"].write(ids.tobytes())
                files["mask"].write(mask.tobytes())
                files["idx"].write((ends + written).astype(np.uint64).tobytes())
                written += len(ids)
                windows += len(st)
                got += len(st)
                if done:
                    break
            if done:
                break
        passes += 1
        print(f"tail-text: pass {passes}: {windows:,} windows, {written:,} tokens ({time.time() - t0:.0f}s)", flush=True)
        if not done:
            if (not a.allow_repeat or not got) and written >= min_tokens:
                short = True
                print(f"tail-text: WARNING the input holds only {written:,} tokens of windows, {a.target - written:,} under the "
                      f"target {a.target:,} but over the merge's need {min_tokens:,}: continuing; the tail's text slice will "
                      f"be that much smaller than planned, check the per source realized counts of tail_prep", flush=True)
                break
            if not a.allow_repeat or not got:
                for f in files.values():
                    f.close()
                for path in tmp.values():
                    os.remove(path)
                print(f"tail-text: the input holds {written:,} tokens of windows, under the merge's need {min_tokens:,} (target {a.target:,})", file=sys.stderr)
                return 1
            print(f"tail-text: input exhausted at {written:,} tokens, cycling it (pass {passes + 1}; --allow-repeat)", flush=True)
    for f in files.values():
        f.flush()
        os.fsync(f.fileno())
        f.close()
    for s, path in tmp.items():
        os.replace(path, f"{a.out}.{s}")
    info = {"src": a.src, "windows": windows, "tokens": written, "target": a.target, "input_passes": passes,
            "repeat": passes > 1, "min_tokens": min_tokens, "shortfall": max(0, a.target - written) if short else 0, "bos": bos, "eos": eos, "window": TAIL_WINDOW}
    with open(a.out + ".json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"tail-text: {windows:,} windows, {written:,} tokens (target {a.target:,}), repeat {'yes' if passes > 1 else 'no'}")
    return 0


def cmd_split_tokens(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="build_run2_corpus.py split-tokens")
    p.add_argument("--prefix", required=True)
    p.add_argument("--plus", type=int, default=0)
    a = p.parse_args(argv)
    print(split_tokens(a.prefix) + a.plus)
    return 0


def cmd_order_total(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="build_run2_corpus.py order-total")
    p.add_argument("--info", required=True)
    a = p.parse_args(argv)
    with open(a.info) as f:
        print(json.load(f)["total_tokens"])
    return 0


def check_offline(argv: Sequence[str]) -> int:
    """``check-offline``: what each requested stage needs from the offline cache, present or missing."""
    p = argparse.ArgumentParser(prog="build_run2_corpus.py check-offline")
    p.add_argument("--offline-stage", required=True)
    p.add_argument("--mirror-inbox", default=None)
    p.add_argument("--mirror-listing", default=None)
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--stage", default="all")
    a = p.parse_args(argv)
    names = STAGE_ORDER if a.stage == "all" else a.stage.split(",")
    cfg = Cfg(data_root="", plan=Plan(), offline_stage=a.offline_stage, mirror_inbox=a.mirror_inbox,
              mirror_listing=a.mirror_listing, tokenizer_dir=a.tokenizer_dir)
    bad = offline_report(cfg, names)
    for line in bad:
        print("MISSING: " + line)
    print("offline cache complete for the requested stages" if not bad else f"{len(bad)} needs missing")
    return 2 if bad else 0


def add_plan_args(p: argparse.ArgumentParser) -> None:
    """The flags that change the plan's numbers; ``mirror_sources list`` takes the same ones so the
    listing's file counts follow the plan the driver will run."""
    p.add_argument("--scale", type=float, default=1.0, help="multiplies every token quantity (smoke runs)")
    p.add_argument("--total", type=float, default=None)
    p.add_argument("--chain-tokens", type=float, default=None, help="absolute chain prompt tokens (default 2.8e9)")
    p.add_argument("--chain-share", type=float, default=None, help="chains as a share of the split instead")
    p.add_argument("--bridge-fraction", type=float, default=None)
    p.add_argument("--qa-target", type=float, default=None)
    p.add_argument("--phase2-tokens", type=float, default=None)
    p.add_argument("--ir-tokens", type=float, default=None)
    p.add_argument("--persons-per-tier", default=None)
    p.add_argument("--no-gold-only", action="store_true")
    p.add_argument("--val-windows", type=int, default=None)


def plan_from_args(a: argparse.Namespace) -> Plan:
    plan = Plan(scale=a.scale, support_gold_only=not a.no_gold_only)
    for name in ("total", "chain_tokens", "chain_share", "bridge_fraction", "qa_target", "phase2_tokens", "ir_tokens"):
        v = getattr(a, name)
        if v is not None:
            setattr(plan, name, v)
    if a.persons_per_tier:
        plan.persons_per_tier = a.persons_per_tier
    if a.val_windows:
        plan.val_windows = a.val_windows
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "wrap-prepare-data":
        wrap_prepare_data(argv[1:])
        return 0
    if argv and argv[0] == "wrap-chains":
        wrap_chains(argv[1:])
        return 0
    if argv and argv[0] == "wrap-hub":
        wrap_hub(argv[1:])
        return 0
    if argv and argv[0] == "check-offline":
        return check_offline(argv[1:])
    if argv and argv[0] == "order":
        return cmd_order(argv[1:])
    if argv and argv[0] == "order-total":
        return cmd_order_total(argv[1:])
    if argv and argv[0] == "split-tokens":
        return cmd_split_tokens(argv[1:])
    if argv and argv[0] == "tail-text":
        return cmd_tail_text(argv[1:])
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["plan", "build"])
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--dry-run", action="store_true", help="print every command, run nothing")
    p.add_argument("--stage", default="all", help="comma separated stage names or all")
    p.add_argument("--force", action="store_true", help="rerun stages that have a marker")
    add_plan_args(p)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fineweb-prefix", default="data/")
    p.add_argument("--filler-dir", default=None, help="where the biography build reads the filler bins")
    p.add_argument("--link-filler-from", default=None, help="smoke: symlink local phase bins into prepared/")
    p.add_argument("--heldout-max-questions", type=int, default=None)
    p.add_argument("--store-max-questions", type=int, default=3000)
    p.add_argument("--chain-eval-per-hop", type=int, default=1000)
    p.add_argument("--mirror-inbox", default=None, help="no internet: the filler reads shards from this inbox")
    p.add_argument("--mirror-listing", default=None, help="the listing json mirror_sources list wrote")
    p.add_argument("--mirror-window-gb", type=float, default=120.0, help="the inbox size bound, counted in the disk plan")
    p.add_argument("--offline-stage", default=None, help="no internet: HF_HOME staged by mirror_sources stage-small")
    p.add_argument("--disk-budget", type=float, default=3.4e12)
    p.add_argument("--standing-budget", type=float, default=2.6e12)
    p.add_argument("--parallel", default="", help="comma separated independent stages to run as concurrent processes "
                   "(one node, several CPU jobs); each keeps its log and marker")
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--allow-filler-repeat", action="store_true",
                   help="do not fail the biography stages when the filler repeated (smoke runs; implied by --link-filler-from)")
    a = p.parse_args(argv)

    plan = plan_from_args(a)
    window = a.mirror_window_gb * 1e9 if a.mirror_inbox else None
    if a.mirror_inbox and not a.mirror_listing:
        p.error("--mirror-inbox needs --mirror-listing")
    if a.mirror_inbox and not a.offline_stage:
        p.error("--mirror-inbox needs --offline-stage: no node has internet, the other stages read the staged cache")
    parallel = [x for x in a.parallel.split(",") if x]
    wanted_early = STAGE_ORDER if a.stage == "all" else a.stage.split(",")
    if a.offline_stage and not a.mirror_inbox and "filler" in wanted_early and not a.link_filler_from:
        p.error("--offline-stage without --mirror-inbox: the filler stage would run with HF_HUB_OFFLINE=1 and no "
                "source for its shards; add --mirror-inbox and --mirror-listing, or leave filler out of --stage")
    for x in parallel:
        if x not in STAGE_ORDER:
            p.error(f"--parallel names unknown stage {x!r}")
    if a.mode == "plan":
        return 2 if print_plan(plan, window, a.disk_budget, a.standing_budget) else 0
    cfg = Cfg(data_root=os.path.abspath(a.data_root), plan=plan, device=a.device, filler_dir=a.filler_dir,
              fineweb_prefix=a.fineweb_prefix, heldout_max_questions=a.heldout_max_questions,
              store_max_questions=a.store_max_questions, chain_eval_per_hop=a.chain_eval_per_hop,
              persons_per_tier=a.persons_per_tier, link_filler_from=a.link_filler_from,
              mirror_inbox=a.mirror_inbox, mirror_listing=a.mirror_listing, offline_stage=a.offline_stage,
              tokenizer_dir=a.tokenizer_dir, mirror_window_gb=a.mirror_window_gb, allow_filler_repeat=a.allow_filler_repeat, disk_budget=a.disk_budget, standing_budget=a.standing_budget)
    wanted = STAGE_ORDER if a.stage == "all" else a.stage.split(",")
    for w in wanted:
        if w not in STAGE_ORDER:
            p.error(f"unknown stage {w!r}; choose from {STAGE_ORDER}")
    stages = {s.name: s for s in build_stages(cfg)}
    run_order = [n for n in STAGE_ORDER if n in wanted]
    plan_problems = parallel_problems(run_order, parallel, stages)
    plan_problems += print_plan(plan, window, cfg.disk_budget, cfg.standing_budget, parallel=parallel)
    if plan_problems:
        print("!!! the plan is refused: " + "; ".join(plan_problems))
        return 2
    if not a.dry_run:
        os.makedirs(cfg.data_root, exist_ok=True)
        if cfg.link_filler_from:
            link_filler(cfg)
        if "filler" in wanted and not cfg.link_filler_from:
            bad = filler_preflight(cfg)
            if bad:
                print("!!! " + "; ".join(bad))
                return 2
        if cfg.offline_stage or cfg.mirror_inbox:
            print("offline preflight")
            filler_done = os.path.exists(marker_path(cfg, "filler")) and not a.force
            bad = offline_report(cfg, [n for n in run_order
                                       if not (n == "filler" and (cfg.link_filler_from or filler_done))])
            if bad:
                for line in bad:
                    print("!!! " + line)
                print("!!! a needed cache entry is missing; run mirror_sources stage-small and push first")
                return 2
        install_signal_handlers()
    t0 = time.time()
    try:
        for item in run_plan(run_order, parallel if len([n for n in run_order if n in parallel]) > 1 else []):
            if isinstance(item, list):
                if not run_parallel(cfg, item, stages, a.dry_run, a.force):
                    return 1
                continue
            name = item
            if name == "filler" and cfg.link_filler_from:
                print("=== stage filler: skipped, bins linked from", cfg.link_filler_from)
                if not a.dry_run:
                    os.makedirs(cfg.done, exist_ok=True)
                    with open(marker_path(cfg, "filler"), "w") as f:
                        json.dump({"stage": "filler", "linked_from": cfg.link_filler_from}, f)
                continue
            if not run_stage(cfg, stages[name], a.dry_run, a.force):
                return 1
    except Interrupted:
        return 20
    print(f"all requested stages finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
