"""Build the Run 2 corpus with the existing builders: the pilot's three merges at scale.

The recipe is the pilot's (``ckpts/inject/build_pilot_bios.sh``, ``merge_pilot.sh``,
``merge_pilot2.sh``) scaled to the token target of ``docs/plans/RUN2.md``, with
``--support-gold-only`` on the biography builds, the written bridge chains beside the plain ones, QA
at its finite passes and the filler from ``prepare_data.py``. No builder is changed here: the two
things the builders lack (a fineweb prefix covering more than one crawl, a hop mix with 4-hop
training rows, files already used by an earlier phase) are done by patching the imported module in
``wrap-prepare-data`` and ``wrap-chains`` below, in memory, before its ``main`` runs.

The derivation of every number is ``derive()``; ``python scripts/build_run2_corpus.py plan`` prints
it as tables and ``docs/plans/RUN2_corpus.md`` carries them with their sources.

Stages, in order (``--stage NAME[,NAME]`` runs some, default all):

    filler        prepare_data: phase1, phase2, ir (the manifest key rule: ir needs its own key)
    bios_main     prepare_injection_data: arms full and retrieval, swap 0.15, gold-only support
    bios_warm     prepare_injection_data: arm retrieval, swap 1.0, gold-only support (the warm-up form)
    qa            prepare_evidence_data: SQuAD v2, HotpotQA, web text, chat replay
    chains_plain  prepare_chain_data: hops 1 / 2 / 3 at 25 / 50 / 25
    chains_bridge prepare_chain_data --bridge: hops 1 to 4 at 25 each
    merge_warm    merge_evidence_splits: the warm-up split, defines the document order
    merge_main    merge_evidence_splits --order-from the warm-up split
    merge_control merge_evidence_splits --no-evidence, the short matched control
    heldout_qa    prepare_evidence_data --heldout: evidence_dev, evidence_fixed
    heldout_chains prepare_chain_data: chains_eval, chains_heldout_tmpl, chains_hop4
    val_clean     prepare_clean_val: inject_val_clean
    store_openqa  build_store: the A4 / A7 store; store_edit: its edit twin

A stage prints each command, runs it through bash (``cd ROOT && source ENV_INIT && ...``, the way
``tests/run_tests.sh`` does, with ``TINY_LLM_ROOT`` and ``TINY_LLM_ENV_INIT`` read from the
environment), tees it to ``DATA_ROOT/run2_logs/STAGE.log``, checks its expected outputs and writes a
marker ``DATA_ROOT/run2_done/STAGE.json``. A stage with a marker and its outputs present is skipped
(``--force`` reruns it), so a rerun after a failure resumes at the first unfinished stage.

Usage:

    python scripts/build_run2_corpus.py plan [--scale S]
    python scripts/build_run2_corpus.py build --data-root DIR [--dry-run] [--scale S] [--stage A,B]
    python scripts/build_run2_corpus.py wrap-prepare-data [opts] -- PREPARE_DATA_ARGS
    python scripts/build_run2_corpus.py wrap-chains [--hops 4] -- PREPARE_CHAIN_DATA_ARGS

Cluster (day 1, ``TINY_LLM_ENV_INIT=/dev/null`` as the rented box has no env_init):

    TINY_LLM_ROOT=/workspace/tiny-llm TINY_LLM_ENV_INIT=/dev/null HF_TOKEN=... \\
        nohup python scripts/build_run2_corpus.py build --data-root /data/run2 --device cuda \\
        > /data/run2/build.log 2>&1 &
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

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
BIOS_CHUNKS_PER_TOKEN = 648_000 / 599_638_781            # merge.json, bios slice
BIOS_EVTOK_PER_TOKEN = 0.05006
QA_CHUNKS_PER_TOKEN = 1_480_692 / 210_607_046
QA_EVTOK_PER_TOKEN = 0.9624
KEY_BYTES = 768          # fp16 bge-small key per chunk occurrence
TOK_BYTES = 2            # uint16 per token, prompt or evidence (ev and evchunk each)
TOKENS_PER_DOC = {"bios": 616.0, "qa": 349.0, "plain": PLAIN_TOKENS_PER_QUESTION,
                  "bridge": BRIDGE_TOKENS_PER_QUESTION}   # pilot documents per token
MERGE_DOCS_PER_SECOND = 3_488_744 / (30 * 60)            # pilot_main_train: 30 minutes for 3.49M documents
INJECT_HOURS_PER_B = (5 / 60) / 0.6                      # one 600M build in about 5 minutes (12:40 to 12:45)
CHAIN_HOURS_PER_B = (55 / 60) / 0.10466                  # 2M questions in about 55 minutes on cuda
WEBTEXT_PROMPT_PER_IR_TOKEN = 60_092_611 / 149_321_284   # the pilot QA build: ir tokens read per prompt token
QA_ARM_WEIGHTS = {"squad_v2": 0.20, "hotpot_qa": 0.15, "webtext": 0.40, "smoltalk2": 0.12,
                  "ultrachat": 0.08, "no_robots": 0.05}
QA_ARM_PER_PASS = {"squad_v2": 5_862_565, "hotpot_qa": 4_861_098}     # prompt tokens, pilot state file
QA_ARM_SUPPLY = {"ultrachat": 242e6, "no_robots": 5.3e6}               # ultrachat: 1,163 tokens x 207k rows
SMOLTALK2_SUPPLY = 1.8e9     # 24 no_think SFT files at about 75M tokens (the 16B manifest: 675M in 9 files)

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
TOKEN_TARGET = 280e9
EPOCHS = 3


@dataclass
class Plan:
    """Every free number of the corpus, each with its source in docs/plans/RUN2_corpus.md.

    Args:
        total: prompt tokens of the warm-up and main splits, TOKEN_TARGET / EPOCHS (RUN2 had 100B unique at up to 4 epochs; the trainer runs whole epochs, so 280B / 3).
        run_tokens: tokens the run trains on (RUN2 decision 6: 280B), so epochs = run_tokens / total.
        warm_max: the most the warm-up is read for before the stop rule (RUN2 Schedule: 28B).
        control: prompt tokens of the short matched control (RUN2 decision 8: 10B).
        chain_share: share of the merged split for chains, plain and bridge together (RUN2: 10%, costs 7.3 TB and 3 to 4 days at the old size; default 0.03).
        bridge_fraction: the bridge's part of the chain share (assumption, not in RUN2).
        qa_target: the QA builder's prompt token target (assumption, see the sheet).
        phase2_tokens: prepare_data phase 2 target; ir_tokens: its ir target (assumptions, smoltalk2
            supply is the limit, see the sheet).
        margin: extra filler built over the need (windows set aside for validation, the filler repeat).
        persons_per_tier: the biography tier structure (the pilot's, unchanged).
        scale: multiplies every token quantity above (smoke runs).
    """
    total: float = TOKEN_TARGET / EPOCHS
    run_tokens: float = TOKEN_TARGET
    warm_max: float = 28e9
    control: float = 10e9
    chain_share: float = 0.03
    bridge_fraction: float = 0.5
    qa_target: float = 2.5e9
    phase2_tokens: float = 5e9
    ir_tokens: float = 2.5e9
    margin: float = 0.01
    persons_per_tier: str = "1000:100,100:500,10:1000,1:2000"
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


def derive(plan: Plan) -> dict:
    """Every target of the pipeline from the plan, as one dict of plain numbers."""
    total = plan.scaled("total")
    chains = plan.chain_share * total
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
    plain_q = plain / PLAIN_TOKENS_PER_QUESTION
    bridge_q = bridge / BRIDGE_TOKENS_PER_QUESTION
    tiers = parse_tiers(plan.persons_per_tier)
    bio_renders = sum(t * c for t, c in tiers.items())
    out = {
        "total": total, "epochs": plan.run_tokens / plan.total, "warm_max": plan.scaled("warm_max"),
        "control": plan.scaled("control"), "bios": bios, "qa": qa, "qa_arms": qa_arms,
        "qa_target": plan.scaled("qa_target"), "chains": chains, "plain": plain, "bridge": bridge,
        "phase1": phase1, "phase2": phase2, "ir": ir,
        "plain_questions": int(plain_q * 1.05), "bridge_questions": int(bridge_q * 1.05),
        "bio_renders": bio_renders, "persons": sum(tiers.values()),
        "shares": {"bios": bios / total, "qa": qa / total, "chains": plain / total,
                   "bridge": bridge / total},
        "val_fraction": min(0.005, plan.val_windows / max(1.0, bios / 721.0)),
    }
    # the bios slice is taken whole (share above its size, one pass): the warm-up and main builds differ
    # in token length (swaps), and --order-from needs the same document count per slice in both merges
    out["merge_shares"] = dict(out["shares"], bios=out["shares"]["bios"] * 1.02)
    out["injection_target"] = int(bios)
    out["qa_val_fraction"] = 0.001
    return out


def filler_supply(plan: Plan, d: dict) -> List[dict]:
    """Per source, tokens needed across the three phases against what the Hub holds."""
    rows = []
    for src, sup in SOURCE_SUPPLY.items():
        need = 0.0
        for phase in ("phase1", "phase2", "ir"):
            w = PHASE_WEIGHTS[phase]
            if src in w:
                need += w[src] / sum(w.values()) * d[phase]
        s = plan.scale
        row = {"source": src, "need": need, "per_file": sup["per_file"],
               "files_needed": int(-(-need // sup["per_file"])) if need else 0,
               "supply_default": sup["files_default"] * sup["per_file"] * s,
               "supply_patched": sup["files_all_crawls"] * sup["per_file"] * s}
        rows.append(row)
    return rows


def estimate(plan: Plan, d: dict) -> dict:
    """Disk (bytes) and one-process hours per stage, scaled linearly from the pilot logs."""
    bios, qa, plain, bridge = d["bios"], d["qa"], d["plain"], d["bridge"]
    hop_docs = TOKENS_PER_DOC
    b_chunks = BRIDGE_CHUNKS_PER_QUESTION / BRIDGE_TOKENS_PER_QUESTION
    b_ev = BRIDGE_EVTOK_PER_QUESTION / BRIDGE_TOKENS_PER_QUESTION

    def evbytes(tokens, chunks_per_tok, ev_per_tok):
        return tokens * (chunks_per_tok * KEY_BYTES + ev_per_tok * 2 * TOK_BYTES)

    def split_bytes(tokens, chunks_per_tok, ev_per_tok, plain_only=False):
        base = tokens * (TOK_BYTES + 1)
        return base if plain_only else base + evbytes(tokens, chunks_per_tok, ev_per_tok)

    filler_tokens = d["phase1"] + d["phase2"] + d["ir"]
    disk = {
        "filler (phase1, phase2, ir bins)": filler_tokens * TOK_BYTES,
        "bios main build (full + retrieval)": bios * (4 + 4 + 0.83 + 0.2),
        "bios warm build (retrieval)": bios * (4 + 0.83 + 0.2),
        "qa build": split_bytes(qa, QA_CHUNKS_PER_TOKEN, QA_EVTOK_PER_TOKEN),
        "chains plain build": split_bytes(plain, PLAIN_CHUNKS_PER_TOKEN, PLAIN_EVTOK_PER_TOKEN),
        "chains bridge build": split_bytes(bridge, b_chunks, b_ev),
    }
    merged = (split_bytes(bios, BIOS_CHUNKS_PER_TOKEN, BIOS_EVTOK_PER_TOKEN) + split_bytes(qa, QA_CHUNKS_PER_TOKEN, QA_EVTOK_PER_TOKEN)
              + split_bytes(plain, PLAIN_CHUNKS_PER_TOKEN, PLAIN_EVTOK_PER_TOKEN) + split_bytes(bridge, b_chunks, b_ev))
    disk["merge warm"] = merged
    disk["merge main"] = merged
    ctl = d["control"] / d["total"]
    disk["merge control (no evidence)"] = (bios + qa + plain + bridge) * ctl * (TOK_BYTES + 1)
    hours = {
        "filler": None,
        "bios main build": bios / 1e9 * INJECT_HOURS_PER_B * 2,
        "bios warm build": bios / 1e9 * INJECT_HOURS_PER_B,
        "qa build": None,
        "chains plain build": plain / 1e9 * CHAIN_HOURS_PER_B,
        "chains bridge build": bridge / 1e9 * CHAIN_HOURS_PER_B,
    }
    docs = (bios / hop_docs["bios"] + qa / hop_docs["qa"] + plain / hop_docs["plain"]
            + bridge / hop_docs["bridge"])
    hours["merge warm"] = docs / MERGE_DOCS_PER_SECOND / 3600
    hours["merge main"] = hours["merge warm"]
    hours["merge control"] = hours["merge warm"] * ctl * 0.1   # no evidence files: 2 min against 30 in the pilot
    return {"disk": disk, "hours": hours, "merge_docs": docs}


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
    def bios_store(self) -> str:
        return os.path.join(self.index, "run2_bios")


MAIN_SUFFIX = "s15a50g1"
WARM_SUFFIX = "s100a50g1"
QA_PREFIX = "evidence_run2"
PLAIN_PREFIX = "chains_run2"
BRIDGE_PREFIX = "chains_bridge_run2"


def share(v: float) -> str:
    return f"{v:.6f}"


def merge_slices(cfg: Cfg, d: dict, bios_split: str) -> List[str]:
    """The four ``--slice`` arguments; the order is fixed because ``--order-from`` replays it."""
    return [
        f"bios={os.path.join(cfg.run2, bios_split)}:{share(d['merge_shares']['bios'])}:1",
        f"qa={os.path.join(cfg.prepared, QA_PREFIX + '_train')}:{share(d['shares']['qa'])}:1",
        f"chains={os.path.join(cfg.prepared, PLAIN_PREFIX + '_train')}:{share(d['shares']['chains'])}:1",
        f"bridge={os.path.join(cfg.prepared, BRIDGE_PREFIX + '_train')}:{share(d['shares']['bridge'])}:1",
    ]


def split_files(prefix: str, suffixes: Sequence[str]) -> List[str]:
    return [f"{prefix}.{s}" for s in suffixes]


@dataclass
class Stage:
    name: str
    commands: List[str]
    outputs: List[str]
    note: str = ""


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

    p1 = (f"{me} wrap-prepare-data --fineweb-prefix {q(cfg.fineweb_prefix)} -- "
          f"--data-dir {q(P)} --phases phase1 --phase1-tokens {int(d['phase1'])} "
          f"--manifest-key data_prep --seed {plan.seed}")
    p2 = (f"{me} wrap-prepare-data --fineweb-prefix {q(cfg.fineweb_prefix)} --exclude-files-of phase1 -- "
          f"--data-dir {q(P)} --phases phase2 --phase2-tokens {int(d['phase2'])} "
          f"--manifest-key data_prep --seed {plan.seed}")
    ir = (f"{me} wrap-prepare-data --fineweb-prefix {q(cfg.fineweb_prefix)} --exclude-files-of phase1,phase2 -- "
          f"--data-dir {q(P)} --phases ir --ir-tokens {int(d['ir'])} --manifest-key ir_prep --seed {plan.seed}")
    stages.append(Stage("filler", [p1, p2, ir],
                        [os.path.join(P, f"{ph}.{s}") for ph in ("phase1", "phase2", "ir") for s in ("bin", "idx")],
                        "prepare_data; HF_TOKEN needed for the gated nemotron_math"))

    def inject(split_arms: str, swap: float, suffix: str) -> str:
        return (f"{py}prepare_injection_data.py --out-dir {q(R)} --filler-dir {q(filler_dir)} "
                f"--filler-phases phase1,phase2 --store-dir {q(cfg.bios_store)} "
                f"--target-tokens {d['injection_target']} --seq-length 4096 --swap-rate {swap} --anon-rate 0.5 "
                f"--gold-drop-rate 0.2 --distractors 3 --persons-per-tier {tiers} "
                f"--val-fraction {d['val_fraction']:.8f} --seed {plan.seed} --arms {split_arms} "
                f"--suffix {suffix}{gold} --device {cfg.device}")
    full_name = f"inject_full_{MAIN_SUFFIX}_train"
    main_name = f"inject_retrieval_{MAIN_SUFFIX}_train"
    warm_name = f"inject_retrieval_{WARM_SUFFIX}_train"
    val = [os.path.join(R, f"inject_val.{s}") for s in ("bin", "idx", "mask")]
    stages.append(Stage("bios_main", [inject("full,retrieval", 0.15, MAIN_SUFFIX)],
                        split_files(os.path.join(R, full_name), PLAIN_FILES + ("factspan",))
                        + split_files(os.path.join(R, main_name), EVIDENCE_FILES + ("factspan",)) + val,
                        "swap 0.15, anonymization 0.5, gold card only supports a span"))
    stages.append(Stage("bios_warm", [inject("retrieval", 1.0, WARM_SUFFIX)],
                        split_files(os.path.join(R, warm_name), EVIDENCE_FILES + ("factspan",)),
                        "the copy-first warm-up form: swap 1.0"))

    held = f" --heldout-max-questions {cfg.heldout_max_questions}" if cfg.heldout_max_questions else ""
    qa = (f"{py}prepare_evidence_data.py --data-dir {q(P)} --target-tokens {int(d['qa_target'])} "
          f"--max-evidence-tokens 4608 --max-source-epochs 2 --split-prefix {QA_PREFIX} --webtext-phase ir "
          f"--val-fraction {d['qa_val_fraction']} --seed {plan.seed} --device {cfg.device}")
    stages.append(Stage("qa", [qa], split_files(os.path.join(P, QA_PREFIX + "_train"), EVIDENCE_FILES),
                        "needs manifest.json data_prep.smoltalk2_holdout_hashes (written by the phase2 build)"))

    def chain(prefix: str, questions: int, bridge: bool) -> str:
        args = (f"--out-dir {q(P)} --prefix {prefix} --splits train,val --train-questions {questions} "
                f"--val-questions 2000 --seed {plan.seed} --device {cfg.device}")
        if bridge:
            return f"{me} wrap-chains --hops 4 -- {args} --bridge"
        return f"{py}prepare_chain_data.py {args}"
    stages.append(Stage("chains_plain", [chain(PLAIN_PREFIX, d["plain_questions"], False)],
                        split_files(os.path.join(P, PLAIN_PREFIX + "_train"), EVIDENCE_FILES), "hops 1 / 2 / 3 at 25 / 50 / 25"))
    stages.append(Stage("chains_bridge", [chain(BRIDGE_PREFIX, d["bridge_questions"], True)],
                        split_files(os.path.join(P, BRIDGE_PREFIX + "_train"), EVIDENCE_FILES),
                        "hops 1 to 4 at 25 each, the written bridge"))

    def merge(split: str, target: float, slices: List[str], extra: str) -> str:
        sl = " ".join(f"--slice {q(s)}" for s in slices)
        return (f"{py}merge_evidence_splits.py --out-dir {q(R)} --split {split} --target-tokens {int(target)} "
                f"--seed {plan.seed} --max-evidence-tokens 4608 {extra}{sl}").replace("  ", " ")
    warm_split = "run2_warm_train"
    main_split = "run2_main_train"
    ctl_split = "run2_control_train"
    stages.append(Stage("merge_warm", [merge(warm_split, d["total"], merge_slices(cfg, d, warm_name), "")],
                        split_files(os.path.join(R, warm_split), EVIDENCE_FILES + ("slice", "merge.json")),
                        "the warm-up split; its .slice file is the document order"))
    order = f"--order-from {q(os.path.join(R, warm_split + '.slice'))} "
    stages.append(Stage("merge_main", [merge(main_split, d["total"], merge_slices(cfg, d, main_name), order)],
                        split_files(os.path.join(R, main_split), EVIDENCE_FILES + ("slice", "merge.json")),
                        "the same document order as the warm-up split, so a switch continues by position"))
    stages.append(Stage("merge_control", [merge(ctl_split, d["control"], merge_slices(cfg, d, full_name), "--no-evidence ")],
                        split_files(os.path.join(R, ctl_split), PLAIN_FILES + ("slice", "merge.json")),
                        "no evidence files, the biographies in the full form, the first 10B of the stream"))

    stages.append(Stage("heldout_qa", [f"{py}prepare_evidence_data.py --heldout --data-dir {q(P)} "
                                       f"--max-evidence-tokens 4608{held} --device {cfg.device}"],
                        split_files(os.path.join(P, "evidence_dev"), EVIDENCE_FILES)
                        + split_files(os.path.join(P, "evidence_fixed"), EVIDENCE_FILES), "SQuAD v2 and HotpotQA dev"))
    stages.append(Stage("heldout_chains", [f"{py}prepare_chain_data.py --out-dir {q(P)} --prefix chains "
                                           f"--splits eval,heldout_tmpl,hop4 --eval-questions-per-hop {cfg.chain_eval_per_hop} "
                                           f"--seed {plan.seed} --device cpu"],
                        sum([split_files(os.path.join(P, f"chains_{n}"), EVIDENCE_FILES + ("chains.jsonl",))
                             for n in ("eval", "heldout_tmpl", "hop4")], []), "same seed, so the held out compositions are the pilot's"))
    stages.append(Stage("val_clean", [f"{py}prepare_clean_val.py --out-dir {q(R)} --filler-dir {q(filler_dir)} "
                                      f"--filler-phases phase1,phase2 --seed {plan.seed} "
                                      f"--val-fraction {d['val_fraction']:.8f} "
                                      f"--exclude-evidence-state {q(os.path.join(P, '_prepare_state_' + QA_PREFIX + '.json'))}"],
                        split_files(os.path.join(R, "inject_val_clean"), ("bin", "idx", "mask", "factspan")),
                        "inject_val windows whose documents the QA builder never read"))
    stages.append(Stage("store_openqa", [f"{py}build_store.py --name openqa --sources nq,triviaqa,hotpotqa "
                                         f"--max-questions {cfg.store_max_questions} --chunk-tokens 128 "
                                         f"--index-root {q(I)} --device {cfg.device} --seed {plan.seed}"],
                        [os.path.join(I, "openqa", "meta.json")], "A4 and A7 store"))
    stages.append(Stage("store_edit", [f"{py}build_store.py --edit-from openqa --name openqa_edit --edit-fraction 0.5 "
                                       f"--neighbour-cos 0.85 --index-root {q(I)} --device cpu --seed {plan.seed}"],
                        [os.path.join(I, "openqa_edit", "meta.json")], "the A4 edit twin"))
    return stages


STAGE_ORDER = ["filler", "bios_main", "bios_warm", "qa", "chains_plain", "chains_bridge", "merge_warm",
               "merge_main", "merge_control", "heldout_qa", "heldout_chains", "val_clean", "store_openqa",
               "store_edit"]


# --------------------------------------------------------------------------- running

def shell_line(cmd: str) -> str:
    return f"cd {q(REPO)} && source {q(ENV_INIT)} && export PYTHONUNBUFFERED=1 && {cmd}"


def check_merge(cfg: Cfg, stage: str) -> List[str]:
    """Problems with a merge's json: all four slices present, md5 lines, warm and main agree."""
    problems = []
    split = {"merge_warm": "run2_warm_train", "merge_main": "run2_main_train",
             "merge_control": "run2_control_train"}[stage]
    path = os.path.join(cfg.run2, split + ".merge.json")
    if not os.path.exists(path):
        return [f"{path} missing"]
    with open(path) as f:
        meta = json.load(f)
    md5 = meta.get("md5_by_slice", {})
    for label in ("bios", "qa", "chains", "bridge"):
        if label not in md5:
            problems.append(f"{split}: md5_by_slice has no {label}")
    if stage == "merge_main":
        wpath = os.path.join(cfg.run2, "run2_warm_train.merge.json")
        if os.path.exists(wpath):
            with open(wpath) as f:
                warm = json.load(f).get("md5_by_slice", {})
            for label in ("qa", "chains", "bridge"):
                if warm.get(label) != md5.get(label):
                    problems.append(f"qa/chains/bridge subsequence differs between warm and main: {label}")
            if warm.get("bios") == md5.get("bios"):
                problems.append("bios slice identical in warm and main: the swap rate did not differ")
    for label, value in md5.items():
        print(f"  md5_by_slice {split} {label} {value}")
    return problems


def run_stage(cfg: Cfg, stage: Stage, dry: bool, force: bool) -> bool:
    marker = os.path.join(cfg.done, stage.name + ".json")
    print(f"=== stage {stage.name}: {stage.note}")
    for c in stage.commands:
        print(f"+ {c}")
    if dry:
        return True
    present = all(os.path.exists(p) for p in stage.outputs)
    if os.path.exists(marker) and present and not force:
        print(f"=== stage {stage.name}: done before (marker and {len(stage.outputs)} outputs present), skipped")
        return True
    os.makedirs(cfg.logs, exist_ok=True)
    os.makedirs(cfg.done, exist_ok=True)
    if os.path.exists(marker):
        os.remove(marker)
    log = os.path.join(cfg.logs, stage.name + ".log")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        for c in stage.commands:
            lf.write(f"\n+ {c}\n")
            lf.flush()
            proc = subprocess.Popen(["bash", "-c", shell_line(c)], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, cwd=REPO)
            for line in proc.stdout:
                lf.write(line)
                sys.stdout.write(line)
            code = proc.wait()
            lf.write(f"exit {code}\n")
            if code != 0:
                print(f"!!! stage {stage.name}: command exited {code}, see {log}")
                return False
    missing = [p for p in stage.outputs if not os.path.exists(p)]
    problems = [f"missing output {p}" for p in missing]
    if stage.name in ("merge_warm", "merge_main", "merge_control") and not missing:
        problems += check_merge(cfg, stage.name)
    if problems:
        for p in problems:
            print(f"!!! stage {stage.name}: {p}")
        return False
    with open(marker, "w") as f:
        json.dump({"stage": stage.name, "seconds": round(time.time() - t0, 1), "commands": stage.commands,
                   "outputs": stage.outputs}, f, indent=1)
    print(f"=== stage {stage.name}: ok in {time.time() - t0:.0f}s")
    return True


def link_filler(cfg: Cfg) -> None:
    """Smoke only: symlink small local filler bins into the prepared dir instead of building them."""
    os.makedirs(cfg.prepared, exist_ok=True)
    for phase in ("phase1", "phase2", "ir"):
        for s in ("bin", "idx"):
            dst = os.path.join(cfg.prepared, f"{phase}.{s}")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(os.path.join(cfg.link_filler_from, f"{phase}.{s}")), dst)


def filler_preflight(cfg: Cfg) -> List[str]:
    """Refuse a filler build that would overwrite the 16B run's manifest entries."""
    path = os.path.join(REPO, "manifest.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        meta = json.load(f)
    if "phase1" in meta.get("data_prep", {}):
        return [f"{path} already holds data_prep.phase1 (the 16B run's counts); the filler build would "
                f"overwrite it. Move manifest.json away first (a fresh cluster has none)"]
    return []


# --------------------------------------------------------------------------- wrappers

def split_dashes(argv: Sequence[str]):
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def wrap_prepare_data(argv: Sequence[str]) -> None:
    """Runs ``scripts/prepare_data.py`` with fineweb's prefix widened and earlier phases' files skipped."""
    own, rest = split_dashes(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--fineweb-prefix", default=None)
    p.add_argument("--exclude-files-of", default="", help="comma separated phases whose files are skipped")
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
    for spec in pd.SOURCES:
        if spec.key == "fineweb" and a.fineweb_prefix:
            spec.file_prefix = a.fineweb_prefix
        skip = used.get(spec.key, set())
        if skip:
            old = spec.file_filter
            spec.file_filter = (lambda f, old=old, skip=skip: f not in skip and (old is None or old(f)))
            print(f"wrap-prepare-data: {spec.key} skips {len(skip)} files an earlier phase used")
    sys.argv = ["prepare_data.py"] + rest
    pd.main()


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


# --------------------------------------------------------------------------- reports

def gb(x: float) -> str:
    return f"{x / 1e9:,.1f} GB" if x < 1e12 else f"{x / 1e12:,.2f} TB"


def print_plan(plan: Plan) -> None:
    d = derive(plan)
    e = estimate(plan, d)
    print(f"TOKEN_TARGET {TOKEN_TARGET:,.0f}  EPOCHS {EPOCHS}  T = {TOKEN_TARGET / EPOCHS:,.0f}")
    print(f"scale {plan.scale}: {d['total']:,.0f} prompt tokens per split, {d['epochs']:.2f} epochs of it in a "
          f"{plan.run_tokens:,.0f} token run")
    print("slice shares of the merged split")
    for k in ("bios", "qa", "chains", "bridge"):
        print(f"  {k:7s} {d[k if k != 'chains' else 'plain']:>18,.0f} tokens  share {d['shares'][k]:.6f}")
    print("qa arms (estimated realised prompt tokens)")
    for k, v in d["qa_arms"].items():
        print(f"  {k:10s} {v:>16,.0f}")
    print(f"filler builds: phase1 {d['phase1']:,.0f}  phase2 {d['phase2']:,.0f}  ir {d['ir']:,.0f}")
    print(f"chain questions: plain {d['plain_questions']:,}  bridge {d['bridge_questions']:,}")
    print(f"biography renderings {d['bio_renders']:,} for {d['persons']:,} persons")
    print("disk (estimate)")
    for k, v in e["disk"].items():
        print(f"  {k:36s} {gb(v)}")
    print(f"  {'total, nothing deleted':36s} {gb(sum(e['disk'].values()))}")
    print("hours, one process (estimate)")
    for k, v in e["hours"].items():
        print(f"  {k:24s} {'unmeasured' if v is None else f'{v:,.1f}'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "wrap-prepare-data":
        wrap_prepare_data(argv[1:])
        return 0
    if argv and argv[0] == "wrap-chains":
        wrap_chains(argv[1:])
        return 0
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["plan", "build"])
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--dry-run", action="store_true", help="print every command, run nothing")
    p.add_argument("--stage", default="all", help="comma separated stage names or all")
    p.add_argument("--force", action="store_true", help="rerun stages that have a marker")
    p.add_argument("--scale", type=float, default=1.0, help="multiplies every token quantity (smoke runs)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total", type=float, default=None)
    p.add_argument("--chain-share", type=float, default=None)
    p.add_argument("--bridge-fraction", type=float, default=None)
    p.add_argument("--qa-target", type=float, default=None)
    p.add_argument("--phase2-tokens", type=float, default=None)
    p.add_argument("--ir-tokens", type=float, default=None)
    p.add_argument("--persons-per-tier", default=None)
    p.add_argument("--no-gold-only", action="store_true")
    p.add_argument("--fineweb-prefix", default="data/")
    p.add_argument("--filler-dir", default=None, help="where the biography build reads the filler bins")
    p.add_argument("--link-filler-from", default=None, help="smoke: symlink local phase bins into prepared/")
    p.add_argument("--heldout-max-questions", type=int, default=None)
    p.add_argument("--store-max-questions", type=int, default=3000)
    p.add_argument("--chain-eval-per-hop", type=int, default=1000)
    p.add_argument("--val-windows", type=int, default=None)
    a = p.parse_args(argv)

    plan = Plan(scale=a.scale, support_gold_only=not a.no_gold_only)
    for name in ("total", "chain_share", "bridge_fraction", "qa_target", "phase2_tokens", "ir_tokens"):
        v = getattr(a, name)
        if v is not None:
            setattr(plan, name, v)
    if a.persons_per_tier:
        plan.persons_per_tier = a.persons_per_tier
    if a.val_windows:
        plan.val_windows = a.val_windows
    if a.mode == "plan":
        print_plan(plan)
        return 0
    cfg = Cfg(data_root=os.path.abspath(a.data_root), plan=plan, device=a.device, filler_dir=a.filler_dir,
              fineweb_prefix=a.fineweb_prefix, heldout_max_questions=a.heldout_max_questions,
              store_max_questions=a.store_max_questions, chain_eval_per_hop=a.chain_eval_per_hop,
              persons_per_tier=a.persons_per_tier, link_filler_from=a.link_filler_from)
    wanted = STAGE_ORDER if a.stage == "all" else a.stage.split(",")
    for w in wanted:
        if w not in STAGE_ORDER:
            p.error(f"unknown stage {w!r}; choose from {STAGE_ORDER}")
    stages = {s.name: s for s in build_stages(cfg)}
    print_plan(plan)
    if not a.dry_run:
        os.makedirs(cfg.data_root, exist_ok=True)
        if cfg.link_filler_from:
            link_filler(cfg)
        if "filler" in wanted and not cfg.link_filler_from:
            bad = filler_preflight(cfg)
            if bad:
                print("!!! " + "; ".join(bad))
                return 2
    t0 = time.time()
    for name in STAGE_ORDER:
        if name not in wanted:
            continue
        if name == "filler" and cfg.link_filler_from:
            print("=== stage filler: skipped, bins linked from", cfg.link_filler_from)
            continue
        if not run_stage(cfg, stages[name], a.dry_run, a.force):
            return 1
    print(f"all requested stages finished in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
