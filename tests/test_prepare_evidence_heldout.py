"""The held-out evidence splits: what ``--heldout`` writes is paired, unleaked and readable.

The fixed split exists so a trainer can teacher force ONE target under every condition and read
CE(none) - CE(gold) as what the evidence is worth. That only means anything if the four rows of a
question really differ in evidence alone, so this checks the properties the measurement leans on:

1. the fixed split is question-major, four rows per question in the order gold, mixed,
   distractors, none, with identical prompt ids and mask inside a group;
2. every fixed row is answerable (``ans`` 1) with the real answer as target, natively unanswerable
   questions never appear, gold is flagged in gold/mixed and nowhere in distractors, and ``none``
   has no chunks and no evidence tokens;
3. no distractor chunk equals its own row's gold text in either split (SQuAD dev shares contexts
   between questions, so an unfiltered pool would leak it), and near-duplicate contexts do not
   oversample the pool;
4. the dev split only uses QA conditions and its targets follow ``apply_condition``'s rule;
5. a question whose evidence overflows is dropped whole from the fixed split;
6. ``EvidenceDataset`` reads both splits back;
7. the training path (``build_corpus`` + ``apply_condition`` with a reservoir) is unchanged by the
   pool parameter: it consumes the same rng draws whether or not the parameter exists.

Needs the pruned tokenizer (``utils.TOKENIZER_DIR``); no GPU.
"""
import os, sys, random, shutil, tempfile, hashlib
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from modules.data import abstention
from modules.data.chat import ChatTemplate
from scripts import prepare_evidence_data as ped
from scripts.prepare_evidence_data import (
    CONDITIONS, EMBED_DIM, FIXED_CONDITIONS, QA_CONDITIONS, EvidenceRow, apply_condition,
    build_heldout,
)
from utils import TOKENIZER_DIR


class FakeEmbedder:
    """Deterministic unit vectors from the text, so equal texts get equal keys."""

    def encode(self, texts):
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")
            v = np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def squad_rows(n_contexts=12, per_context=3):
    """Several questions per context (as in dev) and every third question unanswerable."""
    rows = []
    for c in range(n_contexts):
        context = f"Context number {c} talks about topic {c} in some detail. " * 3
        for q in range(per_context):
            answerable = (c * per_context + q) % 3 != 0
            rows.append({
                "context": context, "question": f"What is fact {c}-{q}?",
                "answers": {"text": [f"answer {c} {q}"] if answerable else []},
            })
    return rows


def hotpot_rows(n=10):
    rows = []
    for i in range(n):
        titles = [f"T{i}_{j}" for j in range(6)]
        rows.append({
            "question": f"Which hop {i}?", "answer": f"hop answer {i}",
            "context": {"title": titles,
                        "sentences": [[f"Paragraph {i} {j} has words about {i} {j}. "] for j in range(6)]},
            "supporting_facts": {"title": titles[:2], "sent_id": [0, 0]},
        })
    return rows


def read_split(data_dir, split):
    """Everything the writer put on disk, per document."""
    def load(suffix, dtype):
        return np.fromfile(os.path.join(data_dir, f"{split}.{suffix}"), dtype=dtype)
    ids, idx, mask = load("bin", np.uint16), load("idx", np.uint64), load("mask", np.uint8)
    ev, evidx, evchunk = load("ev", np.uint16), load("evidx", np.uint64), load("evchunk", np.uint16)
    evkeyidx, evgold = load("evkeyidx", np.uint64), load("evgold", np.uint8)
    cond, ans = load("cond", np.uint8), load("ans", np.uint8)
    docs = []
    for i in range(len(cond)):
        a, b = int(idx[i]), int(idx[i + 1])
        ea, eb = int(evidx[i]), int(evidx[i + 1])
        ca, cb = int(evkeyidx[i]), int(evkeyidx[i + 1])
        chunks = []
        for c in range(cb - ca):
            chunks.append(tuple(ev[ea:eb][evchunk[ea:eb] == c].tolist()))
        docs.append({
            "ids": ids[a:b].tolist(), "mask": mask[a:b].tolist(), "ev_tokens": eb - ea,
            "chunks": chunks, "gold": evgold[ca:cb].tolist(), "cond": CONDITIONS[cond[i]],
            "ans": int(ans[i]),
        })
    return docs


def main():
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}")
        return
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    enc = lambda text: tuple(tokenizer(text, add_special_tokens=False)["input_ids"])

    sq, hp = squad_rows(), hotpot_rows()
    n_squad_answerable = sum(1 for r in sq if r["answers"]["text"])
    sources = [{"key": "squad_v2", "render": "squad_v2", "rows": sq},
               {"key": "hotpot_qa", "render": "hotpot_qa", "rows": hp}]
    tmp = tempfile.mkdtemp(prefix="heldout_")
    try:
        result = build_heldout(sources, template, FakeEmbedder(), tmp, max_evidence_tokens=4608,
                               render_batch=8, seed=7)
        dev, fixed = read_split(tmp, "evidence_dev"), read_split(tmp, "evidence_fixed")
        assert len(dev) == len(sq) + len(hp), (len(dev), len(sq) + len(hp))
        assert len(fixed) == 4 * (n_squad_answerable + len(hp)), len(fixed)

        # 1. groups of four, in order, identical prompts
        for g in range(0, len(fixed), 4):
            group = fixed[g:g + 4]
            assert tuple(d["cond"] for d in group) == FIXED_CONDITIONS, [d["cond"] for d in group]
            assert all(d["ids"] == group[0]["ids"] and d["mask"] == group[0]["mask"] for d in group)
        # sources interleaved: the first quarter of the fixed split already holds both
        # (a group's prompt says which source by its question text)
        first = fixed[:len(fixed) // 2 // 4 * 4]
        texts = [tokenizer.decode(d["ids"]) for d in first[::4]]
        assert any("fact" in t for t in texts) and any("hop" in t for t in texts), "not interleaved"
        print("1. fixed split: 4-row groups, order, identical prompts, interleaved  PASS")

        # 2. labels
        for g in range(0, len(fixed), 4):
            gold_d, mixed_d, dist_d, none_d = fixed[g:g + 4]
            assert all(d["ans"] == 1 for d in fixed[g:g + 4])
            assert any(gold_d["gold"]) and all(gold_d["gold"])
            assert any(mixed_d["gold"]) and not all(mixed_d["gold"])
            assert dist_d["chunks"] and not any(dist_d["gold"])
            assert none_d["chunks"] == [] and none_d["gold"] == [] and none_d["ev_tokens"] == 0
        # the target is the real answer: it decodes out of the supervised span
        for d in fixed[::4]:
            supervised = tokenizer.decode([t for t, m in zip(d["ids"], d["mask"]) if m])
            assert ("answer" in supervised) or ("hop answer" in supervised), supervised
        all_supervised = " ".join(
            tokenizer.decode([t for t, m in zip(d["ids"], d["mask"]) if m]) for d in fixed)
        for phrase in abstention.ABSTENTIONS_PASSAGE_TRAIN:
            assert phrase not in all_supervised, "an abstention leaked into the fixed split"
        # unanswerable questions never appear: fact 0-x, 1-0.. are unanswerable in the fixture
        fixed_prompts = " ".join(tokenizer.decode(d["ids"]) for d in fixed[::4])
        unanswerable = [f"fact {c}-{q}?" for c in range(12) for q in range(3)
                        if not sq[c * 3 + q]["answers"]["text"]]
        assert unanswerable and not any(u in fixed_prompts for u in unanswerable)
        print("2. fixed labels: ans 1, gold flags, none empty, no unanswerable      PASS")

        # 3. no distractor equals its own gold, in either split
        for split_docs in (dev, fixed):
            for d in split_docs:
                gold_chunks = [c for c, g in zip(d["chunks"], d["gold"]) if g]
                for c, g in zip(d["chunks"], d["gold"]):
                    if not g:
                        assert c not in gold_chunks, "a distractor equals the row's gold"
        # and a SQuAD `distractors` buffer never carries the question's own context
        by_prompt = {}
        for r in sq:
            by_prompt[r["question"]] = enc(r["context"].strip())
        for split_docs in (dev, fixed):
            for d in split_docs:
                if d["cond"] not in ("distractors", "none"):
                    continue
                text = tokenizer.decode(d["ids"])
                own = [ctx for q, ctx in by_prompt.items() if f"Question: {q}" in text]
                if own:
                    assert own[0] not in d["chunks"], "own gold in a distractors buffer"
        assert result["gold_rejected"]["squad_v2"] > 0, "the exclusion was never exercised"
        assert result["pool_sizes"]["squad_v2"] == 12, result["pool_sizes"]
        print(f"3. no distractor equals own gold (rejected {result['gold_rejected']['squad_v2']} "
              f"draws), pool deduplicated                       PASS")

        # 4. dev conditions and targets
        assert all(d["cond"] in QA_CONDITIONS for d in dev)
        squad_answerable_q = {r["question"]: bool(r["answers"]["text"]) for r in sq}
        for d in dev:
            text = tokenizer.decode(d["ids"])
            supervised = tokenizer.decode([t for t, m in zip(d["ids"], d["mask"]) if m])
            refusal = any(p in supervised for p in abstention.ABSTENTIONS_PASSAGE_TRAIN)
            native_unanswerable = any(f"Question: {q}" in text and not a
                                      for q, a in squad_answerable_q.items())
            assert refusal == (d["cond"] in ("distractors", "none") or native_unanswerable), \
                (d["cond"], native_unanswerable, supervised)
            assert d["ans"] == (0 if refusal else 1)
        assert len({d["cond"] for d in dev}) >= 3, "dev drew too few conditions to test anything"
        print("4. dev split: QA conditions, targets follow apply_condition          PASS")

        # 5. drop-whole: a cap under one chunk drops every question, and nothing half written
        tmp2 = tempfile.mkdtemp(prefix="heldout_cap_")
        try:
            capped = build_heldout(sources, template, FakeEmbedder(), tmp2,
                                   max_evidence_tokens=120, render_batch=8, seed=7)
            docs = read_split(tmp2, "evidence_fixed")
            assert len(docs) % 4 == 0
            for g in range(0, len(docs), 4):
                assert tuple(d["cond"] for d in docs[g:g + 4]) == FIXED_CONDITIONS
            kept = len(docs) // 4
            dropped = sum(s["drops"].get("evidence_too_long", 0)
                          for s in capped["splits"]["evidence_fixed"]["sources"].values())
            assert dropped > 0 and kept > 0 and kept + dropped == n_squad_answerable + len(hp), (kept, dropped)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        print(f"5. overflow drops the whole question ({dropped} dropped, {kept} kept)   PASS")

        # 6. the dataset can read both back
        try:
            from modules.data.evidence_dataset import EvidenceDataset
            for split, n in (("evidence_dev", len(dev)), ("evidence_fixed", len(fixed))):
                ds = EvidenceDataset(tmp, tokenizer, batch_size=2, max_length=4096, split=split,
                                     num_mtp_tokens=1, shuffle=False, max_evidence_tokens=12288)
                batches = list(iter(ds))
                assert batches and "evidence_ids" in batches[0], split
                assert ds.has_condition_labels
            print("6. EvidenceDataset reads both splits                                PASS")
        except ImportError as e:
            print(f"6. SKIPPED (EvidenceDataset needs {e})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 7. the training call path is unchanged by the pool parameter
    row = EvidenceRow(question="q", answer="a", gold=["g1", "g2"], near=[f"n{i}" for i in range(5)])
    for condition in CONDITIONS:
        for seed in range(20):
            reservoir = deque([f"r{i}" for i in range(9)])
            rng_a, rng_b = random.Random(seed), random.Random(seed)
            with_default = apply_condition(row, condition, reservoir, rng_a, 3)
            explicit_none = apply_condition(row, condition, reservoir, rng_b, 3, pool=None)
            assert with_default == explicit_none
            assert rng_a.random() == rng_b.random()
    print("7. training path: pool=None is the default path                     PASS")
    print("all held-out checks passed")


if __name__ == "__main__":
    main()
