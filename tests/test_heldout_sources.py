"""The per-document source sidecar on the held-out splits, and the backfill that adds it.

1. ``build_heldout`` writes ``{split}.src`` for both splits, one byte per document, aligned with
   ``.cond``, each byte the index in ``SOURCE_KEYS`` of the question's source.
2. ``EvidenceWriter`` without ``with_source`` writes exactly the eleven files of a training build
   and ignores ``source_idx``; with it, ``truncate_to_state`` trims ``.src`` with the other files.
3. ``--sources-only``'s replay (``backfill_sources``) reproduces the bytes of a first build made
   with a different embedder (the keys are the one file it cannot compare), copies only the two
   ``.src`` files, leaves every existing byte alone, and the result equals a direct build.
4. A tampered ``.bin`` aborts the backfill naming the file, and nothing is written.
5. ``EvidenceDataset`` emits ``source_ids`` (long, ``[B, S]``, -1 on row padding, each
   conversation's byte on its own tokens) when the file exists, omits the key without it, and
   refuses a ``.src`` of the wrong length.
6. The in-context ceiling probe's reader (``read_fixed_groups``) recovers each question, answer,
   per condition passages and source from the split, and its summary is the token weighted pool.

Needs the pruned tokenizer (``utils.TOKENIZER_DIR``); no GPU, no network.
"""
import os, sys, shutil, tempfile, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from transformers import AutoTokenizer

from modules.data.chat import ChatTemplate
from scripts.prepare_evidence_data import (
    EVIDENCE_SUFFIXES, SOURCE_KEYS, SOURCE_UNKNOWN, EvidenceWriter, backfill_sources, build_heldout,
)
from test_prepare_evidence_heldout import FakeEmbedder, hotpot_rows, read_split, squad_rows
from utils import TOKENIZER_DIR

BUILD = dict(max_evidence_tokens=4608, render_batch=8, seed=7)


def file_hashes(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                out[name] = hashlib.sha1(f.read()).hexdigest()
    return out


def source_of(tokenizer, doc):
    text = tokenizer.decode(doc["ids"])
    return "squad_v2" if "fact" in text else "hotpot_qa"


def main():
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}")
        return
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    sources = [{"key": "squad_v2", "render": "squad_v2", "rows": squad_rows()},
               {"key": "hotpot_qa", "render": "hotpot_qa", "rows": hotpot_rows()}]
    tmp = tempfile.mkdtemp(prefix="heldout_src_")
    try:
        with_src = os.path.join(tmp, "with_src")
        existing = os.path.join(tmp, "existing")
        scratch = os.path.join(tmp, "scratch")
        for d in (with_src, existing, scratch):
            os.makedirs(d)

        # 1. aligned with .cond, values match each record's source
        build_heldout(sources, template, FakeEmbedder(), with_src, **BUILD)
        for split in ("evidence_dev", "evidence_fixed"):
            src = np.fromfile(os.path.join(with_src, f"{split}.src"), dtype=np.uint8)
            docs = read_split(with_src, split)
            assert src.shape[0] == len(docs) > 0, split
            for byte, doc in zip(src.tolist(), docs):
                assert SOURCE_KEYS[byte] == source_of(tokenizer, doc), (split, byte)
            assert {SOURCE_KEYS[b] for b in set(src.tolist())} == {"squad_v2", "hotpot_qa"}
        fixed = np.fromfile(os.path.join(with_src, "evidence_fixed.src"), dtype=np.uint8)
        assert all(len(set(fixed[g:g + 4].tolist())) == 1 for g in range(0, len(fixed), 4)), \
            "a question's four rows disagree about their source"
        print("1. build_heldout writes .src aligned with .cond, values match sources    PASS")

        # 2. the writer
        build_heldout(sources, template, FakeEmbedder(), existing, with_source=False, **BUILD)
        assert not [f for f in os.listdir(existing) if f.endswith(".src")]
        assert sorted(f.split(".", 1)[1] for f in os.listdir(existing) if f.startswith("evidence_dev")) \
            == sorted(EVIDENCE_SUFFIXES)
        wdir = os.path.join(tmp, "writer")
        os.makedirs(wdir)
        keys = np.zeros((1, 384), dtype=np.float32)
        plain = EvidenceWriter(wdir, "plain", {})
        plain.write([1, 2], [0, 1], [3], [0], keys, [1], 0, 1, source_idx=3)
        plain.close()
        assert not os.path.exists(os.path.join(wdir, "plain.src"))
        state = {}
        tagged = EvidenceWriter(wdir, "tagged", state, with_source=True)
        for source in (0, 1, 2):
            tagged.write([1, 2], [0, 1], [3], [0], keys, [1], 0, 1, source_idx=source)
        tagged.write([1, 2], [0, 1], [3], [0], keys, [1], 0, 1)
        tagged.close()
        assert np.fromfile(os.path.join(wdir, "tagged.src"), dtype=np.uint8).tolist() == \
            [0, 1, 2, SOURCE_UNKNOWN]
        resumed = EvidenceWriter(wdir, "tagged", {"doc_count": 2, "tokens_written": 4, "ev_tokens": 2,
                                                   "chunks_written": 2}, with_source=True)
        resumed.close()
        assert os.path.getsize(os.path.join(wdir, "tagged.src")) == 2
        print("2. writer: no .src without with_source, trimmed to doc_count on resume   PASS")

        # 3. backfill reproduces the bytes (keys aside) and adds only the sidecars
        before = file_hashes(existing)
        written = backfill_sources(sources, template, existing, scratch, **BUILD)
        after = file_hashes(existing)
        assert sorted(written) == ["evidence_dev.src", "evidence_fixed.src"], written
        assert set(after) - set(before) == set(written), set(after) - set(before)
        assert all(after[k] == v for k, v in before.items()), "an existing file changed"
        assert not os.listdir(scratch), "the replay's scratch directory was not removed"
        for split in ("evidence_dev", "evidence_fixed"):
            assert after[f"{split}.src"] == file_hashes(with_src)[f"{split}.src"]
        print("3. backfill: replay matches, only .src added, equals a direct build     PASS")

        # 4. a tampered split aborts it and writes nothing
        tampered = os.path.join(tmp, "tampered")
        shutil.copytree(with_src, tampered)
        for split in ("evidence_dev", "evidence_fixed"):
            os.remove(os.path.join(tampered, f"{split}.src"))
        path = os.path.join(tampered, "evidence_fixed.bin")
        raw = bytearray(open(path, "rb").read())
        raw[len(raw) // 2] ^= 0xFF
        open(path, "wb").write(bytes(raw))
        try:
            backfill_sources(sources, template, tampered, scratch, **BUILD)
            raise AssertionError("a tampered .bin was accepted")
        except SystemExit as e:
            assert "evidence_fixed.bin" in str(e), str(e)
        assert not [f for f in os.listdir(tampered) if f.endswith(".src") or f.endswith(".part")]
        assert not os.listdir(scratch)
        print("4. tampered .bin aborts the backfill, names the file, writes nothing      PASS")

        # 5. the dataset
        from modules.data.evidence_dataset import EvidenceDataset
        for split in ("evidence_dev", "evidence_fixed"):
            ds = EvidenceDataset(with_src, tokenizer, batch_size=2, max_length=4096, split=split,
                                 num_mtp_tokens=1, shuffle=False, max_evidence_tokens=12288)
            assert ds.has_source_labels
            seen = set()
            for batch in ds:
                ids = batch["source_ids"]
                assert str(ids.dtype) == "torch.int64"
                assert ids.shape == batch["input_ids"].shape
                assert bool((ids[:, -1] == -1).all()), "the trailing pad token carries a source"
                seen |= set(ids.unique().tolist())
                # each real segment holds one source value
                for r in range(ids.shape[0]):
                    for seg in batch["document_ids"][r].unique().tolist():
                        values = ids[r][batch["document_ids"][r] == seg].unique().tolist()
                        assert len(values) == 1, values
            assert seen == {-1, SOURCE_KEYS.index("squad_v2"), SOURCE_KEYS.index("hotpot_qa")}, seen
        no_src = os.path.join(tmp, "no_src")
        shutil.copytree(with_src, no_src)
        os.remove(os.path.join(no_src, "evidence_dev.src"))
        ds = EvidenceDataset(no_src, tokenizer, batch_size=2, max_length=4096, split="evidence_dev",
                             num_mtp_tokens=1, shuffle=False, max_evidence_tokens=12288)
        assert not ds.has_source_labels
        assert all("source_ids" not in batch for batch in ds)
        with open(os.path.join(no_src, "evidence_fixed.src"), "r+b") as f:
            f.truncate(os.path.getsize(f.name) - 1)
        try:
            EvidenceDataset(no_src, tokenizer, batch_size=2, max_length=4096, split="evidence_fixed",
                            num_mtp_tokens=1, shuffle=False, max_evidence_tokens=12288)
            raise AssertionError("a short .src was accepted")
        except ValueError as e:
            assert "evidence_fixed.src" in str(e)
        print("5. EvidenceDataset: source_ids with -1 padding, omitted without the file   PASS")

        # 6. the in-context ceiling reads the split back: question, answer, passages, source
        from scripts.evidence_ceiling_probe import (
            build_fixed_conditions, read_fixed_groups, summarize_fixed,
        )
        groups = read_fixed_groups(with_src, "evidence_fixed", tokenizer, template, None)
        n_questions = len(read_split(with_src, "evidence_fixed")) // 4
        assert len(groups) == n_questions
        assert len(read_fixed_groups(with_src, "evidence_fixed", tokenizer, template, 5)) == 5
        for g in groups:
            if g["source"] == "squad_v2":
                assert g["question"].startswith("What is fact") and g["answer"].startswith("answer")
                assert "Context number" in g["passages"]["gold"]
                assert g["passages"]["gold"] in g["passages"]["mixed"]
            else:
                assert g["source"] == "hotpot_qa" and g["answer"].startswith("hop answer")
                assert "Paragraph" in g["passages"]["gold"]
            assert g["passages"]["none"] == "" and g["passages"]["distractors"]
            assert g["passages"]["gold"] not in g["passages"]["distractors"]
        conditions, kept = build_fixed_conditions(groups, template, 3800)
        assert len(kept) == n_questions and all(len(v) == n_questions for v in conditions.values())
        assert build_fixed_conditions(groups, template, 5)[1] == []
        per_source = {"a": {c: {"ce": 1.0 + i, "tokens": 10} for i, c in enumerate(
            ("gold", "mixed", "distractors", "none"))},
                      "b": {c: {"ce": 2.0 + i, "tokens": 30} for i, c in enumerate(
            ("gold", "mixed", "distractors", "none"))}}
        summary = summarize_fixed(per_source, {"a": 4, "b": 6})
        assert summary["by_source"]["a"]["ceiling"] == 3.0 and summary["by_source"]["b"]["content"] == 2.0
        assert abs(summary["all"]["gold"] - (1.0 * 10 + 2.0 * 30) / 40) < 1e-12
        assert summary["all"]["questions"] == 10 and summary["all"]["tokens"] == 40
        print("6. ceiling probe reads question, answer, passages and source back        PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("all held-out source checks passed")


if __name__ == "__main__":
    main()
