"""The fact injection corpus builder: three arms over one shared document stream.

A small build (``{10: 3, 1: 5}`` people, a synthetic filler corpus, a fake embedder) checked for the
properties the experiment leans on:

1. ``full`` and ``masked`` share ``.bin``, ``.idx`` and ``.factspan`` byte for byte; ``full`` is mask 1
   on every token after BOS; ``masked`` is 0 exactly on the tokens a value span covers;
2. every person is rendered exactly ``tier`` times, and ``.factspan`` matches ``.bin`` in length
   in every split;
3. in ``retrieval`` the mask follows the evidence: a value token is supervised exactly when its
   value occurs in a visible card, a swap shows in the target and in the gold card's copy and nowhere
   else, a placeholder name appears only where the gold card is absent, ``.evgold`` marks the
   person's own card;
4. ``SFTDataset`` reads ``full`` and ``masked`` (fewer supervised labels in ``masked``) and
   ``EvidenceDataset`` reads ``retrieval``;
5. the validation split is filler only, identical for every arm, and disjoint from train;
6. a build is deterministic, a second build with other rates next to the first leaves the shared
   files and the first arms untouched, the filler is cycled with a logged repeat factor when short;
7. the biography store (when ``modules.data.store`` exists) holds one card per person with the
   embedder's key.

GPU free. Needs the pruned tokenizer (``utils.TOKENIZER_DIR``).
"""
import os, sys, json, random, shutil, hashlib, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from modules.data import biographies as bio
from scripts.prepare_evidence_data import CONDITIONS, EMBED_DIM
from scripts.prepare_injection_data import (
    build_injection, load_filler_windows, pick_filler, split_name, tokenize_with_codes,
)
from utils import TOKENIZER_DIR


class FakeEmbedder:
    """Deterministic unit vectors from the text."""

    def encode(self, texts):
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")
            v = np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def write_filler(directory, n_docs, seed):
    rng = np.random.RandomState(seed)
    lengths = rng.randint(300, 3000, size=n_docs)
    tokens = rng.randint(10, 60000, size=int(lengths.sum())).astype(np.uint16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint64)
    os.makedirs(directory, exist_ok=True)
    tokens.tofile(os.path.join(directory, "f.bin"))
    offsets.tofile(os.path.join(directory, "f.idx"))
    return [(os.path.join(directory, "f.bin"), os.path.join(directory, "f.idx"))]


def read_plain(directory, split):
    load = lambda suffix, dtype: np.fromfile(os.path.join(directory, f"{split}.{suffix}"), dtype=dtype)
    ids, idx, mask, codes = load("bin", np.uint16), load("idx", np.uint64), load("mask", np.uint8), \
        load("factspan", np.uint8)
    assert len(codes) == len(ids) == len(mask), split
    return [{"ids": ids[int(idx[i]):int(idx[i + 1])], "mask": mask[int(idx[i]):int(idx[i + 1])],
             "codes": codes[int(idx[i]):int(idx[i + 1])]} for i in range(len(idx) - 1)]


def read_evidence(directory, split):
    docs = read_plain(directory, split)
    load = lambda suffix, dtype: np.fromfile(os.path.join(directory, f"{split}.{suffix}"), dtype=dtype)
    ev, evidx, evchunk = load("ev", np.uint16), load("evidx", np.uint64), load("evchunk", np.uint16)
    evkeyidx, evgold, cond, ans = load("evkeyidx", np.uint64), load("evgold", np.uint8), \
        load("cond", np.uint8), load("ans", np.uint8)
    keys = load("evkey", np.float16).reshape(-1, EMBED_DIM)
    for i, doc in enumerate(docs):
        ea, eb = int(evidx[i]), int(evidx[i + 1])
        ca, cb = int(evkeyidx[i]), int(evkeyidx[i + 1])
        doc["chunks"] = [ev[ea:eb][evchunk[ea:eb] == c].tolist() for c in range(cb - ca)]
        doc["gold"] = evgold[ca:cb].tolist()
        doc["keys"] = keys[ca:cb]
        doc["cond"], doc["ans"] = CONDITIONS[cond[i]], int(ans[i])
    return docs


def value_runs(doc):
    """``(attribute, token slice)`` for each maximal run of one value code."""
    runs, codes, start = [], doc["codes"], None
    for t in range(len(codes) + 1):
        code = int(codes[t]) if t < len(codes) else 0
        if start is not None and (code != int(codes[start])):
            runs.append((int(codes[start]), slice(start, t)))
            start = None
        if start is None and 1 <= code <= 5:
            start = t
    return runs


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def main():
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}")
        return
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    tiers = {10: 3, 1: 5}
    root = tempfile.mkdtemp(prefix="inject_")
    try:
        filler = write_filler(os.path.join(root, "filler"), 80, 0)
        out = os.path.join(root, "out")
        store_dir = os.path.join(root, "store")
        common = dict(persons_per_tier=tiers, filler=filler, target_tokens=100_000, seed=7,
                      val_fraction=0.1, store_dir=store_dir, swap_rate=0.5, anon_rate=0.9,
                      gold_drop_rate=0.5, distractors=3)
        metrics = build_injection(out, tokenizer, FakeEmbedder(), **common)
        people = bio.load_facts(os.path.join(out, "inject_facts.jsonl"))
        pools = bio.load_pools(os.path.join(out, "inject_pools.json"))
        by_name = {p.name: p for p in people}
        full = read_plain(out, "inject_full_train")
        masked = read_plain(out, "inject_masked_train")
        retrieval = read_evidence(out, "inject_retrieval_train")
        val = read_plain(out, "inject_val")
        assert len(full) == len(masked) == len(retrieval) and len(full) > 100

        # 1. full and masked
        for suffix in ("bin", "idx", "factspan"):
            a = file_hash(os.path.join(out, f"inject_full_train.{suffix}"))
            b = file_hash(os.path.join(out, f"inject_masked_train.{suffix}"))
            assert a == b, suffix
        n_bio = 0
        for f, m in zip(full, masked):
            assert f["ids"][0] == bos and f["ids"][-1] == eos and f["mask"][0] == 0
            assert f["mask"][1:].all()
            expected = np.ones(len(f["ids"]), dtype=np.uint8)
            expected[0] = 0
            is_value = (f["codes"] >= 1) & (f["codes"] <= 5)
            expected[is_value] = 0
            assert (m["mask"] == expected).all()
            n_bio += int(is_value.any())
        assert n_bio == 35, n_bio
        print("1. full: mask 1 after BOS; masked: 0 exactly on value tokens; shared bytes   PASS")

        # 2. exposures and lengths
        counts = {p.person_id: 0 for p in people}
        for f in full:
            if not ((f["codes"] >= 1) & (f["codes"] <= 5)).any():
                assert not f["codes"].any()
                continue
            text = tokenizer.decode(f["ids"][1:-1])
            owners = [p for p in people if p.name in text]
            assert len(owners) == 1, owners
            counts[owners[0].person_id] += 1
            # the value-code tokens decode to the person's own values
            for attribute_code, tokens in value_runs(f):
                shown = tokenizer.decode(f["ids"][tokens]).strip().rstrip(".")
                assert shown == owners[0].attributes[bio.ATTRIBUTES[attribute_code - 1]], shown
        assert all(counts[p.person_id] == p.tier for p in people), counts
        for split in ("inject_full_train", "inject_masked_train", "inject_retrieval_train", "inject_val"):
            n = os.path.getsize(os.path.join(out, f"{split}.bin")) // 2
            assert os.path.getsize(os.path.join(out, f"{split}.factspan")) == n
        assert metrics["common"]["exposures_by_tier"] == {
            "1": {"people": 5, "exposures_each": 1}, "10": {"people": 3, "exposures_each": 10}}
        print("2. every person rendered exactly tier times; factspan length matches bin   PASS")

        # 3. retrieval
        swapped = anonymized = goldless = gold_docs = collisions = 0
        for f, r in zip(full, retrieval):
            if not ((f["codes"] >= 1) & (f["codes"] <= 5)).any():
                assert r["ids"].tolist() == f["ids"].tolist() and r["mask"].tolist() == f["mask"].tolist()
                assert r["cond"] == "none" and r["chunks"] == [] and r["ans"] == 1
                continue
            assert r["ans"] == 1 and r["cond"] in ("mixed", "distractors")
            assert r["keys"].shape == (len(r["chunks"]), EMBED_DIM)
            texts = [tokenizer.decode(c) for c in r["chunks"]]
            has_gold = any(r["gold"])
            assert (r["cond"] == "mixed") == has_gold and sum(r["gold"]) == int(has_gold)
            assert len(texts) == 4          # three distractors and the gold, or four distractors
            text = tokenizer.decode(r["ids"][1:-1])
            has_placeholder = bool((r["codes"] == 7).any())
            assert has_placeholder == ("Person " in text)
            assert not (has_placeholder and has_gold), "a placeholder next to a gold card"
            assert r["mask"][0] == 0 and r["mask"][-1] == 1
            anonymized += int(has_placeholder)
            goldless += int(not has_gold)
            gold_docs += int(has_gold)
            for attribute_code, tokens in value_runs(r):
                shown = tokenizer.decode(r["ids"][tokens]).strip().rstrip(".")
                visible = any(shown in t for t in texts)
                assert set(r["mask"][tokens].tolist()) == {int(visible)}, (shown, visible)
                if not has_gold and visible:
                    collisions += 1
            # nothing else is masked: language and name tokens are all supervised
            masked_tokens = np.flatnonzero(r["mask"][1:] == 0) + 1
            assert all(1 <= r["codes"][t] <= 5 for t in masked_tokens)
            doc_swaps = 0
            if has_gold:
                gold_text = texts[r["gold"].index(1)]
                name = gold_text.split(". Born")[0]
                person = by_name[name]
                assert name in text and r["codes"].tolist().count(6) >= 1
                key_row = r["keys"][r["gold"].index(1)].astype(np.float32)
                canonical = FakeEmbedder().encode([person.store_chunk])[0]
                assert np.abs(key_row - canonical).max() < 2e-3, "the gold key is not the canonical one"
                for attribute_code, tokens in value_runs(r):
                    attribute = bio.ATTRIBUTES[attribute_code - 1]
                    shown = tokenizer.decode(r["ids"][tokens]).strip().rstrip(".")
                    assert shown in gold_text
                    if shown != person.attributes[attribute]:
                        swapped += 1
                        doc_swaps += 1
                        assert person.attributes[attribute] not in gold_text
                        assert person.attributes[attribute] not in text
                        assert shown in pools[attribute]
            # the arms differ only where a swap or a placeholder rewrote the document
            if doc_swaps == 0 and not has_placeholder:
                assert r["ids"].tolist() == f["ids"].tolist() and r["codes"].tolist() == f["codes"].tolist()
        assert swapped > 0 and anonymized > 0 and goldless > 0 and gold_docs > 0
        entry = metrics["arms"]["retrieval"]
        assert entry["swapped_spans"] == swapped and entry["anonymized_docs"] == anonymized
        assert entry["goldless_docs"] == goldless
        assert entry["evidence_to_prompt_ratio"] > 0 and 0 < entry["supported_fact_token_share"] <= 1
        print(f"3. retrieval: mask follows visibility, {swapped} swaps, {anonymized} placeholders, "
              f"gold key canonical   PASS")

        # 4. dataset readers
        from modules.data.evidence_dataset import EvidenceDataset
        from modules.data.sft_dataset import SFTDataset
        kwargs = dict(data_dir=out, tokenizer=tokenizer, batch_size=2, max_length=1024,
                      num_mtp_tokens=2, shuffle=False)
        supervised = {}
        for arm in ("full", "masked"):
            ds = SFTDataset(split=f"inject_{arm}_train", **kwargs)
            batches = list(iter(ds))
            assert batches
            supervised[arm] = sum(int((b["labels"] != -100).sum()) for b in batches)
        assert supervised["full"] == sum(int(d["mask"].sum()) for d in full)
        assert supervised["masked"] == sum(int(d["mask"].sum()) for d in masked)
        assert supervised["masked"] < supervised["full"]
        ds = EvidenceDataset(split="inject_retrieval_train", max_evidence_tokens=3072, **kwargs)
        batches = list(iter(ds))
        assert any("evidence_ids" in b and "chunk_gold" in b for b in batches) and ds.has_condition_labels
        assert sum(int((b["labels"] != -100).sum()) for b in batches) == sum(int(d["mask"].sum()) for d in retrieval)
        print(f"4. SFTDataset reads full/masked ({supervised['full']} vs {supervised['masked']} "
              f"labels), EvidenceDataset reads retrieval   PASS")

        # 5. validation
        assert val and all(not d["codes"].any() and d["mask"][0] == 0 and d["mask"][1:].all() for d in val)
        assert not os.path.exists(os.path.join(out, "inject_val.ev"))
        train_bodies = {d["ids"].tobytes() for d in full}
        assert not any(d["ids"].tobytes() in train_bodies for d in val)
        print(f"5. validation: {len(val)} filler windows, no evidence, disjoint from train   PASS")

        # 6. determinism, a sweep next to the first build, a short filler cycling
        out2 = os.path.join(root, "out2")
        build_injection(out2, tokenizer, FakeEmbedder(), **dict(common, store_dir=None))
        for name in ("inject_full_train.bin", "inject_retrieval_train.bin", "inject_retrieval_train.ev",
                     "inject_retrieval_train.factspan", "inject_val.bin", "inject_facts.jsonl"):
            assert file_hash(os.path.join(out, name)) == file_hash(os.path.join(out2, name)), name
        before = {n: file_hash(os.path.join(out, n)) for n in os.listdir(out)}
        sweep = build_injection(out, tokenizer, FakeEmbedder(),
                                **dict(common, arms=("retrieval",), suffix="s90a90", swap_rate=0.9))
        after = {n: file_hash(os.path.join(out, n)) for n in before if n != "inject_build.json"}
        assert all(before[n] == after[n] for n in after), "a sweep build changed an existing file"
        assert sweep["arms"]["retrieval"]["swapped_spans"] > entry["swapped_spans"]
        assert os.path.exists(os.path.join(out, f"{split_name('retrieval', 's90a90')}.bin"))
        with open(os.path.join(out, "inject_build.json")) as f:
            recorded = json.load(f)
        assert set(recorded["arms"]) == {"inject_full_train", "inject_masked_train",
                                        "inject_retrieval_train", "inject_retrieval_s90a90_train"}
        try:
            build_injection(out, tokenizer, FakeEmbedder(), **common)
            raise SystemExit("an existing split was overwritten without the flag")
        except FileExistsError:
            pass
        memmaps, train, val_windows = load_filler_windows(filler, 1000, 7, 0.1)
        chosen, repeat = pick_filler(train, int((train[:, 2] + 2).sum()) * 3, 7)
        assert 2.9 < repeat < 3.1 and len(chosen) >= 3 * len(train) - 1
        chosen, repeat = pick_filler(train, 20_000, 7)
        assert repeat == 1.0 and sum(int(c[2]) + 2 for c in chosen) >= 20_000
        assert metrics["common"]["filler_repeat_factor"] >= 1.0
        print(f"6. deterministic, sweep leaves existing files, overwrite refused, cycling   PASS")

        # 7. the biography store and token codes
        try:
            from modules.data.store import load_store
        except ImportError:
            print("7. SKIP: modules.data.store is not there yet")
        else:
            store = load_store(store_dir)
            assert len(store.chunks) == len(people) == len(store.keys)
            assert all(c["chunk_id"] == p.person_id and c["text"] == p.store_chunk and c["title"] == p.name
                       and c["source"] == "bios" and c["doc_id"] == str(p.person_id)
                       for c, p in zip(store.chunks, people))
            canonical = FakeEmbedder().encode([p.store_chunk for p in people])
            assert np.abs(store.keys.astype(np.float32) - canonical).max() < 2e-3
            assert store.meta["sources"] == {"bios": len(people)}
            print("7. biography store: one card per person, canonical keys                 PASS")
        text, spans = bio.render_bio(people[0], random.Random(0), name_override="Person ABC")
        ((ids, codes),) = tokenize_with_codes(tokenizer, [text], [spans], [7])
        assert len(ids) == len(codes) and set(codes.tolist()) == {0, 1, 2, 3, 4, 5, 7}
        assert "Person ABC" in tokenizer.decode(ids[codes == 7])
        print("all injection corpus checks passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
