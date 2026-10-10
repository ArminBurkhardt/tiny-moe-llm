"""The evidence split merger: re-indexing, shares, determinism, the plain form, the chunk cap.

Small synthetic splits (one plain, two with evidence, one of them with the gold/condition/answerable
labels) merged through ``scripts/merge_evidence_splits.py`` and checked for:

1. every output document equals its source document (found through the ``.slice`` sidecar and the
   tool's own selection function): prompt, mask, evidence tokens, chunk ids contiguous from 0, keys,
   gold flags, condition, answerable; offsets rebuilt cumulatively; plain documents carry no
   evidence, condition ``none`` and answerable 1;
2. realised shares track the targets, a short slice leaves a gap and a two pass slice draws a second
   permutation with no document repeated inside a pass;
3. same seed gives byte identical files; other bytes in one slice leave the other slices'
   ``md5_by_slice`` and subsequences unchanged; another seed changes the order; a second run
   without ``--overwrite`` refuses;
4. ``--no-evidence`` writes ``bin idx mask`` equal to the evidence form and nothing else but the
   sidecars;
5. ``--max-chunks-per-doc`` keeps every gold chunk, cuts distractors to the cap, renumbers chunks
   and trims tokens and keys to match;
6. ``EvidenceDataset`` iterates the merged split and ``SFTDataset`` the plain form (needs the pruned
   tokenizer, skipped without it).

GPU free.
"""
import os, sys, json, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts import merge_evidence_splits as mes
from scripts.prepare_evidence_data import CONDITIONS, EMBED_DIM
from utils import TOKENIZER_DIR


def write_split(directory, name, n_docs, seed, evidence, labels=True, max_chunks=6):
    rng = np.random.RandomState(seed)
    os.makedirs(directory, exist_ok=True)
    path = lambda s: os.path.join(directory, f"{name}.{s}")
    lens = rng.randint(20, 120, size=n_docs)
    np.concatenate([[0], np.cumsum(lens)]).astype(np.uint64).tofile(path("idx"))
    rng.randint(10, 60000, size=int(lens.sum())).astype(np.uint16).tofile(path("bin"))
    rng.randint(0, 2, size=int(lens.sum())).astype(np.uint8).tofile(path("mask"))
    if not evidence:
        return
    ev, evchunk, keys, gold, cond, ans = [], [], [], [], [], []
    evidx, evkeyidx = [0], [0]
    for _ in range(n_docs):
        k = int(rng.randint(0, max_chunks + 1))
        flags = (rng.rand(k) < 0.3).astype(np.uint8)
        if k and not flags.any():
            flags[rng.randint(k)] = 1
        for c in range(k):
            n = int(rng.randint(1, 7))
            ev.extend(rng.randint(10, 60000, size=n).tolist())
            evchunk.extend([c] * n)
        keys.append(rng.randn(k, EMBED_DIM).astype(np.float16))
        gold.extend(flags.tolist())
        cond.append(int(rng.randint(len(CONDITIONS))))
        ans.append(int(rng.randint(2)))
        evidx.append(len(ev))
        evkeyidx.append(len(gold))
    np.asarray(ev, dtype=np.uint16).tofile(path("ev"))
    np.asarray(evchunk, dtype=np.uint16).tofile(path("evchunk"))
    np.concatenate(keys).astype(np.float16).tofile(path("evkey"))
    np.asarray(evidx, dtype=np.uint64).tofile(path("evidx"))
    np.asarray(evkeyidx, dtype=np.uint64).tofile(path("evkeyidx"))
    if labels:
        np.asarray(gold, dtype=np.uint8).tofile(path("evgold"))
        np.asarray(cond, dtype=np.uint8).tofile(path("cond"))
        np.asarray(ans, dtype=np.uint8).tofile(path("ans"))


def load(directory, split, suffix, dtype):
    return np.fromfile(os.path.join(directory, f"{split}.{suffix}"), dtype=dtype)


def read_merged(directory, split):
    idx, evidx, evkeyidx = (load(directory, split, s, np.uint64) for s in ("idx", "evidx", "evkeyidx"))
    ids, mask = load(directory, split, "bin", np.uint16), load(directory, split, "mask", np.uint8)
    ev, evchunk = load(directory, split, "ev", np.uint16), load(directory, split, "evchunk", np.uint16)
    keys = load(directory, split, "evkey", np.float16).reshape(-1, EMBED_DIM)
    gold, cond, ans = (load(directory, split, s, np.uint8) for s in ("evgold", "cond", "ans"))
    docs = []
    for i in range(len(idx) - 1):
        a, b, ea, eb, ka, kb = (int(x) for x in (idx[i], idx[i + 1], evidx[i], evidx[i + 1],
                                                 evkeyidx[i], evkeyidx[i + 1]))
        docs.append(dict(ids=ids[a:b], mask=mask[a:b], ev=ev[ea:eb], evchunk=evchunk[ea:eb],
                         keys=keys[ka:kb], gold=gold[ka:kb], cond=int(cond[i]), ans=int(ans[i])))
    return docs


def source_doc(prefix, d):
    """Reads one source document without the tool's code path."""
    p = lambda s: f"{prefix}.{s}"
    idx = np.fromfile(p("idx"), dtype=np.uint64)
    a, b = int(idx[d]), int(idx[d + 1])
    doc = dict(ids=np.fromfile(p("bin"), dtype=np.uint16)[a:b], mask=np.fromfile(p("mask"), dtype=np.uint8)[a:b])
    if not os.path.isfile(p("ev")):
        return doc
    evidx, kidx = np.fromfile(p("evidx"), dtype=np.uint64), np.fromfile(p("evkeyidx"), dtype=np.uint64)
    ea, eb, ka, kb = int(evidx[d]), int(evidx[d + 1]), int(kidx[d]), int(kidx[d + 1])
    doc["ev"] = np.fromfile(p("ev"), dtype=np.uint16)[ea:eb]
    doc["evchunk"] = np.fromfile(p("evchunk"), dtype=np.uint16)[ea:eb]
    doc["keys"] = np.fromfile(p("evkey"), dtype=np.float16).reshape(-1, EMBED_DIM)[ka:kb]
    if os.path.isfile(p("evgold")):
        doc["gold"] = np.fromfile(p("evgold"), dtype=np.uint8)[ka:kb]
        doc["cond"] = int(np.fromfile(p("cond"), dtype=np.uint8)[d])
        doc["ans"] = int(np.fromfile(p("ans"), dtype=np.uint8)[d])
    else:
        doc["gold"] = np.zeros(kb - ka, dtype=np.uint8)
        doc["cond"], doc["ans"] = CONDITIONS.index("none"), 1
    return doc


def apply_cap(doc, cap):
    """Reference chunk cap with plain loops."""
    k = len(doc["gold"])
    if not cap or k <= cap:
        return doc
    n_gold = int(doc["gold"].sum())
    room = max(cap - n_gold, 0)
    keep, seen = [], 0
    for c in range(k):
        if doc["gold"][c]:
            keep.append(c)
        elif seen < room:
            keep.append(c)
            seen += 1
    new = {c: i for i, c in enumerate(keep)}
    tokens = [(t, new[int(c)]) for t, c in zip(doc["ev"], doc["evchunk"]) if int(c) in new]
    out = dict(doc)
    out["ev"] = np.asarray([t for t, _ in tokens], dtype=np.uint16)
    out["evchunk"] = np.asarray([c for _, c in tokens], dtype=np.uint16)
    out["keys"], out["gold"] = doc["keys"][keep], doc["gold"][keep]
    return out


def same_doc(got, want, evidence=True):
    assert np.array_equal(got["ids"], want["ids"]) and np.array_equal(got["mask"], want["mask"])
    if not evidence:
        return
    for key in ("ev", "evchunk", "keys", "gold"):
        assert np.array_equal(got[key], want.get(key, got[key][:0])), key
    assert got["cond"] == want.get("cond", CONDITIONS.index("none")) and got["ans"] == want.get("ans", 1)


def run(out_dir, split, target, seed, slices, extra=()):
    argv = ["--out-dir", out_dir, "--split", split, "--target-tokens", str(target), "--seed", str(seed)]
    for s in slices:
        argv += ["--slice", s]
    return mes.main(argv + list(extra))


def expected_selection(slices, target, seed):
    """Per slice (spec, selected documents), from the tool's selection function."""
    out = []
    for text in slices:
        spec = mes.parse_slice(text)
        src = mes.SliceSource(spec)
        out.append((spec, mes.select_documents(src.lengths, int(spec.share * target), spec.max_passes,
                                               seed, spec.label)))
    return out


def file_bytes(directory, split, suffixes):
    return {s: open(os.path.join(directory, f"{split}.{s}"), "rb").read() for s in suffixes}


def main():
    root = tempfile.mkdtemp(prefix="merge_")
    try:
        src = os.path.join(root, "src")
        write_split(src, "bios", 60, 1, True, max_chunks=6)
        write_split(src, "qa", 40, 2, True, max_chunks=9)
        write_split(src, "plain", 50, 3, False)
        write_split(src, "nolab", 20, 4, True, labels=False, max_chunks=3)
        write_split(os.path.join(root, "src2"), "bios", 60, 11, True, max_chunks=6)
        bt, qt, pt = (int(mes.SliceSource(mes.SliceSpec(n, os.path.join(src, n), 1)).lengths.sum())
                      for n in ("bios", "qa", "plain"))
        slices = [f"bios={src}/bios:0.4", f"qa={src}/qa:0.2:2", f"plain={src}/plain:0.2",
                  f"nolab={src}/nolab:0.05"]
        target = 3000
        out = os.path.join(root, "out")

        # 1. every merged document equals its source document
        result = run(out, "m", target, 7, slices)
        docs = read_merged(out, "m")
        slice_ids = load(out, "m", "slice", np.uint8)
        sel = expected_selection(slices, target, 7)
        assert len(docs) == len(slice_ids) == result["documents"]
        counters = [0] * len(sel)
        for doc, s in zip(docs, slice_ids):
            spec, picks = sel[s]
            d = int(picks[counters[s]])
            counters[s] += 1
            want = source_doc(spec.prefix, d)
            same_doc(doc, want)
            if "ev" in want:
                assert (doc["evchunk"] == np.repeat(np.arange(len(doc["gold"])),
                                                    np.bincount(doc["evchunk"], minlength=len(doc["gold"])))).all()
            else:
                assert len(doc["ev"]) == 0 and len(doc["gold"]) == 0
                assert doc["cond"] == CONDITIONS.index("none") and doc["ans"] == 1
        assert counters == [len(p) for _, p in sel]
        evidx = load(out, "m", "evidx", np.uint64)
        assert evidx[0] == 0 and evidx[-1] == len(load(out, "m", "ev", np.uint16))
        assert (np.diff(load(out, "m", "evkeyidx", np.uint64)) == [len(d["gold"]) for d in docs]).all()
        try:
            run(out, "m", target, 7, slices)
            raise AssertionError("second run without --overwrite should refuse")
        except FileExistsError:
            pass
        print(f"1. {len(docs)} merged documents equal their sources, offsets rebuilt, plain docs empty   PASS")

        # 2. shares, gap, passes
        big = 40000
        tight = [f"bios={src}/bios:0.3", f"plain={src}/plain:0.3"]
        res = run(out, "shares", big, 7, tight, ["--overwrite"])
        assert res["slices"][0]["prompt_tokens"] <= int(0.3 * big)
        res = run(out, "shares", 1000, 7, [f"bios={src}/bios:0.5", f"plain={src}/plain:0.5"], ["--overwrite"])
        for st in res["slices"]:
            assert abs(st["share_realised"] - 0.5) < 0.25 and st["prompt_tokens"] <= st["target_tokens"]
        res = run(out, "short", 10 ** 6, 7, [f"bios={src}/bios:0.1", f"qa={src}/qa:0.1"], ["--overwrite"])
        assert res["slices"][0]["prompt_tokens"] == bt and res["slices"][1]["prompt_tokens"] == qt
        assert res["gap_to_target"] == 10 ** 6 - bt - qt and res["slices"][0]["passes"] == 1.0
        (_, two), = expected_selection([f"qa={src}/qa:5:2"], 10 ** 6, 7)
        assert len(two) == 80 and len(set(two[:40])) == 40 and len(set(two[40:])) == 40
        assert not np.array_equal(two[:40], two[40:])
        res = run(out, "twopass", 10 ** 6, 7, [f"qa={src}/qa:5:2"], ["--overwrite"])
        assert res["slices"][0]["documents"] == 80 and res["slices"][0]["passes"] == 2.0
        res = run(out, "onepass", 10 ** 6, 7, [f"qa={src}/qa:5"], ["--overwrite"])
        assert res["slices"][0]["documents"] == 40
        print("2. shares track targets, a short slice leaves its gap, the pass cap and second permutation hold   PASS")

        # 3. determinism and slice independence
        again = os.path.join(root, "again")
        res2 = run(again, "m", target, 7, slices)
        suffixes = mes.EVIDENCE_SUFFIXES + ("slice",)
        assert file_bytes(out, "m", suffixes) == file_bytes(again, "m", suffixes)
        assert res2["md5_by_slice"] == result["md5_by_slice"]
        swapped = [s.replace(f"{src}/bios", os.path.join(root, "src2", "bios")) for s in slices]
        other = os.path.join(root, "other")
        res3 = run(other, "m", target, 7, swapped)
        for label in ("qa", "plain", "nolab"):
            assert res3["md5_by_slice"][label] == result["md5_by_slice"][label], label
        assert res3["md5_by_slice"]["bios"] != result["md5_by_slice"]["bios"]
        docs3, ids3 = read_merged(other, "m"), load(other, "m", "slice", np.uint8)
        for s in (1, 2, 3):
            mine = [d["ids"].tobytes() for d, k in zip(docs, slice_ids) if k == s]
            theirs = [d["ids"].tobytes() for d, k in zip(docs3, ids3) if k == s]
            assert mine == theirs
        res4 = run(os.path.join(root, "seed"), "m", target, 8, slices)
        assert res4["md5_by_slice"] != result["md5_by_slice"]
        assert not np.array_equal(load(os.path.join(root, "seed"), "m", "slice", np.uint8), slice_ids)
        # --order-from reuses an order and refuses one with other per slice counts
        reused = os.path.join(root, "reused")
        run(reused, "m", target, 7, swapped, ["--order-from", os.path.join(other, "m.slice")])
        assert file_bytes(reused, "m", suffixes) == file_bytes(other, "m", suffixes)
        counts = lambda r: [s["documents"] for s in r["slices"]]
        if counts(res3) != counts(result):
            try:
                run(os.path.join(root, "refused"), "m", target, 7, swapped,
                    ["--order-from", os.path.join(out, "m.slice")])
            except ValueError:
                pass
            else:
                raise AssertionError("--order-from accepted an order with other per slice counts")
        print("3. same seed byte identical, other slices unchanged by a different bios slice, seed changes order, "
              "--order-from reuses an order   PASS")

        # 4. the plain form
        plain_dir = os.path.join(root, "plainform")
        res5 = run(plain_dir, "m", target, 7, slices, ["--no-evidence"])
        assert file_bytes(plain_dir, "m", ("bin", "idx", "mask", "slice")) == \
            file_bytes(out, "m", ("bin", "idx", "mask", "slice"))
        assert sorted(os.listdir(plain_dir)) == ["m.bin", "m.idx", "m.mask", "m.merge.json", "m.slice"]
        assert res5["md5_by_slice"] == result["md5_by_slice"]
        print("4. --no-evidence equals the evidence form on bin idx mask and writes nothing else   PASS")

        # 5. the chunk cap
        cap = 2
        capped_dir = os.path.join(root, "capped")
        resc = run(capped_dir, "m", target, 7, slices, ["--max-chunks-per-doc", f"bios={cap},qa=1"])
        capped = read_merged(capped_dir, "m")
        assert np.array_equal(load(capped_dir, "m", "slice", np.uint8), slice_ids)
        caps = {0: cap, 1: 1}
        counters, trimmed = [0] * len(sel), 0
        for doc, s in zip(capped, slice_ids):
            spec, picks = sel[s]
            want = source_doc(spec.prefix, int(picks[counters[s]]))
            counters[s] += 1
            if "ev" in want:
                full = want
                want = apply_cap(want, caps.get(int(s), 0))
                trimmed += len(want["gold"]) != len(full["gold"])
                n_gold = int(full["gold"].sum())
                if caps.get(int(s)) and len(full["gold"]) > caps[int(s)]:
                    assert int(want["gold"].sum()) == n_gold
                    assert len(want["gold"]) == max(caps[int(s)], n_gold)
                assert (np.unique(doc["evchunk"]) == np.arange(len(doc["gold"]))[
                    np.isin(np.arange(len(doc["gold"])), doc["evchunk"])]).all()
            same_doc(doc, want)
        assert trimmed > 0
        assert resc["evidence_tokens"] < result["evidence_tokens"] and resc["prompt_tokens"] == result["prompt_tokens"]
        assert resc["md5_by_slice"]["plain"] == result["md5_by_slice"]["plain"]
        print(f"5. chunk cap keeps all gold, cuts distractors, renumbers ({trimmed} documents trimmed)   PASS")

        # 6. the dataset readers
        if not os.path.isdir(TOKENIZER_DIR):
            print(f"6. SKIP: no tokenizer at {TOKENIZER_DIR}")
        else:
            from transformers import AutoTokenizer
            from modules.data.evidence_dataset import EvidenceDataset
            from modules.data.sft_dataset import SFTDataset
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
            kwargs = dict(tokenizer=tokenizer, batch_size=2, max_length=1024, num_mtp_tokens=2, shuffle=False)
            batches = list(iter(EvidenceDataset(data_dir=out, split="m", max_evidence_tokens=3072, **kwargs)))
            assert any("evidence_ids" in b and "chunk_gold" in b for b in batches)
            plain_batches = list(iter(SFTDataset(data_dir=plain_dir, split="m", **kwargs)))
            assert batches and plain_batches
            print(f"6. EvidenceDataset ({len(batches)} batches) and SFTDataset ({len(plain_batches)}) read the merge   PASS")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
