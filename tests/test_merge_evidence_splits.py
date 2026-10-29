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
   tokenizer, skipped without it);
7. ``over_evidence_reader_row_limit`` counts exactly the documents whose prompt, plus BOS when
   missing, plus the MTP slots, exceed ``seq_length - 1``, per slice and in total;
8. ``md5_mask_by_slice`` is the md5 of each slice's mask bytes in output order and moves only for a
   slice whose mask changed, with ``md5_by_slice`` unchanged;
9. ``src`` is carried per document when every slice has it and written nowhere when one lacks it.

GPU free.
"""
import os, sys, json, shutil, hashlib, tempfile
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


def copy_split(src_dir, name, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f.startswith(name + "."):
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))


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
        stamp = {n: os.path.getmtime(os.path.join(out, n)) for n in os.listdir(out)}
        again = run(out, "m", target, 7, slices)
        assert again["md5_by_slice"] == result["md5_by_slice"]
        assert stamp == {n: os.path.getmtime(os.path.join(out, n)) for n in os.listdir(out)}
        try:
            run(out, "m", target, 8, slices)
            raise AssertionError("a second run with other arguments without --overwrite should refuse")
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
            assert tokenizer.bos_token_id == mes.build_parser().get_default("bos_id")
            print(f"6. EvidenceDataset ({len(batches)} batches) and SFTDataset ({len(plain_batches)}) read the merge   PASS")

        # 7. the evidence reader's row limit report
        seq, mtp, bos = 64, 2, 0
        lens = [61, 62, 61, 60, 63, 20, 59]
        first = [bos, bos, 7, 7, bos, 7, 9]
        long_dir = os.path.join(root, "long")
        os.makedirs(long_dir)
        ids = np.concatenate([np.r_[f, np.full(n - 1, 11)] for n, f in zip(lens, first)]).astype(np.uint16)
        ids.tofile(os.path.join(long_dir, "long.bin"))
        np.ones(len(ids), dtype=np.uint8).tofile(os.path.join(long_dir, "long.mask"))
        np.concatenate([[0], np.cumsum(lens)]).astype(np.uint64).tofile(os.path.join(long_dir, "long.idx"))
        over = [n + int(f != bos) + mtp > seq - 1 for n, f in zip(lens, first)]
        assert over == [False, True, True, False, True, False, False]
        res7 = run(os.path.join(root, "rowlimit"), "m", 10 ** 6, 7,
                   [f"long={long_dir}/long:0.5", f"qa={src}/qa:0.5"],
                   ["--seq-length", "4096", "--num-mtp-tokens", str(mtp), "--bos-id", str(bos)])
        assert res7["over_evidence_reader_row_limit"]["documents"] == 0
        res7 = run(os.path.join(root, "rowlimit"), "m", 10 ** 6, 7,
                   [f"long={long_dir}/long:0.5", f"qa={src}/qa:0.5"],
                   ["--seq-length", str(seq), "--num-mtp-tokens", str(mtp), "--bos-id", str(bos),
                    "--overwrite"])
        rl = res7["over_evidence_reader_row_limit"]
        qa_over = [int(n) + 1 + mtp > seq - 1 for n in mes.SliceSource(mes.parse_slice(f"qa={src}/qa:1")).lengths]
        assert rl["row_limit"] == seq - 1 and rl["by_slice"]["qa"]["documents"] == sum(qa_over)
        assert rl["by_slice"]["long"] == dict(documents=sum(over),
                                              tokens=sum(n for n, o in zip(lens, over) if o)), rl
        qa_lens = mes.SliceSource(mes.parse_slice(f"qa={src}/qa:1")).lengths
        assert rl["documents"] == 3 + sum(qa_over) and 0 < sum(qa_over) < len(qa_over)
        assert rl["tokens"] == 62 + 61 + 63 + int(qa_lens[np.asarray(qa_over)].sum())
        on_disk = json.load(open(os.path.join(root, "rowlimit", "m.merge.json")))
        assert on_disk["over_evidence_reader_row_limit"] == rl
        assert result["over_evidence_reader_row_limit"]["documents"] == 0
        print(f"7. evidence reader row limit: {rl['documents']} docs, {rl['tokens']} tokens over   PASS")

        # 8. the mask digest
        for si, (spec, _) in enumerate(sel):
            got = b"".join(d["mask"].tobytes() for d, k in zip(docs, slice_ids) if k == si)
            assert result["md5_mask_by_slice"][spec.label] == hashlib.md5(got).hexdigest(), spec.label
        masked_src = os.path.join(root, "src_mask")
        copy_split(src, "qa", masked_src)
        qa_mask = np.fromfile(os.path.join(masked_src, "qa.mask"), dtype=np.uint8)
        (1 - qa_mask).astype(np.uint8).tofile(os.path.join(masked_src, "qa.mask"))
        res8 = run(os.path.join(root, "maskdrift"), "m", target, 7,
                   [s.replace(f"{src}/qa", f"{masked_src}/qa") for s in slices])
        assert res8["md5_by_slice"] == result["md5_by_slice"]
        for label in ("bios", "plain", "nolab"):
            assert res8["md5_mask_by_slice"][label] == result["md5_mask_by_slice"][label], label
        assert res8["md5_mask_by_slice"]["qa"] != result["md5_mask_by_slice"]["qa"]
        print("8. md5_mask_by_slice catches a mask-only drift in its slice alone   PASS")

        # 9. src all or none
        labelled = os.path.join(root, "src_lab")
        source_bytes = {}
        for i, name in enumerate(("bios", "qa", "plain", "nolab")):
            copy_split(src, name, labelled)
            n_docs = len(np.fromfile(os.path.join(src, f"{name}.idx"), dtype=np.uint64)) - 1
            source_bytes[name] = ((np.arange(n_docs) + 3 * i) % 6).astype(np.uint8)
            source_bytes[name].tofile(os.path.join(labelled, f"{name}.src"))
        lab_slices = [s.replace(f"{src}/", f"{labelled}/") for s in slices]
        lab_out = os.path.join(root, "src_out")
        res9 = run(lab_out, "m", target, 7, lab_slices)
        assert res9["src_carried"] and not result["src_carried"]
        merged_src = load(lab_out, "m", "src", np.uint8)
        assert len(merged_src) == len(slice_ids)
        counters = [0] * len(sel)
        for n, s_id in enumerate(slice_ids):
            spec, picks = sel[s_id]
            assert merged_src[n] == source_bytes[spec.label][int(picks[counters[s_id]])]
            counters[s_id] += 1
        assert file_bytes(lab_out, "m", suffixes) == file_bytes(out, "m", suffixes)
        os.remove(os.path.join(labelled, "nolab.src"))
        res9b = run(lab_out, "m", target, 7, lab_slices, ["--overwrite"])
        assert not res9b["src_carried"] and not os.path.exists(os.path.join(lab_out, "m.src"))
        assert not os.path.exists(os.path.join(out, "m.src"))
        print("9. src carried per document when every slice has it, absent otherwise   PASS")

        # 10. --order-prefix-from
        ref_slice = load(out, "m", "slice", np.uint8)
        ref_path = os.path.join(out, "m.slice")
        pre_target = 1000
        pre_dir = os.path.join(root, "prefix")
        resp = run(pre_dir, "m", pre_target, 7, slices, ["--order-prefix-from", ref_path])
        pre_slice = load(pre_dir, "m", "slice", np.uint8)
        n_pre = len(pre_slice)
        assert n_pre == resp["num_docs"] == resp["documents"] == resp["order_prefix"]["documents"]
        assert np.array_equal(pre_slice, ref_slice[:n_pre]) and n_pre < len(ref_slice)
        pre_docs = read_merged(pre_dir, "m")
        pre_lens = [len(d["ids"]) for d in pre_docs]
        assert sum(pre_lens) >= pre_target and sum(pre_lens[:-1]) < pre_target
        assert resp["prompt_tokens"] == sum(pre_lens)
        seen = [0] * len(slices)
        prefixes = [mes.parse_slice(t).prefix for t in slices]
        for doc, s in zip(pre_docs, pre_slice):
            same_doc(doc, source_doc(prefixes[s], seen[s]))
            seen[s] += 1
        assert resp["order_prefix"]["documents_per_slice"] == {
            mes.parse_slice(t).label: seen[i] for i, t in enumerate(slices)}
        by_hand = os.path.join(root, "prefix_nb")
        run(by_hand, "m", pre_target, 7, slices, ["--order-prefix-from", ref_path, "--no-block-copy"])
        assert file_bytes(by_hand, "m", suffixes) == file_bytes(pre_dir, "m", suffixes)
        pre_plain = os.path.join(root, "prefix_plain")
        resq = run(pre_plain, "m", pre_target, 7, slices, ["--order-prefix-from", ref_path, "--no-evidence"])
        assert sorted(os.listdir(pre_plain)) == ["m.bin", "m.idx", "m.mask", "m.merge.json", "m.slice"]
        assert file_bytes(pre_plain, "m", ("bin", "idx", "mask", "slice")) == \
            file_bytes(pre_dir, "m", ("bin", "idx", "mask", "slice"))
        assert resq["md5_by_slice"] == resp["md5_by_slice"]
        # a shorter build that is a prefix by index gives the same bytes; one too short fails by name
        short_src = os.path.join(root, "short_src")
        os.makedirs(short_src)
        keep = 12
        bios_idx = np.fromfile(os.path.join(src, "bios.idx"), dtype=np.uint64)
        evidx = np.fromfile(os.path.join(src, "bios.evidx"), dtype=np.uint64)
        kidx = np.fromfile(os.path.join(src, "bios.evkeyidx"), dtype=np.uint64)
        cut = lambda suffix, dtype, n, width=1: np.fromfile(os.path.join(src, f"bios.{suffix}"), dtype=dtype) \
            .reshape(-1, width)[:n].tofile(os.path.join(short_src, f"bios.{suffix}"))
        cut("bin", np.uint16, int(bios_idx[keep])), cut("mask", np.uint8, int(bios_idx[keep]))
        cut("ev", np.uint16, int(evidx[keep])), cut("evchunk", np.uint16, int(evidx[keep]))
        cut("evkey", np.float16, int(kidx[keep]), EMBED_DIM), cut("evgold", np.uint8, int(kidx[keep]))
        cut("cond", np.uint8, keep), cut("ans", np.uint8, keep)
        bios_idx[:keep + 1].tofile(os.path.join(short_src, "bios.idx"))
        evidx[:keep + 1].tofile(os.path.join(short_src, "bios.evidx"))
        kidx[:keep + 1].tofile(os.path.join(short_src, "bios.evkeyidx"))
        short_slices = [slices[0].replace(f"{src}/bios", f"{short_src}/bios")] + slices[1:]
        need_bios = seen[0]
        if need_bios <= keep:
            res_short = run(os.path.join(root, "prefix_short"), "m", pre_target, 7, short_slices,
                            ["--order-prefix-from", ref_path])
            assert file_bytes(os.path.join(root, "prefix_short"), "m", suffixes) == \
                file_bytes(pre_dir, "m", suffixes)
            assert res_short["md5_by_slice"] == resp["md5_by_slice"]
        try:
            run(os.path.join(root, "prefix_fail"), "m", 10 ** 6, 7, short_slices, ["--order-prefix-from", ref_path])
        except ValueError as err:
            assert "bios" in str(err) or "order" in str(err), err
        else:
            raise AssertionError("an input or an order that runs out before the target should fail")
        try:
            run(os.path.join(root, "prefix_excl"), "m", pre_target, 7, slices,
                ["--order-prefix-from", ref_path, "--order-from", ref_path])
        except SystemExit:
            pass
        else:
            raise AssertionError("--order-prefix-from and --order-from are exclusive")
        print(f"10. --order-prefix-from: {n_pre} documents, slice file is the prefix of the recorded order, "
              f"block copy equals per document, --no-evidence works   PASS")

        # 11. resume across a stop
        def strip(result):
            result = json.loads(json.dumps(result))
            result["args"].pop("out_dir")
            return result

        def stopped_then_resumed(name, target_tokens, extra, stop_at, garbage=True):
            d = os.path.join(root, name)
            try:
                run(d, "m", target_tokens, 7, slices, extra + ["--stop-after-docs", str(stop_at)])
            except SystemExit as exit_:
                assert exit_.code == 20, exit_.code
            else:
                raise AssertionError("the stop hook should exit 20")
            state_file = os.path.join(d, "m.merge_state.json")
            assert os.path.exists(state_file) and not os.path.exists(os.path.join(d, "m.slice"))
            state = json.load(open(state_file))
            assert state["documents_done"] == stop_at
            if garbage:
                for s in state["offsets"]:
                    with open(os.path.join(d, f"m.{s}"), "ab") as f:
                        f.write(b"\xff" * 37)
            try:
                run(d, "m", target_tokens + 1, 7, slices, extra)
            except ValueError:
                pass
            else:
                raise AssertionError("a resume with other arguments should be refused")
            res = run(d, "m", target_tokens, 7, slices, extra)
            assert not os.path.exists(state_file)
            return d, res

        for label, extra, ref_dir, ref_res in (
                ("default", [], out, result),
                ("prefix", ["--order-prefix-from", ref_path], pre_dir, resp),
                ("prefix plain", ["--order-prefix-from", ref_path, "--no-evidence"], pre_plain, resq)):
            target_here = target if label == "default" else pre_target
            suff = suffixes if "plain" not in label else ("bin", "idx", "mask", "slice")
            n_here = ref_res["documents"]
            for stop_at in sorted({1, n_here // 3, n_here - 1}):
                d, res = stopped_then_resumed(f"resume_{label}_{stop_at}".replace(" ", "_"), target_here,
                                              extra + ["--checkpoint-docs", "4"], stop_at)
                assert file_bytes(d, "m", suff) == file_bytes(ref_dir, "m", suff), (label, stop_at)
                assert strip(res) == strip(ref_res), (label, stop_at)
                assert strip(json.load(open(os.path.join(d, "m.merge.json")))) == strip(ref_res)
        sigdir = os.path.join(root, "sigterm")
        sig_args = mes.build_parser().parse_args(
            ["--out-dir", sigdir, "--split", "m", "--target-tokens", str(target), "--seed", "7",
             "--checkpoint-docs", "4"] + sum((["--slice", s] for s in slices), []))
        mes.STOP_REQUESTED = True
        try:
            mes.merge(sig_args)
        except mes.MergeInterrupted:
            pass
        else:
            raise AssertionError("a stop request should save and stop")
        finally:
            mes.STOP_REQUESTED = False
        assert json.load(open(os.path.join(sigdir, "m.merge_state.json")))["documents_done"] == 4
        mes.merge(sig_args)
        assert file_bytes(sigdir, "m", suffixes) == file_bytes(out, "m", suffixes)
        fresh = run(sigdir, "m", target, 7, slices, ["--no-resume"])
        assert strip(fresh) == strip(result)
        print("11. interrupted merges resume byte identical (files and digests), a stale tail is truncated, other "
              "arguments are refused, a stop request saves state   PASS")
        # 12. a kill before the first state save, a finished merge, the order file fingerprint
        kill_dir = os.path.join(root, "kill0")
        real_save = mes.save_state_atomic

        def save_then_die(path, state):
            real_save(path, state)
            raise RuntimeError("killed")

        mes.save_state_atomic = save_then_die
        try:
            run(kill_dir, "m", target, 7, slices)
            raise AssertionError("the merge should have died")
        except RuntimeError:
            pass
        finally:
            mes.save_state_atomic = real_save
        assert json.load(open(os.path.join(kill_dir, "m.merge_state.json")))["documents_done"] == 0
        res12 = run(kill_dir, "m", target, 7, slices, ["--overwrite"])
        assert file_bytes(kill_dir, "m", suffixes) == file_bytes(out, "m", suffixes)
        assert strip(res12) == strip(result)
        assert not os.path.exists(os.path.join(kill_dir, "m.merge_state.json"))
        stamp = {n: os.path.getmtime(os.path.join(kill_dir, n)) for n in os.listdir(kill_dir)}
        assert run(kill_dir, "m", target, 7, slices)["md5_by_slice"] == result["md5_by_slice"]
        assert stamp == {n: os.path.getmtime(os.path.join(kill_dir, n)) for n in os.listdir(kill_dir)}
        # a state left behind by a kill between merge.json and its removal: complete, state dropped
        mid = os.path.join(root, "mid")
        try:
            run(mid, "m", target, 7, slices, ["--checkpoint-docs", "4", "--stop-after-docs", "4"])
        except SystemExit:
            pass
        old_state = open(os.path.join(mid, "m.merge_state.json")).read()
        run(mid, "m", target, 7, slices, ["--checkpoint-docs", "4"])
        open(os.path.join(mid, "m.merge_state.json"), "w").write(old_state)
        stamp = {n: os.path.getmtime(os.path.join(mid, n)) for n in os.listdir(mid) if "state" not in n}
        run(mid, "m", target, 7, slices, ["--overwrite"])
        assert not os.path.exists(os.path.join(mid, "m.merge_state.json"))
        assert stamp == {n: os.path.getmtime(os.path.join(mid, n)) for n in os.listdir(mid)}
        assert file_bytes(mid, "m", suffixes) == file_bytes(out, "m", suffixes)
        # a finished merge with --overwrite and no state starts clean; a damaged one is not trusted
        os.truncate(os.path.join(mid, "m.bin"), os.path.getsize(os.path.join(mid, "m.bin")) - 1)
        try:
            run(mid, "m", target, 7, slices)
            raise AssertionError("a damaged merge was taken as complete")
        except FileExistsError:
            pass
        run(mid, "m", target, 7, slices, ["--overwrite"])
        assert file_bytes(mid, "m", suffixes) == file_bytes(out, "m", suffixes)
        # the order file counts by content, not by path
        order_a = os.path.join(root, "order_a.slice")
        order_b = os.path.join(root, "order_b.slice")
        data = open(os.path.join(out, "m.slice"), "rb").read()
        open(order_a, "wb").write(data)
        open(order_b, "wb").write(data)
        digest = lambda path: mes.args_digest(mes.build_parser().parse_args(
            ["--out-dir", root, "--split", "m", "--target-tokens", str(target), "--order-prefix-from", path]
            + sum((["--slice", x] for x in slices), [])))
        assert digest(order_a) == digest(order_b)
        open(order_b, "wb").write(bytes([data[0] ^ 1]) + data[1:])
        assert digest(order_a) != digest(order_b)
        print("12. a kill before the first state save resumes under --overwrite, a finished merge is left alone, "
              "a leftover state is dropped, a damaged merge is refused, the order file counts by content   PASS")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
