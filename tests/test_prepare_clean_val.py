"""The clean validation slice: windows, exclusion, byte layout and the overlap check.

Tiny synthetic filler corpora, CPU only. Checks that the recomputed windows equal
``load_filler_windows``'s validation set, that exclusion drops exactly the documents under the state
cursor, that the written files match ``PlainWriter``'s ``inject_val`` layout, and that the overlap
check reads 1.0 on text copied from the training rows and 0.0 on disjoint text.
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts import prepare_clean_val as pcv
from scripts.prepare_injection_data import load_filler_windows, MIN_WINDOW

BOS, EOS = 0, 1
WINDOW = 100


def write_corpus(directory, phase, docs):
    bin_path, idx_path = os.path.join(directory, f"{phase}.bin"), os.path.join(directory, f"{phase}.idx")
    np.concatenate(docs).astype(np.uint16).tofile(bin_path)
    np.concatenate(([0], np.cumsum([len(d) for d in docs]))).astype(np.uint64).tofile(idx_path)
    return bin_path, idx_path


def make_fixture(directory):
    rng = np.random.default_rng(0)
    ir = [rng.integers(10, 60000, size=int(rng.integers(150, 450))) for _ in range(60)]
    p1 = [rng.integers(10, 60000, size=int(rng.integers(150, 450))) for _ in range(30)]
    return [write_corpus(directory, "ir", ir), write_corpus(directory, "phase1", p1)], ["ir", "phase1"]


def write_plain(directory, name, docs, mask_value=1):
    np.concatenate(docs).astype(np.uint16).tofile(os.path.join(directory, f"{name}.bin"))
    np.concatenate([np.full(len(d), mask_value) for d in docs]).astype(np.uint8).tofile(
        os.path.join(directory, f"{name}.mask"))
    np.concatenate(([0], np.cumsum([len(d) for d in docs]))).astype(np.uint64).tofile(
        os.path.join(directory, f"{name}.idx"))


with tempfile.TemporaryDirectory() as tmp:
    filler, phases = make_fixture(tmp)
    seed, frac = 7, 0.3

    # 1. the recomputed windows are load_filler_windows's val set, in its order
    memmaps, _, val = load_filler_windows(filler, WINDOW, seed, frac)
    assert len(val) > 20, len(val)
    assert pcv.window_size(4096) == 1000 and pcv.window_size(1024) == 1000 and pcv.window_size(512) == 488

    # 2. exclusion drops exactly the documents below the cursor, nothing else
    state = os.path.join(tmp, "state.json")
    with open(state, "w") as f:
        json.dump({"sources": {"webtext": {"file_idx": 0, "row_idx": 25, "epoch": 0}}}, f)
    used = pcv.used_documents([state], "ir", phases)
    assert used == {"ir": 25, "phase1": 0}, used
    offsets = [np.asarray(np.memmap(i, dtype=np.uint64, mode="r")) for _, i in filler]
    kept, docs, stats = pcv.filter_windows(val, offsets, phases, used)
    ir_doc = np.searchsorted(offsets[0], val[val[:, 0] == 0][:, 1], side="right") - 1
    assert stats[0]["dropped"] == int((ir_doc < 25).sum()) and stats[0]["dropped"] > 0
    assert stats[1]["dropped"] == 0
    assert len(kept) == len(val) - stats[0]["dropped"]
    assert all(not (s == 0 and d < 25) for (s, _, _), d in zip(kept, docs))
    assert (kept[kept[:, 0] == 1] == val[val[:, 0] == 1]).all()
    with open(state, "w") as f:
        json.dump({"sources": {"webtext": {"row_idx": 3, "epoch": 1}}}, f)
    assert pcv.used_documents([state], "ir", phases)["ir"] is None
    with open(state, "w") as f:
        json.dump({"sources": {}}, f)
    try:
        pcv.used_documents([state], "ir", phases)
        raise AssertionError("a state with no webtext cursor must fail")
    except SystemExit:
        pass
    with open(state, "w") as f:
        json.dump({"sources": {"webtext": {"row_idx": 25}}}, f)

    # 3. the written files are inject_val's layout: BOS, body, EOS, mask 0 on BOS only
    out = os.path.join(tmp, "out")
    meta = pcv.build(out, filler, phases, WINDOW, seed, frac, [state], "ir", "clean", None, BOS, EOS)
    assert meta["windows"] == len(kept)
    b = np.fromfile(os.path.join(out, "clean.bin"), dtype=np.uint16)
    m = np.fromfile(os.path.join(out, "clean.mask"), dtype=np.uint8)
    ix = np.fromfile(os.path.join(out, "clean.idx"), dtype=np.uint64)
    fs = np.fromfile(os.path.join(out, "clean.factspan"), dtype=np.uint8)
    assert len(ix) == len(kept) + 1 and ix[0] == 0 and ix[-1] == len(b) == len(m) == len(fs)
    for i, (s, st, ln) in enumerate(kept):
        a, e = int(ix[i]), int(ix[i + 1])
        assert e - a == ln + 2 and b[a] == BOS and b[e - 1] == EOS
        assert (b[a + 1:e - 1] == memmaps[s][st:st + ln]).all()
        assert m[a] == 0 and (m[a + 1:e] == 1).all()
    assert not fs.any()
    meta_disk = json.load(open(os.path.join(out, "clean.meta.json")))
    assert meta_disk["sources"][0]["excluded_documents"] == [0, 25]

    # a target cuts the slice short
    short = pcv.build(out, filler, phases, WINDOW, seed, frac, [state], "ir", "short", 1000, BOS, EOS)
    assert 0 < short["tokens"] <= 1000 and short["windows"] < meta["windows"]

    # 4. the overlap check: 1.0 on text copied from training rows, 0.0 on disjoint text
    rng = np.random.default_rng(1)
    train_docs = [rng.integers(100, 30000, size=300) for _ in range(20)]
    write_plain(tmp, "train", train_docs)
    write_plain(tmp, "copied", [d[20:220] for d in train_docs[:5]])
    disjoint = [rng.integers(40000, 60000, size=300) for _ in range(5)]
    write_plain(tmp, "disjoint", disjoint)
    full = pcv.overlap_share(os.path.join(tmp, "copied"), os.path.join(tmp, "train"), rows=0)
    none = pcv.overlap_share(os.path.join(tmp, "disjoint"), os.path.join(tmp, "train"), rows=0)
    # stride 32 and length 64 tile a 200 token doc up to token 192 + 64 > 200: tokens past the last gram
    # are uncovered, so full coverage is the grams' reach
    reach = 5 * (((200 - pcv.GRAM) // pcv.STRIDE) * pcv.STRIDE + pcv.GRAM)
    assert full["covered_tokens"] == reach, full
    assert none["covered_tokens"] == 0 and none["covered_share"] == 0.0, none
    # unsupervised training tokens do not count
    write_plain(tmp, "train_masked", train_docs, mask_value=0)
    masked = pcv.overlap_share(os.path.join(tmp, "copied"), os.path.join(tmp, "train_masked"), rows=0)
    assert masked["covered_tokens"] == 0, masked
    # a slice whose length is a gram multiple reads exactly 1.0
    write_plain(tmp, "copied64", [d[:128] for d in train_docs[:5]])
    exact = pcv.overlap_share(os.path.join(tmp, "copied64"), os.path.join(tmp, "train"), rows=0)
    assert exact["covered_share"] == 1.0, exact
    print("overlap copied:", full["covered_share"], "exact:", exact["covered_share"],
          "disjoint:", none["covered_share"])

print("test_prepare_clean_val: all checks passed")
