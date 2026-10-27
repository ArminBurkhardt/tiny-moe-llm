"""Network-free checks for scripts/prepare_data.py's resumable interleave/write/checkpoint core
(PLAN.md Step 11). Synthetic in-memory sources stand in for real Hub shard files -- no HF Hub
calls, no GPU/TE dependency, runs on CPU like test_dataset_packing.py / test_dataset_resume.py.
"""
import os
import sys
import re
import random
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scripts.prepare_data import run_phase, truncate_to_state


class MockTokenizer:
    """char -> ord(char), trivially invertible so tests can decode bin content back to the
    original source string and verify exactly which documents were consumed."""
    def __call__(self, texts, add_special_tokens=False, truncation=False):
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def decode(ids):
    return "".join(chr(i) for i in ids)


def make_fake_generator_factory(docs):
    """splits docs into two "files" so file_idx/row_idx bookkeeping crosses a file boundary,
    same shape as the real HF-hub-backed factory in scripts/prepare_data.py."""
    mid = len(docs) // 2
    files = [docs[:mid], docs[mid:]]

    def factory(start_file_idx, start_row_idx):
        def gen():
            for file_idx in range(start_file_idx, len(files)):
                rows = files[file_idx]
                row_start = start_row_idx if file_idx == start_file_idx else 0
                for row_idx in range(row_start, len(rows)):
                    yield file_idx, row_idx, rows[row_idx]
        return gen()
    return factory


def make_sources(seed=0, n=200):
    rng = random.Random(seed)
    # "key#index|padding" -- key/index parseable back out after decoding bin content, padding
    # just varies document length like real text would
    return {
        k: [f"{k}#{i}|" + k * rng.randint(3, 8) for i in range(n)]
        for k in ("a", "b", "c")
    }


def source_entries_from(sources, weights):
    return [
        {"key": k, "weight": weights[k], "generator_factory": make_fake_generator_factory(v)}
        for k, v in sources.items()
    ]


def read_bin_idx(data_dir, phase):
    idx = np.fromfile(os.path.join(data_dir, f"{phase}.idx"), dtype=np.uint64)
    binv = np.fromfile(os.path.join(data_dir, f"{phase}.bin"), dtype=np.uint16)
    return idx, binv


def decode_all_docs(idx, binv):
    return [decode(binv[idx[i]:idx[i + 1]].tolist()) for i in range(len(idx) - 1)]


def per_source_indices(docs):
    out = {}
    for d in docs:
        m = re.match(r"^(\w+)#(\d+)\|", d)
        out.setdefault(m.group(1), []).append(int(m.group(2)))
    return out


def test_mix_ratio_and_bin_idx_consistency():
    d = tempfile.mkdtemp()
    sources = make_sources()
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    entries = source_entries_from(sources, weights)
    target = sum(len(t) for texts in sources.values() for t in texts) // 2  # well within availability
    state = run_phase("phase1", entries, target, MockTokenizer(), d, os.path.join(d, "_state.json"))

    idx, binv = read_bin_idx(d, "phase1")
    assert idx[0] == 0
    assert int(idx[-1]) == len(binv), "idx's last entry must equal len(bin)"
    assert np.all(np.diff(idx) >= 0), "idx must be monotonically non-decreasing"
    assert len(idx) - 1 == state["doc_count"]
    assert int(idx[-1]) == state["tokens_written"]

    realized = {k: state["sources"][k]["tokens"] for k in weights}
    total = sum(realized.values())
    for k, w in weights.items():
        got_frac = realized[k] / total
        assert abs(got_frac - w) < 0.05, f"{k}: realized share {got_frac:.3f} vs weight {w} -- SWRR should track weights closely"
    print("[ok] bin/idx internally consistent (idx monotone, last entry == len(bin)), per-source share tracks its weight")
    shutil.rmtree(d)


def test_interrupt_and_resume_no_gap_no_repeat():
    """crash-and-resume must never skip or double-consume a document from any single source,
    even though the global interleave order across the resume boundary isn't reproduced exactly
    (a fresh process re-derives SWRR state from scratch on restart -- see run_phase's docstring)."""
    d = tempfile.mkdtemp()
    sources = make_sources(seed=1)
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    total_available = sum(len(t) for texts in sources.values() for t in texts)
    target = total_available // 3
    state_path = os.path.join(d, "_state.json")

    # first leg: stop partway (checkpoint_docs small so the eventual truncation test below has
    # something to bite on), simulating an instance preemption
    entries_leg1 = source_entries_from(sources, weights)
    run_phase("phase1", entries_leg1, target // 2, MockTokenizer(), d, state_path, checkpoint_docs=5)

    # resume: a fresh process would rebuild generator_factory from scratch and reread state.json
    entries_leg2 = source_entries_from(sources, weights)
    run_phase("phase1", entries_leg2, target, MockTokenizer(), d, state_path, checkpoint_docs=5)

    idx, binv = read_bin_idx(d, "phase1")
    assert int(idx[-1]) == len(binv)
    assert np.all(np.diff(idx) >= 0)

    docs = decode_all_docs(idx, binv)
    consumed = per_source_indices(docs)
    for key, indices in consumed.items():
        assert sorted(indices) == list(range(len(indices))), (
            f"source {key}: consumed row indices must be a contiguous 0..N-1 prefix with no gap or "
            f"repeat across the interrupt boundary, got {sorted(indices)[:5]}...{sorted(indices)[-5:]}"
        )
    print(f"[ok] interrupt + resume: every source's consumed documents form a contiguous, "
          f"gap-free, repeat-free prefix ({sum(len(v) for v in consumed.values())} docs total)")
    shutil.rmtree(d)


def test_truncate_to_state_drops_uncommitted_tail():
    """simulate a crash that left bin/idx with bytes beyond the last saved checkpoint (buffered
    writes that made it to disk but were never fsynced+recorded) -- truncate_to_state must trim
    both files back down to exactly what state.json last confirmed."""
    d = tempfile.mkdtemp()
    bin_path, idx_path = os.path.join(d, "phase1.bin"), os.path.join(d, "phase1.idx")

    committed_tokens, committed_docs = 10, 3
    idx_committed = np.array([0, 3, 7, 10], dtype=np.uint64)  # 3 docs, cumulative lengths
    with open(bin_path, "wb") as f:
        f.write(np.arange(committed_tokens, dtype=np.uint16).tobytes())
        f.write(np.array([999, 999, 999], dtype=np.uint16).tobytes())  # uncommitted tail
    with open(idx_path, "wb") as f:
        f.write(idx_committed.tobytes())
        f.write(np.array([13], dtype=np.uint64).tobytes())  # uncommitted 4th entry

    state = {"tokens_written": committed_tokens, "doc_count": committed_docs}
    truncate_to_state(bin_path, idx_path, state)

    assert os.path.getsize(bin_path) == committed_tokens * 2
    assert os.path.getsize(idx_path) == (committed_docs + 1) * 8
    print("[ok] truncate_to_state drops bytes written after the last checkpoint")
    shutil.rmtree(d)


def test_holdout_hashes_recorded_for_designated_source():
    d = tempfile.mkdtemp()
    sources = make_sources(seed=2, n=50)
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    entries = source_entries_from(sources, weights)
    target = sum(len(t) for texts in sources.values() for t in texts) // 4
    state = run_phase("phase1", entries, target, MockTokenizer(), d, os.path.join(d, "_state.json"), holdout_source_key="a")

    idx, binv = read_bin_idx(d, "phase1")
    docs_from_a = sum(1 for doc in decode_all_docs(idx, binv) if doc.startswith("a#"))
    assert docs_from_a > 0, "sanity: source 'a' must have actually contributed documents"
    assert len(state["holdout_hashes"]) == docs_from_a, "one holdout hash per document committed from the designated source"
    print(f"[ok] holdout hashes recorded 1:1 with documents from the designated source ({len(state['holdout_hashes'])} entries)")
    shutil.rmtree(d)


def test_shard_seed_is_pure_across_processes():
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); "
            "from scripts.prepare_data import shard_shuffle_seed; "
            "print(shard_shuffle_seed(42, 'data/train-00001-of-00099.parquet'))"
            % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outs = []
    for hs in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip().splitlines()[-1])
    assert len(set(outs)) == 1, f"shard seed depends on the process hash salt: {outs}"
    from scripts.prepare_data import shard_shuffle_seed
    assert shard_shuffle_seed(42, "a") != shard_shuffle_seed(42, "b")
    assert shard_shuffle_seed(42, "a") != shard_shuffle_seed(43, "a")
    print(f"[ok] shard shuffle seed {outs[0]} identical across PYTHONHASHSEED 1, 2, random")


def test_manifest_key_required_for_ir():
    import argparse
    from scripts.prepare_data import check_manifest_key
    def parse(phases, key):
        parser = argparse.ArgumentParser()
        parser.add_argument("--phases", nargs="+", default=["phase1", "phase2"])
        parser.add_argument("--manifest-key", default=None)
        argv = ["--phases"] + phases + (["--manifest-key", key] if key else [])
        args = parser.parse_args(argv)
        check_manifest_key(parser, args)
        return args
    try:
        parse(["ir"], None)
        raise AssertionError("ir without --manifest-key must be refused")
    except SystemExit as e:
        assert e.code == 2
    try:
        parse(["phase1", "ir"], None)
        raise AssertionError("ir among phases without --manifest-key must be refused")
    except SystemExit:
        pass
    assert parse(["ir"], "ir_prep").manifest_key == "ir_prep"
    assert parse(["phase1", "phase2"], None).manifest_key == "data_prep"
    print("[ok] --manifest-key required with ir, default kept otherwise")


def test_finished_phase_with_pruned_bins_is_not_declared_done():
    """prune_filler deletes the bins; a --force rerun of the stage must fail loudly, not pass with empty files."""
    d = tempfile.mkdtemp()
    sources = make_sources(seed=3)
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    target = sum(len(t) for texts in sources.values() for t in texts) // 4
    state_path = os.path.join(d, "_state.json")
    run_phase("phase1", source_entries_from(sources, weights), target, MockTokenizer(), d, state_path)
    os.remove(os.path.join(d, "phase1.bin"))
    os.remove(os.path.join(d, "phase1.idx"))
    try:
        run_phase("phase1", source_entries_from(sources, weights), target, MockTokenizer(), d, state_path)
    except RuntimeError as e:
        assert "deleted or lost" in str(e), e
    else:
        raise AssertionError("a finished phase whose bins are gone must not pass with empty files")
    os.remove(state_path)
    run_phase("phase1", source_entries_from(sources, weights), target, MockTokenizer(), d, state_path)
    idx, binv = read_bin_idx(d, "phase1")
    assert int(idx[-1]) == len(binv) > 0, "removing the state rebuilds the phase"
    print("[ok] a finished phase with pruned bins fails instead of passing empty; no state rebuilds it")
    shutil.rmtree(d)


if __name__ == "__main__":
    test_finished_phase_with_pruned_bins_is_not_declared_done()
    test_mix_ratio_and_bin_idx_consistency()
    test_interrupt_and_resume_no_gap_no_repeat()
    test_truncate_to_state_drops_uncommitted_tail()
    test_holdout_hashes_recorded_for_designated_source()
    test_shard_seed_is_pure_across_processes()
    test_manifest_key_required_for_ir()
    print("\nPREPARE_DATA CHECKS PASSED")
