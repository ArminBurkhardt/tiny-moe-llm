"""The data parallel logic of ``scripts/sft.py``, tested with fake ranks (no process group).

One GPU is all the dev box has, so what a real 8 GPU run does is covered here by faking
``process_index`` / ``num_processes`` and the collectives (each reduction is a function the trainer
receives, so a test hands it one that sums over fake ranks), and by the one process path staying
what it was.

1. **Dataset sharding.** The ranks' readers cover every conversation exactly once, disjointly, by
   the interleaved position rule; with DataLoader workers each (rank, worker) pair is its own
   stream; one rank is the unsharded reader byte for byte; the evidence reader pairs every row with
   its own evidence on every rank.
2. **The resume cursor** (the minimum stream position plus the stride, the ranks' states merged by
   MAX) skips no conversation, on any number of ranks and for ranks that consumed different amounts.
3. **Lockstep loaders.** Ranks with different batch counts all yield the same number, exactly the
   smallest, with no rank waiting on a collective the others skipped.
4. **The log step's reduction.** Counts sum, a scalar a rank does not have is left out of its mean,
   per loop CE averages, and any rank's SIGTERM / STOP file / SIGUSR1 request gives every rank the
   same exit code and the same checkpoint decision.
5. **The loss-free balancer moves the same bias on every rank** from the summed counts, and would
   not from the local ones.
6. **Eval is sharded by batch index and merged**: the two rank states merged equal the one process
   pass (GPU; tiny model).
7. **The startup checksum names a tensor that differs between ranks; an IR refresh is broadcast from
   rank 0; a resumed rank derives its CUDA seed from rank 0's state.**
8. **Accelerator arguments**: one process keeps the old ones, several get DDP with unused
   parameters allowed, no buffer broadcast and per rank loaders; non-zero ranks stay quiet.

GPU needed only for 6; the rest is CPU but ``scripts/sft.py`` imports Transformer Engine, so run it
in WSL after env_init.
"""
import os
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from accelerate import Accelerator
from modules.data.evidence_dataset import EvidenceDataset
from modules.data.sft_dataset import SFTDataset
from modules.model.moe import LoopMixtureOfExperts
from modules.runtime.control import EXIT_PREEMPTED, EXIT_USER_STOP
from scripts.sft import (
    COUNT_KEYS, LOG_SCALAR_KEYS, RankLog, accelerator_kwargs, check_ranks_identical,
    control_counts, dist_env, estimate_packed_rows, evaluate, ir_sync_tensors, lockstep_batches,
    merge_eval_states, pack_log_vector, plan_total_steps, reseed_cuda_for_rank, stop_exit_code,
    sync_ir_from_main, unpack_log_vector, worker_offset,
)
from scripts.pretrain import reduce_balance_counts

import test_eval_fixed_readings as fixture
import test_evidence_dataset as evidence_fixture
import test_sft_dataset as sft_fixture


def _corpus(tmp, n_docs, shuffle_seed_split="sft_train"):
    """n conversations of 9 tokens, each carrying a marker id 600 + i (not supervised)."""
    docs = []
    for i in range(n_docs):
        ids = [500, 600 + i] + [ord("a")] * 4 + [901, ord("z"), 501]
        docs.append((ids, [0] * (len(ids) - 2) + [1, 1]))
    sft_fixture.write_corpus(tmp, shuffle_seed_split, docs)


def _markers(batches):
    found = []
    for batch in batches:
        ids = batch["input_ids"].reshape(-1).tolist()
        found.extend(i - 600 for i in ids if 600 <= i < 700)
    return found


def _make(tmp, rank, world, **kw):
    kwargs = dict(batch_size=1, max_length=32, split="sft_train", num_mtp_tokens=2,
                  shuffle=False)
    kwargs.update(kw)
    return SFTDataset(tmp, sft_fixture.MockTokenizer(), rank=rank, world_size=world, **kwargs)


def test_sharding():
    tmp = tempfile.mkdtemp()
    try:
        n = 41
        _corpus(tmp, n)
        for shuffle in (False, True):
            for world in (1, 2, 3, 8):
                seen = []
                for rank in range(world):
                    dataset = _make(tmp, rank, world, shuffle=shuffle)
                    order = dataset.document_order(n)
                    got = sorted(_markers(list(dataset)))
                    want = sorted(int(order[p]) for p in range(n) if p % world == rank)
                    assert got == want, (shuffle, world, rank, got, want)
                    seen.extend(got)
                assert sorted(seen) == list(range(n)), (shuffle, world, "not a partition")
        # one rank is the unsharded reader, batch for batch
        plain = list(_make(tmp, 0, 1, shuffle=True))
        again = list(SFTDataset(tmp, sft_fixture.MockTokenizer(), batch_size=1, max_length=32,
                                split="sft_train", num_mtp_tokens=2, shuffle=True))
        assert len(plain) == len(again)
        for a, b in zip(plain, again):
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a), "world 1 changed a batch"
        # DataLoader workers: each (rank, worker) pair is one stream, position % (workers * world)
        world, workers = 2, 2
        seen, per_stream = [], {}
        for rank in range(world):
            dataset = _make(tmp, rank, world)
            loader = DataLoader(dataset, batch_size=None, num_workers=workers)
            for batch in loader:
                vid = int(batch["worker_id"][0])
                assert vid % world == rank, (vid, rank)
                per_stream.setdefault(vid, []).extend(_markers([batch]))
            seen.extend(m for v in per_stream if v % world == rank for m in per_stream[v])
        assert sorted(set(seen)) == list(range(n)) and len(seen) == len(set(seen)), "workers overlap"
        for vid, docs in per_stream.items():
            assert all(d % (workers * world) == vid for d in docs), (vid, docs)
        # a rank outside the world is refused
        try:
            _make(tmp, 2, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("rank 2 of 2 was accepted")
        # the evidence reader: every rank's rows carry their own conversations and evidence
        ev_dir = tempfile.mkdtemp(prefix="evidence_shard_")
        try:
            evidence_fixture.build_corpus(ev_dir)
            tok = SimpleNamespace(bos_token_id=evidence_fixture.BOS,
                                  pad_token_id=evidence_fixture.PAD,
                                  eos_token_id=evidence_fixture.PAD)
            docs_seen = []
            for rank in range(2):
                dataset = EvidenceDataset(ev_dir, tok, batch_size=2,
                                          max_length=evidence_fixture.MAX_LEN,
                                          split="evidence_train", num_mtp_tokens=1,
                                          shuffle=False, rank=rank, world_size=2)
                rows_docs, rows_ev = set(), set()
                for batch in dataset:
                    ids = batch["input_ids"].reshape(-1).tolist()
                    rows_docs.update(i for i in ids if i in (10, 20, 30, 40, 50))
                    if "evidence_ids" in batch:
                        rows_ev.update(i for i in batch["evidence_ids"].reshape(-1).tolist() if i > 1)
                want_docs = {10 * (p + 1) for p in range(5) if p % 2 == rank}
                assert rows_docs == want_docs, (rank, rows_docs, want_docs)
                own_ev = {t for p, d in enumerate(evidence_fixture.DOCS) if p % 2 == rank for t in d[1]}
                assert rows_ev <= own_ev, (rank, rows_ev, own_ev)
                docs_seen.extend(rows_docs)
            assert sorted(docs_seen) == [10, 20, 30, 40, 50]
        finally:
            shutil.rmtree(ev_dir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("1. ranks partition the conversations by position, one rank is the plain reader   PASS")


def test_resume_cursor():
    tmp = tempfile.mkdtemp()
    try:
        n = 60
        _corpus(tmp, n)
        for world, consumed in ((1, [3]), (2, [1, 4]), (3, [1, 3, 2]), (8, [1, 2, 3, 1, 2, 3, 1, 2])):
            stride = world
            states = [torch.full((stride,), -1, dtype=torch.long) for _ in range(world)]
            before = []
            for rank in range(world):
                iterator = iter(_make(tmp, rank, world))
                for _ in range(consumed[rank]):
                    batch = next(iterator)
                    states[rank][int(batch["worker_id"][0])] = int(batch["doc_idx"][0])
                    before.extend(_markers([batch]))
            merged = states[0].clone()
            for other in states[1:]:
                merged = torch.maximum(merged, other)       # the ranks' MAX merge
            start = worker_offset(merged.tolist(), stride, 0)
            after = []
            for rank in range(world):
                after.extend(_markers(list(_make(tmp, rank, world, start_doc_idx=start))))
            assert set(before) | set(after) == set(range(n)), (
                world, sorted(set(range(n)) - set(before) - set(after)))
            assert len(set(before) & set(after)) <= 4 * world, "resume repeats too much"
        # a state with nothing produced yet falls back
        assert worker_offset([-1, -1], 2, 17) == 17
        # a different number of ranks resumes from the same scalar: positions stay valid
        start = worker_offset([5, 7], 2, 0)
        assert start == 7
        wider = []
        for rank in range(4):
            wider.extend(_markers(list(_make(tmp, rank, 4, start_doc_idx=start))))
        assert set(wider) == set(range(start, n))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("2. the resume cursor skips nothing on any number of ranks   PASS")


def test_lockstep():
    counts = [25, 31, 27]
    world = len(counts)
    barrier = threading.Barrier(world)
    slots = [None] * world

    def make_reduce(rank):
        def reduce_min(n):
            slots[rank] = n
            barrier.wait()
            low = min(slots)
            barrier.wait()
            return low
        return reduce_min

    out = [None] * world

    def run(rank):
        batches = [(rank, i) for i in range(counts[rank])]
        out[rank] = list(lockstep_batches(batches, 10, make_reduce(rank)))

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "a rank hung in the lockstep reduction"
    assert [len(o) for o in out] == [25, 25, 25], [len(o) for o in out]
    for rank in range(world):
        assert out[rank] == [(rank, i) for i in range(25)], "batches reordered"
    # equal counts end together with nothing dropped
    out2 = [list(lockstep_batches(list(range(20)), 10, lambda n: n)) for _ in range(2)]
    assert out2[0] == list(range(20))
    # a rank with nothing ends the epoch for everyone at once
    assert list(lockstep_batches([], 10, lambda n: n)) == []
    print("3. loaders of unequal length yield the same number on every rank   PASS")


class _FakeControl:
    def __init__(self, stop=False, code=0, checkpoint=False):
        self.stop_requested, self.exit_code, self.reason = stop, code, "fake"
        self._checkpoint = checkpoint

    def take_checkpoint_request(self):
        requested, self._checkpoint = self._checkpoint, False
        return requested


def test_log_reduction():
    n_loops = 3
    nan = torch.tensor(float("nan"))

    def rank_vec(tokens, evidence, control, scalars, per_loop, include_user=True):
        counts = control_counts(control, include_user_stop=include_user)
        counts.update(tokens=tokens, evidence=evidence)
        return pack_log_vector(counts, scalars, [torch.tensor(v) for v in per_loop], "cpu")

    base = {"loss": torch.tensor(2.0), "ce": torch.tensor(1.5), "aux": torch.tensor(0.5),
            "p_max": torch.tensor(0.4), "top1": torch.tensor(0.3)}
    v0 = rank_vec(100, 10, _FakeControl(), dict(base, selection=torch.tensor(0.6),
                  contrastive=nan, grounded=torch.tensor(0.2)), [3.0, 2.0, 1.0])
    v1 = rank_vec(120, 0, _FakeControl(), dict(base, loss=torch.tensor(4.0), selection=nan,
                  contrastive=nan, grounded=None), [5.0, 4.0, 3.0])
    v2 = rank_vec(80, 30, _FakeControl(), dict(base, selection=torch.tensor(0.2),
                  contrastive=nan, grounded=torch.tensor(0.4)), [4.0, 3.0, 2.0])
    total = (v0 + v1 + v2)
    assert total.numel() == len(COUNT_KEYS) + 2 * len(LOG_SCALAR_KEYS) + n_loops
    out = unpack_log_vector(total.tolist(), 3)
    assert out["counts"]["tokens"] == 300 and out["counts"]["evidence"] == 40
    scalars = out["scalars"]
    assert abs(scalars["loss"] - (2.0 + 4.0 + 2.0) / 3) < 1e-12
    assert abs(scalars["selection"] - 0.4) < 1e-6, "a rank without a selection loss was counted"
    assert abs(scalars["grounded"] - 0.3) < 1e-6
    assert scalars["contrastive"] != scalars["contrastive"], "no rank had a value: NaN expected"
    assert out["per_loop"] == [4.0, 3.0, 2.0], out["per_loop"]

    # control: rank 0's STOP file, rank 1's SIGTERM, rank 2's SIGUSR1; the sum decides for everyone
    def counts_of(control, main):
        return control_counts(control, include_user_stop=main)

    quiet = [counts_of(_FakeControl(), r == 0) for r in range(3)]
    assert stop_exit_code({k: sum(c.get(k, 0) for c in quiet) for k in COUNT_KEYS}) is None
    user = counts_of(_FakeControl(True, EXIT_USER_STOP), True)
    preempt = counts_of(_FakeControl(True, EXIT_PREEMPTED), False)
    usr1 = counts_of(_FakeControl(checkpoint=True), False)
    summed = lambda cs: {k: sum(c.get(k, 0) for c in cs) for k in COUNT_KEYS}
    assert stop_exit_code(summed([user, quiet[1], quiet[2]])) == EXIT_USER_STOP
    assert stop_exit_code(summed([quiet[0], preempt, quiet[2]])) == EXIT_PREEMPTED
    assert stop_exit_code(summed([user, preempt, quiet[2]])) == EXIT_PREEMPTED, "SIGTERM must win"
    assert summed([quiet[0], quiet[1], usr1])["checkpoint"] == 1
    # a STOP file seen by a rank that is not rank 0 is ignored (only rank 0 polls it)
    assert stop_exit_code(summed([counts_of(_FakeControl(True, EXIT_USER_STOP), False)])) is None
    # a checkpoint request is consumed once
    control = _FakeControl(checkpoint=True)
    assert control_counts(control)["checkpoint"] == 1 and control_counts(control)["checkpoint"] == 0
    print("4. one reduction carries counts, means and every rank's stop and checkpoint requests   PASS")


def _balance_stub(n_loops=3, num_experts=6, first=2):
    bias = torch.zeros(n_loops, num_experts)
    counts = torch.zeros_like(bias)
    load = torch.zeros_like(bias)
    moe = SimpleNamespace(first_mlp_index=first, balance_bias=bias.clone(), _balance_stamp=None)
    state = (bias.clone(), counts, load)
    moe._balance_state = lambda device: state
    return moe


def test_balance_reduction():
    torch.manual_seed(0)
    local = [torch.randint(0, 50, (3, 6)).float() for _ in range(2)]
    total = local[0] + local[1]

    def step(counts_to_use, reduce_fn=None):
        moe = _balance_stub()
        moe._balance_state(None)[1].copy_(counts_to_use)
        if reduce_fn is not None:
            reduce_balance_counts(moe, reduce_fn)
        LoopMixtureOfExperts.update_balance_bias(moe, 0.1)
        return moe.balance_bias.clone(), moe._balance_state(None)[2].clone()

    single_bias, single_load = step(total)
    ranks = []
    for r in range(2):
        other = local[1 - r]
        ranks.append(step(local[r], reduce_fn=lambda t, o=other: t.add_(o)))
    for bias, load in ranks:
        assert torch.equal(bias, single_bias) and torch.equal(load, single_load), (
            "a rank's bias differs from the one process bias over the summed counts")
    assert torch.equal(ranks[0][0], ranks[1][0])
    unreduced = [step(local[r])[0] for r in range(2)]
    assert not torch.equal(unreduced[0], unreduced[1]), "the fixture does not separate local counts"
    print("5. the balance bias moves identically on every rank from the summed counts   PASS")


def _eval_batches():
    out = []
    for seed in range(4):
        batch = fixture._batch()
        torch.manual_seed(100 + seed)
        batch["input_ids"][:, 1:11] = torch.randint(1, fixture.P["vocab_size"], (2, 10))
        batch["source_ids"] = torch.where(batch["condition_ids"] >= 0, seed % 2, -1)
        out.append(batch)
    return out


def test_eval_sharding():
    if not torch.cuda.is_available():
        print("6. eval sharding   SKIP (no CUDA)")
        return
    from modules.model.transformer import TinyMoETransformer
    torch.manual_seed(0)
    model = TinyMoETransformer(**fixture.P).to("cuda").to(torch.bfloat16).eval()
    model.set_checkpointing(False, False)
    batches = _eval_batches()
    args = ("cuda", fixture.PAD, 4)
    single = evaluate(model, batches, *args, selector_by_loop=True)
    captured = []
    evaluate(model, batches, *args, selector_by_loop=True, rank=1, world_size=2,
             gather=lambda state: captured.append(state) or [state, state])
    merged = evaluate(model, batches, *args, selector_by_loop=True, rank=0, world_size=2,
                      gather=lambda state: [state, captured[0]])

    def close(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b), (path, sorted(a), sorted(b))
            for k in a:
                close(a[k], b[k], f"{path}/{k}")
        else:
            assert abs(a - b) <= 2e-5 * max(1.0, abs(a)), (path, a, b)

    close(single, merged)
    assert merged["batches"] == 4 and merged["tokens"] == single["tokens"]
    assert "per_source_condition_ce" in merged and "grounded_auroc" in merged
    # states merge by keys the ranks do not share
    assert merge_eval_states([captured[0], captured[0]])["n_batches"] == 2 * captured[0]["n_batches"]
    print("6. eval sharded by batch index and merged equals the one process pass   PASS")


def test_startup_and_refresh():
    rank0 = [("a", torch.arange(4.0)), ("b", torch.ones(3))]
    same = [("a", torch.arange(4.0)), ("b", torch.ones(3))]
    other = tensors_for_rank = [("a", torch.arange(4.0)), ("b", torch.ones(3) * 2)]
    from scripts.sft import tensors_checksum
    peer_same = tensors_checksum([t for _, t in same])
    check_ranks_identical(rank0, lambda t: torch.minimum(t, peer_same),
                          lambda t: torch.maximum(t, peer_same))
    peer_other = tensors_checksum([t for _, t in other])
    try:
        check_ranks_identical(rank0, lambda t: torch.minimum(t, peer_other),
                              lambda t: torch.maximum(t, peer_other))
    except RuntimeError as e:
        assert "['b']" in str(e), str(e)
    else:
        raise AssertionError("ranks with different weights passed the startup check")

    # IR refresh: rank 0's tables, masters, moments and buffers reach another rank
    def make_ir(seed):
        torch.manual_seed(seed)
        ir = nn.Module()
        ir.z_keys = nn.Parameter(torch.randn(8, 4))
        ir.y_values = nn.Parameter(torch.randn(8, 4))
        ir.register_buffer("centroids", torch.randn(2, 4))
        moe = SimpleNamespace(ir_modules=[ir])
        pairs = [(p, p.detach().clone().float().requires_grad_(True)) for p in ir.parameters()]
        opt = optim.AdamW([m for _, m in pairs], lr=1e-3)
        for _, m in pairs:
            m.grad = torch.randn_like(m)
        opt.step()
        return moe, opt, pairs

    moe0, opt0, pairs0 = make_ir(0)
    moe1, opt1, pairs1 = make_ir(1)
    tensors0 = ir_sync_tensors(moe0, opt0, pairs0)
    ptrs = {t.data_ptr() for t in tensors0}
    ir0 = moe0.ir_modules[0]
    assert ir0.z_keys.data_ptr() in ptrs and ir0.centroids.data_ptr() in ptrs
    assert all(m.data_ptr() in ptrs for _, m in pairs0), "a master is not synced"
    assert all(opt0.state[m]["exp_avg"].data_ptr() in ptrs for _, m in pairs0), "a moment is not synced"
    source = iter(ir_sync_tensors(moe0, opt0, pairs0))
    count = sync_ir_from_main(moe1, opt1, pairs1, lambda t: t.copy_(next(source)))
    assert count == len(tensors0)
    for a, b in zip(ir_sync_tensors(moe0, opt0, pairs0), ir_sync_tensors(moe1, opt1, pairs1)):
        assert torch.equal(a, b)

    state = torch.arange(16, dtype=torch.uint8)
    kept, seed = reseed_cuda_for_rank(state, 0)
    assert kept is state and seed is None
    seeds = {r: reseed_cuda_for_rank(state, r)[1] for r in (1, 2, 3)}
    assert len(set(seeds.values())) == 3 and all(isinstance(v, int) for v in seeds.values())
    assert seeds[1] == reseed_cuda_for_rank(state, 1)[1], "the derived seed is not deterministic"
    print("7. startup checksum, IR broadcast and the per rank CUDA seed   PASS")


def test_launch_arguments():
    one = accelerator_kwargs(1, 4)
    assert one == dict(device_placement=True, split_batches=True, gradient_accumulation_steps=4)
    many = accelerator_kwargs(8, 2)
    assert many["gradient_accumulation_steps"] == 2 and "split_batches" not in many
    config = many["dataloader_config"]
    assert config.split_batches is False and config.dispatch_batches is False
    ddp = [h for h in many["kwargs_handlers"] if type(h).__name__ == "DistributedDataParallelKwargs"]
    assert len(ddp) == 1 and ddp[0].find_unused_parameters is True and ddp[0].broadcast_buffers is False
    # a diverged loss on one rank must stay NaN in the mean, an absent selection term must not
    nanv = torch.tensor(float("nan"))
    vec = pack_log_vector({}, {"loss": nanv, "selection": nanv}, [], "cpu")
    got = unpack_log_vector(vec.tolist(), 1)["scalars"]
    assert got["loss"] != got["loss"] and got["selection"] != got["selection"]
    Accelerator(cpu=True, **many)         # the arguments are accepted

    logged = []
    base = SimpleNamespace(info=lambda m: logged.append(("info", m)),
                           warning=lambda m: logged.append(("warning", m)),
                           error=lambda m: logged.append(("error", m)))
    RankLog(base, 3).info("quiet")
    RankLog(base, 3).warning("loud")
    RankLog(base, 0).info("main")
    assert logged == [("warning", "[rank 3] loud"), ("info", "main")], logged
    os.environ.pop("RANK", None), os.environ.pop("WORLD_SIZE", None)
    assert dist_env() == (0, 1)
    os.environ["RANK"], os.environ["WORLD_SIZE"] = "5", "8"
    assert dist_env() == (5, 8)
    os.environ.pop("RANK"), os.environ.pop("WORLD_SIZE")

    # the estimator over the interleaved streams keeps the epoch's rows, the plan the tokens
    tmp = tempfile.mkdtemp()
    try:
        _corpus(tmp, 400)
        idx = os.path.join(tmp, "sft_train.idx")
        rows1 = estimate_packed_rows(idx, 32, 2, num_workers=4)
        rows8 = estimate_packed_rows(idx, 32, 2, num_workers=4 * 8)
        assert abs(rows8 - rows1) <= 4 * 8, (rows1, rows8)
        steps1 = plan_total_steps(rows1, 2, 4, 2, 1)
        steps8 = plan_total_steps(rows8, 2, 4, 2, 8)
        assert abs(steps8 * 8 - steps1) <= 8 + 8 * 4, (steps1, steps8)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("8. launch arguments, quiet ranks, per rank plan   PASS")


def test_grad_reduction_and_retention():
    from scripts.pretrain import all_reduce_grads
    from scripts.sft import select_retention_deletions
    torch.manual_seed(0)

    def rank_params(grads):
        ps = [nn.Parameter(torch.zeros(5, dtype=torch.bfloat16)), nn.Parameter(torch.zeros(3))]
        for p, g in zip(ps, grads):
            p.grad = None if g is None else g.clone()
        return ps

    g0 = [torch.randn(5).bfloat16(), None]                   # rank 0 has no gradient for the port
    g1 = [torch.randn(5).bfloat16(), torch.randn(3)]
    ps = rank_params(g0)
    other = [g1[0].float().reshape(-1), g1[1].reshape(-1)]
    state = {"i": 0}

    def fake_all_reduce(flat, bucket_numel=None):
        flat.add_(torch.cat(other))

    all_reduce_grads(ps, fake_all_reduce, 2)
    want0 = ((g0[0].float() + g1[0].float()) / 2).bfloat16()
    assert torch.equal(ps[0].grad, want0) and ps[0].grad.dtype == torch.bfloat16
    assert torch.allclose(ps[1].grad, g1[1] / 2), "an absent gradient must count as zeros"
    # small buckets give the same result
    ps2 = rank_params(g0)
    calls = []
    def fake_small(flat):
        at = sum(calls); calls.append(flat.numel()); flat.add_(torch.cat(other)[at:at + flat.numel()])
    all_reduce_grads(ps2, fake_small, 2, bucket_numel=4)
    assert len(calls) == 2 and torch.equal(ps2[0].grad, want0)

    names = [f"checkpoint_evidence_tok{m}M_loss1.0000.pt" for m in
             (50, 100, 150, 200, 250, 9990, 10003, 10050, 10100, 19999, 20001, 20050, 20100)]
    names.append("checkpoint_evidence_final.pt")
    doomed = [os.path.basename(p) for p in select_retention_deletions(names, 2, 10_000_000_000)]
    kept = [n for n in names if n not in doomed]
    assert "checkpoint_evidence_final.pt" in kept
    assert "checkpoint_evidence_tok10003M_loss1.0000.pt" in kept       # first past 10B
    assert "checkpoint_evidence_tok20001M_loss1.0000.pt" in kept       # first past 20B
    assert "checkpoint_evidence_tok20050M_loss1.0000.pt" in kept and "checkpoint_evidence_tok20100M_loss1.0000.pt" in kept
    assert "checkpoint_evidence_tok10050M_loss1.0000.pt" in doomed and "checkpoint_evidence_tok50M_loss1.0000.pt" in doomed
    assert select_retention_deletions(names, 2, 0) == []
    print("9. gradients average in fp32 buckets with absent ones as zeros; retention keeps milestones   PASS")


def main():
    test_sharding()
    test_resume_cursor()
    test_lockstep()
    test_log_reduction()
    test_balance_reduction()
    test_eval_sharding()
    test_startup_and_refresh()
    test_launch_arguments()
    test_grad_reduction_and_retention()
    print("all sft multi GPU logic checks passed")


if __name__ == "__main__":
    main()
