"""What ``scripts/sft.py`` carries across a stop and relaunch, on CPU with a tiny ``nn.Linear``.

1. **The save and eval grid is fixed.** ``next_grid_point`` gives the next multiple of the
   cadence, equal to ``tokens + every`` at a fresh start.
2. **A resume on other data is refused unless the switch is asked for.** A checkpoint without
   the data fields never refuses; a relative and an absolute spelling of one directory agree.
3. **A stop within one log interval of the last save does not save again**; a fresh run with no
   save always does.
4. **The fp32 masters round trip** through a save and a load in place: the optimizer keeps its
   tensor objects and its state, and a missing, misshapen or inconsistent entry reseeds every
   master from bf16 as before.
5. **The depth sampling RNG round trips** through a payload saved and loaded the way the trainer
   does it; a payload without it restores nothing.
6. **The payload fields survive ``save_sft_checkpoint`` / ``load_sft_checkpoint``**, and a
   payload written before them loads with empty fields.
7. **A finished run is recognised**: the final checkpoint is the newest and ``run_state.json``
   names it.
8. **The epoch loop selects the permutation through the prepared loader**: accelerate's loader
   sets its dataset's epoch from its own pass counter at the start of every pass, which starts
   at zero in a new process, so a resume past epoch 0 that set the epoch on the dataset alone
   trained on epoch 0's order again; ``select_epoch`` keeps the two in step and a fresh run's
   passes read epochs 0, 1, 2 as before.

9. **Every save and eval point is on the grid.** A simulated run of coarse log steps, taking each
   point at the first log step past it and scheduling the next with ``next_grid_point`` from the
   tokens it landed on, schedules only multiples of the cadence (the old ``tokens + every`` walked
   off the grid by each overshoot), and a SIGUSR1 save in between does not move it.
10. **The resume check names a lost run**: a ``run_state.json`` of this phase with no checkpoint,
    or one more than ``2 * checkpoint_every_tokens`` ahead of the checkpoint that loaded, fails;
    no state, another phase's state or a recent enough checkpoint passes.
11. **The plan the cosine was fitted to is recorded and compared**: ``total_steps``,
    ``rows_per_epoch`` and ``world_size`` survive the payload, differences are reported, and the
    per rank step count spans the same tokens at any number of ranks.

GPU free, but ``scripts/sft.py`` imports Transformer Engine, so run it in WSL after env_init.
"""
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset

from modules.runtime import checkpoints as ckpt_lib
from scripts.sft import (EVIDENCE_PHASE, check_split_switch, fp32_master_state, load_sft_checkpoint,
                         check_run_state, next_grid_point, plan_changed, plan_total_steps,
                         restore_fp32_masters, restore_rng_states, run_already_complete,
                         save_sft_checkpoint, select_epoch, split_changes, stop_needs_save)


def test_grid():
    assert next_grid_point(0, 25) == 25
    assert next_grid_point(0, 10_000_000) == 0 + 10_000_000
    assert next_grid_point(300_400_000, 25_000_000) == 325_000_000
    assert next_grid_point(325_000_000, 25_000_000) == 350_000_000
    assert next_grid_point(324_999_999, 25_000_000) == 325_000_000
    assert next_grid_point(17, 0) == 17
    print("1. save and eval grid fixed, fresh start equals tokens + every   PASS")


def test_split_switch():
    current = {"data_dir": "data/prepared_pilot", "train_split": "pilot_main_train",
               "val_split": "inject_val"}
    assert split_changes({}, current, "/repo") == []
    assert check_split_switch({}, current, False, "/repo") == []
    same = dict(current, data_dir="/repo/data/prepared_pilot/")
    assert split_changes(same, current, "/repo") == [], split_changes(same, current, "/repo")
    warm = dict(current, train_split="pilot_warm_train")
    try:
        check_split_switch(warm, current, False, "/repo")
    except SystemExit as e:
        assert "pilot_warm_train" in str(e) and "pilot_main_train" in str(e), str(e)
        assert "--switch-split" in str(e), str(e)
    else:
        raise AssertionError("a resume on another train split was not refused")
    changes = check_split_switch(warm, current, True, "/repo")
    assert changes == [("train_split", "pilot_warm_train", "pilot_main_train")], changes
    other_dir = dict(current, data_dir="data/prepared_inject")
    assert [c[0] for c in split_changes(other_dir, current, "/repo")] == ["data_dir"]
    print("2. other data refused without --switch-split, allowed with it, old payloads pass   PASS")


def test_stop_save():
    assert stop_needs_save(0, None, 5) is True
    assert stop_needs_save(100, 100, 5) is False
    assert stop_needs_save(105, 100, 5) is False
    assert stop_needs_save(106, 100, 5) is True
    print("3. a stop within one log interval of the last save exits on it   PASS")


def _linear_masters():
    torch.manual_seed(0)
    model = nn.Linear(8, 4).to(torch.bfloat16)
    pairs = []
    for name, param in model.named_parameters():
        master = param.detach().clone().float().requires_grad_(True)
        pairs.append((name, param, master))
    return model, pairs


def test_masters_roundtrip():
    model, pairs = _linear_masters()
    # sub-ulp progress: the masters move by less than half a bf16 ulp, so the bf16 weights do not
    with torch.no_grad():
        for _, param, master in pairs:
            master.add_(param.detach().float().abs().clamp_min(1e-3) * 1e-4)
            param.copy_(master.to(param.dtype))
    saved = fp32_master_state(pairs)
    assert all(v.device.type == "cpu" and v.dtype == torch.float32 for v in saved.values())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.pt")
        torch.save({"fp32_masters": saved}, path)
        loaded = torch.load(path, map_location="cpu")["fp32_masters"]

    # a relaunch: fresh masters cloned from the bf16 weights, an optimizer over them
    fresh = [(n, p, p.detach().clone().float().requires_grad_(True)) for n, p, _ in pairs]
    optimizer = optim.AdamW([m for _, _, m in fresh], lr=1e-3)
    ids_before = [id(m) for _, _, m in fresh]
    msg = restore_fp32_masters(fresh, loaded)
    assert msg.startswith("restored"), msg
    for (name, _, old), (_, _, new) in zip(pairs, fresh):
        assert torch.equal(old.detach(), new.detach()), f"{name}: master not restored exactly"
        assert not torch.equal(new.detach(), new.detach().to(torch.bfloat16).float()), (
            f"{name}: fixture carries no sub-ulp progress, so the check proves nothing")
    assert [id(m) for _, _, m in fresh] == ids_before
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == ids_before
    # one step still works on the restored masters
    for _, _, m in fresh:
        m.grad = torch.ones_like(m)
    optimizer.step()

    # every fallback reseeds from bf16, which is the old resume
    for bad, why in ((None, "none"),
                     ({k: v for k, v in loaded.items() if k != "bias"}, "missing"),
                     (dict(loaded, weight=torch.zeros(2, 2)), "shape"),
                     (dict(loaded, weight=loaded["weight"] + 1.0), "rounding")):
        again = [(n, p, p.detach().clone().float().add(5.0).requires_grad_(True)) for n, p, _ in pairs]
        msg = restore_fp32_masters(again, bad)
        assert msg.startswith("reseeded"), (why, msg)
        for _, param, master in again:
            assert torch.equal(master.detach(), param.detach().float()), why
    print("4. fp32 masters round trip in place; missing, misshapen or inconsistent ones reseed   PASS")


def test_rng_roundtrip():
    rng = random.Random(42)
    for _ in range(17):
        rng.random()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.pt")
        torch.save({"sft": {"loop_rng_state": rng.getstate()}}, path)
        sft_extra = torch.load(path, map_location="cpu")["sft"]
    expected = [rng.random() for _ in range(5)]
    fresh = random.Random(42)
    restored = restore_rng_states(fresh, sft_extra)
    assert "loop_rng" in restored, restored
    assert [fresh.random() for _ in range(5)] == expected
    untouched = random.Random(42)
    assert restore_rng_states(untouched, {}) == []
    assert untouched.random() == random.Random(42).random()
    print("5. depth sampling RNG round trips, an old payload restores nothing   PASS")


def test_payload_roundtrip():
    model, pairs = _linear_masters()
    optimizer = optim.AdamW([m for _, _, m in pairs], lr=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    fields = {"data_dir": "data/prepared_pilot", "train_split": "pilot_main_train",
              "val_split": "inject_val", "loop_rng_state": random.Random(1).getstate(),
              "cuda_rng_state": None, "total_steps": 30170, "rows_per_epoch": 241398,
              "world_size": 8, "num_docs": 12345, "plan_split": "pilot_main_train"}
    common = dict(epoch=0, step=3, token_count=10, start_token_count=0, global_offset=0,
                  losses=[], seed=0, phase=EVIDENCE_PHASE)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "checkpoint_evidence_test.pt")
        save_sft_checkpoint(model, optimizer, scheduler, path, sft_fields=fields,
                            fp32_masters=fp32_master_state(pairs), **common)
        state = load_sft_checkpoint(model, optimizer, scheduler, path, EVIDENCE_PHASE, tmp)
        assert state["data"] == {k: fields[k] for k in ("data_dir", "train_split", "val_split")}
        assert set(state["rng"]) == {"loop_rng_state"}, state["rng"]
        assert set(state["fp32_masters"]) == {"weight", "bias"}
        assert state["optimizer_steps"] == 3, state["optimizer_steps"]
        assert (state["total_steps"], state["rows_per_epoch"], state["world_size"]) == (
            30170, 241398, 8), state
        assert (state["num_docs"], state["plan_split"]) == (12345, "pilot_main_train"), state

        # a payload written before the fields: nothing recorded, nothing refused
        save_sft_checkpoint(model, optimizer, scheduler, path, **common)
        payload = torch.load(path, map_location="cpu")
        assert "fp32_masters" not in payload
        assert set(payload["sft"]) == {"start_token_count", "seed", "kill_checked"}, payload["sft"]
        state = load_sft_checkpoint(model, optimizer, scheduler, path, EVIDENCE_PHASE, tmp)
        assert state["data"] == {} and state["rng"] == {} and state["fp32_masters"] is None
        assert state["total_steps"] is None and state["rows_per_epoch"] is None
        assert state["world_size"] is None
        assert state["num_docs"] is None and state["plan_split"] is None
    print("6. payload fields survive save/load, an older payload loads with none   PASS")


def _touch(path, mtime):
    with open(path, "wb") as f:
        f.write(b"x")
    os.utime(path, (mtime, mtime))


def test_run_complete():
    phase = EVIDENCE_PHASE
    final = ckpt_lib.final_name(phase)
    rolling = ckpt_lib.rolling_name(phase, 300_000_000, 3.2)
    now = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "run_state.json")
        assert not run_already_complete(os.path.join(tmp, "absent"), phase)
        assert not run_already_complete(tmp, phase)
        _touch(os.path.join(tmp, rolling), now - 100)
        ckpt_lib.write_run_state(state_path, phase, 300, rolling)
        assert not run_already_complete(tmp, phase), "a run mid way read as complete"
        _touch(os.path.join(tmp, final), now)
        assert not run_already_complete(tmp, phase), "complete without run_state naming the final"
        ckpt_lib.write_run_state(state_path, phase, 900, final)
        assert run_already_complete(tmp, phase)
        assert not run_already_complete(tmp, "sft"), "another phase read as complete"
        # a rolling save newer than the final: the run went on, so it is not complete
        _touch(os.path.join(tmp, rolling), now + 100)
        assert not run_already_complete(tmp, phase)
    print("7. a finished run is recognised by its final checkpoint and run_state.json   PASS")


class _EpochDataset(IterableDataset):
    """Records the epoch each pass ran under, as the SFT datasets' permutation would."""

    def __init__(self):
        self.epoch = 0
        self.seen = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        self.seen.append(self.epoch)
        yield torch.zeros(1)


def test_epoch_select():
    dataset = _EpochDataset()
    loader = Accelerator(cpu=True).prepare(DataLoader(dataset, batch_size=None, num_workers=0))
    # a fresh run: the loop's epochs and the loader's pass counter agree
    for epoch in range(3):
        select_epoch(loader, dataset, epoch)
        list(loader)
    assert dataset.seen == [0, 1, 2], dataset.seen
    # a resume at epoch 1 in a new process: the loader's counter starts at zero again
    dataset = _EpochDataset()
    loader = Accelerator(cpu=True).prepare(DataLoader(dataset, batch_size=None, num_workers=0))
    for epoch in range(1, 3):
        select_epoch(loader, dataset, epoch)
        list(loader)
    assert dataset.seen == [1, 2], dataset.seen
    # a plain loader has no set_epoch of its own: the dataset's is used
    dataset = _EpochDataset()
    plain = DataLoader(dataset, batch_size=None, num_workers=0)
    select_epoch(plain, dataset, 2)
    list(plain)
    assert dataset.seen == [2], dataset.seen
    # a dataset without set_epoch (the LM reader) is left alone
    select_epoch(loader, object(), 1)
    print("8. the epoch permutation is selected through the prepared loader on a resume   PASS")


def test_grid_simulation():
    every = 25_000_000
    rng = random.Random(0)

    def run(rule):
        tokens, scheduled, taken = 0, [], []
        due = rule(tokens)
        for _ in range(400):
            tokens += rng.randint(100_000, 9_000_000)
            sigusr1 = rng.random() < 0.03
            if tokens >= due or sigusr1:
                taken.append(tokens)
                due = rule(tokens)
                scheduled.append(due)
        return scheduled, taken

    scheduled, taken = run(lambda t: next_grid_point(t, every))
    assert len(scheduled) > 20
    assert all(point % every == 0 for point in scheduled), scheduled[:6]
    assert all(b >= a for a, b in zip(scheduled, scheduled[1:]))
    rng = random.Random(0)
    drifted, _ = run(lambda t: t + every if t else every)
    assert not all(point % every == 0 for point in drifted), "the old rule stayed on the grid"
    assert next_grid_point(60_000_000, every) == 75_000_000    # landed just past 50M: next is 75M
    print("9. save and eval points stay on the grid across overshoots and extra saves   PASS")


def test_run_state_check():
    slack = 2 * 50_000_000
    state = {"phase": "evidence", "token_count": 300_000_000, "checkpoint": "c.pt"}
    assert check_run_state(None, "evidence", None, slack) is None
    assert check_run_state({"phase": "sft", "token_count": 9e9}, "evidence", None, slack) is None
    assert check_run_state({"phase": "evidence", "token_count": 0}, "evidence", None, slack) is None
    message = check_run_state(state, "evidence", None, slack)
    assert message and "300,000,000" in message and "no checkpoint loads" in message, message
    assert check_run_state(state, "evidence", 300_000_000, slack) is None
    assert check_run_state(state, "evidence", 200_000_000, slack) is None       # exactly the slack
    message = check_run_state(state, "evidence", 199_999_999, slack)
    assert message and "300,000,000" in message and "199,999,999" in message, message
    assert check_run_state(state, "evidence", 400_000_000, slack) is None       # ahead is fine
    print("10. a lost run directory or a checkpoint far behind run_state.json is refused   PASS")


def test_plan():
    one = plan_total_steps(241_398, 1, 8, 2, 1)
    assert one == max(1, int(241_398 * 1 / 8 / 2)), one      # the old expression, unchanged
    eight = plan_total_steps(241_398, 1, 8, 2, 8)
    # a step eats batch * accum * world rows, so the same epoch is a world-th of the steps
    assert abs(eight * 8 - one) <= 8, (one, eight)
    assert plan_total_steps(3, 1, 8, 2, 8) == 1
    saved = {"total_steps": 30170, "rows_per_epoch": 241398, "world_size": 1}
    same = {"total_steps": 30170, "rows_per_epoch": 241398, "world_size": 1}
    assert plan_changed(saved, same) == []
    assert plan_changed({}, same) == [], "an old payload records no plan and changes nothing"
    moved = plan_changed(saved, {"total_steps": 3771, "rows_per_epoch": 241398, "world_size": 8})
    assert moved == [("total_steps", 30170, 3771), ("world_size", 1, 8)], moved
    print("11. the plan is compared on resume and the cosine spans the same tokens per rank   PASS")


def main():
    test_grid()
    test_split_switch()
    test_stop_save()
    test_masters_roundtrip()
    test_rng_roundtrip()
    test_payload_roundtrip()
    test_run_complete()
    test_epoch_select()
    test_grid_simulation()
    test_run_state_check()
    test_plan()
    print("all sft resume state checks passed")


if __name__ == "__main__":
    main()
