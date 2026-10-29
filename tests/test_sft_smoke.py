"""``scripts/sft.py``'s ``sft()`` end to end on the micro shape, one process, a few log steps.

Runs the real trainer in miniature (the micro seed, the micro retrieval warm-up corpus, the real
loaders and DataLoader workers, accelerate, the fp32 masters, the validation pass, a save) and
stops it through the ``RunControl`` poll after a few log steps, so every code path the data
parallel work touched is executed once in the one process configuration, which must behave as it
always did:

1. the run starts, trains, validates, and stops with exit 10 on a checkpoint that carries the plan
   (``total_steps``, ``rows_per_epoch``, ``world_size``);
2. a relaunch resumes it (token count and step continue) and stops again on the grid;
3. a run directory whose checkpoints are gone but whose ``run_state.json`` is not exits 30.

Skips without CUDA, the micro seed or the micro corpus. Writes ``ckpts/evidence_smoke_*`` and
removes it. GPU required.
"""
import os
import shutil
import sys
from types import SimpleNamespace

os.environ["TINY_LLM_CONFIG"] = "config_micro.yaml"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_RESUME_FAILED, EXIT_USER_STOP, RunControl
import scripts.sft as sft_module
from utils import BASE_DIR

SEED = os.path.join(BASE_DIR, "ckpts", "inject", "seed_micro.pt")
DATA = os.path.join(BASE_DIR, "data", "prepared_inject")
SPLIT = "inject_retrieval_s100a50_train"
RUN = "smoke_mgpu_test"


class StopAfterPolls(RunControl):
    """a RunControl whose STOP file shows up at the n-th poll (the log cadence)."""
    polls_left = 3

    def poll(self):
        type(self).polls_left -= 1
        if type(self).polls_left <= 0 and not self.stop_requested:
            self._request_stop(EXIT_USER_STOP, "smoke test stop")


def args():
    return SimpleNamespace(
        repair=False, ir=False, evidence=True, reader_no_rotary=False, reader_kv=True,
        run_name=RUN, data_dir="data/prepared_inject", train_split=SPLIT, val_split="inject_val",
        switch_split=False, ceiling_json=None, checkpoint="ckpts/inject/seed_micro.pt",
        from_hub=False, hub_repo=None, hub_file="", upload_repo="",
    )


def launch(polls):
    StopAfterPolls.polls_left = polls
    sft_module.RunControl = StopAfterPolls
    return sft_module.sft(args())


def main():
    if not torch.cuda.is_available() or not os.path.isfile(SEED) or not os.path.isfile(
            os.path.join(DATA, f"{SPLIT}.ev")):
        print("sft smoke   SKIP (needs CUDA, the micro seed and the micro retrieval corpus)")
        return
    run_dir = f"{sft_module.EVIDENCE_CHECKPOINT_DIR}_{RUN}"
    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        code = launch(3)
        assert code == EXIT_USER_STOP, code
        state_path = os.path.join(run_dir, "run_state.json")
        state = ckpt_lib.read_run_state(state_path)
        assert state and state["phase"] == "evidence" and state["token_count"] > 0, state
        ckpt = os.path.join(run_dir, state["checkpoint"])
        payload = torch.load(ckpt, map_location="cpu")
        plan = {k: payload["sft"][k] for k in ("total_steps", "rows_per_epoch", "world_size")}
        assert plan["world_size"] == 1 and plan["total_steps"] > 1 and plan["rows_per_epoch"] > 1, plan
        first_tokens = state["token_count"]
        print(f"1. started, validated, stopped on a save with the plan {plan}   PASS")

        code = launch(3)
        assert code == EXIT_USER_STOP, code
        state = ckpt_lib.read_run_state(state_path)
        assert state["token_count"] > first_tokens, (state, first_tokens)
        print(f"2. resumed and stopped again ({first_tokens:,} -> {state['token_count']:,} tokens)   PASS")

        for name in os.listdir(run_dir):
            if name.startswith(ckpt_lib.CKPT_PREFIX) and name.endswith(".pt"):
                os.remove(os.path.join(run_dir, name))
        code = launch(3)
        assert code == EXIT_RESUME_FAILED, code
        print("3. checkpoints gone, run_state.json kept: exit 30   PASS")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    print("sft smoke passed")


if __name__ == "__main__":
    main()
