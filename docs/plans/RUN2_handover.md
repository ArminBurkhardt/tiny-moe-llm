# Run 2 on MareNostrum 5: handover

Written 2026-10-27. What you do, and how the cluster work goes. The details are in
[scripts/slurm/README.md](../../scripts/slurm/README.md) (the runbook), [RUN2_corpus.md](RUN2_corpus.md) (the corpus)
and [scripts/ops/SUPERVISOR.md](../../scripts/ops/SUPERVISOR.md) (the supervising session).

## The plan in five lines

- 800B tokens: 266.67B unique prompt tokens per split, 3 epochs, on 16 H100 (4 nodes), as a chain of 72 hour jobs.
- About 3,300 GPU hours at the expected rate (67k tokens per second per GPU), about 206 hours of wall time; the run fits
  4,500 hours as long as the rate stays above about 54k per GPU (about 59k if the corpus build's estimated 400 hours
  come out of the same budget).
- Order: corpus, the 2 GPU check, the 10B control, the warm-up (stop at 4B, read), the switch to the main split, the run.
- No node has internet. Everything goes in and out through the transfer nodes from your PC.
- Nobody watches the cluster live: a Claude Code session on your PC polls it every 30 to 60 minutes and pings your phone.

## What you have to do

**Before access (PC, WSL)**
1. Commit the tree (the one line command from the session).
2. Decide where the corpus is built: on the cluster (about 400 GPU hours of node time, an estimate; about 450 GB of
   filler sources pushed in, the listing in RUN2_corpus.md) or
   on the PC (needs about 3 TB of local disk; D: has about 125 GB free). The cluster is the default.
3. Build the container: `bash scripts/slurm/container/build.sh` (or `build.sh docker`). Check that the smoke lines pass.
4. Stage the tokenizer and bge: `python scripts/mirror_sources.py stage-small --hf-home DIR`.
5. Copy `scripts/ops/ops.env.example` to `scripts/ops/ops.env` and fill in user and paths.

**Day 1 (cluster)**
1. Open the two ssh master sockets in `tmux` in WSL (README, "Supervision"). Log in by hand once; 2FA is yours alone.
2. `bash scripts/ops/push_code.sh`, then copy the container and the staged models over.
3. On the cluster: copy `scripts/slurm/mn5_env.local.sh.example` to `mn5_env.local.sh` and fill in the account,
   `acc_ehpc`, the paths, `SIF` and the `module load singularity` line.
4. Run the day 1 checks in the README, in order: queues and quota, the 5 minute signal test, the container smoke, NCCL
   on 2 nodes.
5. Run one supervisor action by hand (`python scripts/ops/supervise.py --act` in WSL) and check that the job shows up in
   `squeue`. The tests could not cover the real ssh shell.

**Corpus (days 1 to about 5)**
1. Push the filler sources: `mirror_sources.py push` from the PC through the transfer node (RUN2_corpus.md).
2. Submit the stage groups in the README's order (`STAGES=... bash scripts/slurm/submit_chain.sh CORPUS 3`).
   Every long stage resumes after a kill.

**Training**
1. The 2 GPU against 1 GPU check: README Day 1 step 5. Touch STOP after about 100 steps; compare the loss lines.
2. The 10B control: `bash scripts/slurm/submit_chain.sh CONTROL 1` (about 2 hours).
3. The warm-up: `STOP_AFTER='4000M' bash scripts/slurm/submit_chain.sh TRAIN 4`.
4. Check the first hour's rate against 54k tokens per second per GPU (the supervisor alerts below it).
5. At the 4B stop: read the save, decide the switch. Then
   `SPLIT=run2_main_train bash scripts/slurm/submit_chain.sh TRAIN 4`.
6. Start the supervising session: a local Claude Code session in the repo, `/loop 45m` with SUPERVISOR.md as its
   standing instruction, Remote Control on for the phone. `/loop` expires after 7 days: re-arm it weekly.

**Your calls, when they come up**
- The switch at the warm-up stop (and whether to read again at 12B).
- Anything after a HALT, an exit 30, or a chain you cancelled: read the reason, then restart by hand.
- Pulling weights out: `bash scripts/ops/pull_exports.sh` (counted against the 2 TB ledger).
- Still open from the plan: the control's exposure (probe tiers or a denser build) and 6f's abstention reward.

## How the cluster work goes

- **Jobs.** Each training job runs up to 72 hours. Twenty minutes before its limit it saves and exits; the next job,
  already queued behind it, resumes from that save. Saves every 1B tokens; the newest two and the first one at or past
  each 10B are kept (about 370 GB).
- **On the cluster side.** A watchdog in every job kills and halts it on any of these: a stall (no step for 15 minutes), no
  save within the job's limit (49 minutes at 16 GPUs), a loss that turns NaN or climbs and stays up, a crash loop (3 bad jobs
  in the last 5, 2 early deaths, 2 jobs with no new save). Any setup error that would repeat (container, paths, config,
  missing row estimates) halts the chain on the first try, with the reason in `HALT`. Nothing is ever deleted on a halt.
- **On the PC side.** The supervisor reads one status JSON per poll. It resubmits a broken chain at most once an hour,
  pins the 10B saves, batches the evaluation reads and alerts you on everything else. It never cancels, never deletes,
  never switches the split and never restarts after a HALT. Its alerts include a slow rate, a missing save, a rising loss,
  the disk, and GPU hours projected over 4,500.
- **If your PC sleeps** the ssh socket dies, the supervisor reports "cannot reach" and the run goes on untouched. Reopen
  the socket.
- **Getting results out.** Weights are exported as bf16 (about 0.67 GB each) and pulled with `pull_exports.sh`. The full
  checkpoints stay on the project disk.
