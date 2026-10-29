# scripts/slurm

Operator's runbook for the real run on MareNostrum 5 (accelerated partition: 4 x H100 64 GB, 80 cores, 512 GB RAM and
a 480 GB node disk at `$TMPDIR` per node; 72 hour job limit; 4,500 GPU hours, 4 TB project disk, 2 TB transfer). Token
target 800B: 266.67B prompt tokens per split, 3 epochs, 524,288 tokens per update. Written from research, never run
under real Slurm: every "unconfirmed" below is a day 1 check.

**No node has outgoing internet.** The container, the code, the tokenizer, the bge model, the datasets and the corpus
are pushed in from outside, checkpoints are pulled out from outside. Inbound ssh works (`alogin1.bsc.es`,
`alogin2.bsc.es`; transfers through `transfer1..4.bsc.es`). Jobs export `HF_HUB_OFFLINE=1` unless `ONLINE=1`.

| file | what |
|---|---|
| `mn5_env.sh` | tracked, no site value: sources `mn5_env.local.sh`, defaults, offline flags, `BIND` and `CTR` rebuilt per shell |
| `mn5_env.local.sh.example` | copy to `mn5_env.local.sh` on the cluster (gitignored, never pushed): paths, account, qos, modules, `SIF` |
| `container/` | `run2.def` (Apptainer), `Dockerfile` (same image), `build.sh` (build off cluster, size, sha256, smoke), `smoke.py` |
| `train_run2.sbatch`, `train_run2_control.sbatch` | the run and the 10B control; both source `train_job.sh` |
| `train_job.sh` | one job: precheck, the srun line, the watchdog, the exit decision, the chain state |
| `watchdog.sh` | sidecar of the training job: stalls, divergence (no GPU) |
| `submit_chain.sh` | `TRAIN\|CONTROL\|CORPUS N`: N jobs chained with `afterany` from the login node |
| `cancel_chain.sh` | `[RUN]`: the only supported way to end a chain by hand (marks it cancelled, then scancels its jobs) |
| `rank_exec.sh` | one training process per GPU: rank variables, SIGTERM forwarding, its exit code to a file |
| `corpus.sbatch` | the corpus driver for given stages, with a stall watchdog |
| `reads.sbatch` | up to 4 `(CKPT, TAG)` pairs, one per GPU of one node, optional weight export |
| `status.sh`, `status.py` | one compact JSON about the run (login node, no GPU, seconds) |
| `lib.sh` | pure functions, tested in `tests/test_run2_launchers.py` and `tests/test_slurm_chain.py` |
| `../ops/` | PC side: `push_code.sh`, `push_corpus.sh`, `pull_exports.sh`, `supervise.py`, `SUPERVISOR.md`, `ops.env.example` |

Everything in `scripts/ops/` runs on the user's PC **inside WSL** (Windows OpenSSH has no ControlMaster; rsync is a Linux
tool). Credentials never leave the PC: accounts are personal, so no cloud session gets them.

## What you fill in

On the cluster, once: `cp scripts/slurm/mn5_env.local.sh.example scripts/slurm/mn5_env.local.sh` and edit it: `REPO`,
`DATA_ROOT`, `SBATCH_ACCOUNT`, `SBATCH_QOS` (`acc_ehpc`, the only QOS of an ACC only EuroHPC allocation, and the default;
`acc_debug` cannot hold a 71:50 job, such a job pends for ever), `SIF`, the `module load singularity` line,
`SLURM_CPU_BIND=none` if binding fails. That file is gitignored and `push_code.sh` never sends it, so a push does not
overwrite it; the tracked `mn5_env.sh` holds no site value. A job or a submit with a `CHANGE` placeholder left exits 2 and
says which. `BIND` and `CTR` are rebuilt in every shell (the node's `$TMPDIR` goes into the bind list only when it exists on
that node, so a login node path carried by `--export=ALL` is dropped) and are never taken from the login shell.
`TINY_LLM_TOKENIZER` defaults to `$HF_HOME/tokenizer`, where `mirror_sources.py stage-small` puts the tokenizer; singularity
passes the host environment into the container and `$HF_HOME` is bound. A training or reads job without a tokenizer there
stops before the trainer (the training chain halts with the reason). A `SIF` with no `singularity` or `apptainer` on PATH exits 2 (no host python by accident).
`scripts/ops/ops.env` (copy `ops.env.example`, on the PC): user, project paths, the two socket paths,
`TRANSFER_BUDGET_GB`, optionally `HF_EXPORT_REPO`. Submit with `submit_chain.sh`, which sources the environment and
creates `logs` itself.

## Container

Python runs inside the image when `SIF` is set (`singularity exec --nv --bind $REPO,$DATA_ROOT,$TMPDIR,$HF_HOME`, or
`apptainer`, whichever is on PATH; no extra InfiniBand binds); empty `SIF` runs the plain venv. The repo is bind
mounted, never baked in. Base image `nvcr.io/nvidia/pytorch:25.06-py3` (CUDA 12.9 like the local torch 2.8.0+cu129,
inside what the host driver supports; ships torch, transformer-engine and flash-attn, which `build.sh` never
reinstalls). The local WSL stack is torch 2.8.0, TE 2.15.0, flash-attn 2.8.3, transformers 5.10.2, accelerate 1.13.0;
the image's TE is older: `smoke.py` (imports `modules.model.transformer` and `scripts.pretrain`) decides, and a failure
there means trying `BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3` (docker mode) before any code change.

## Day 0, off the cluster (the PC, in WSL)

1. Image: `bash scripts/slurm/container/build.sh` (apptainer) or `build.sh docker` (writes `run2.tar`; on the cluster
   `module load singularity; singularity build run2.sif docker-archive://run2.tar` in an interactive job). Check the printed
   size and sha256 and the smoke lines, then `rsync -P run2.sif USER@transfer1.bsc.es:/gpfs/projects/PROJECT/run2.sif`.
2. Models and tokenizer: `python scripts/mirror_sources.py stage-small --hf-home DIR`, copy `DIR` to `$HF_HOME`.
3. Code: `bash scripts/ops/push_code.sh` (`head` and `bundle` variants). 4. Corpus: build locally and
   `bash scripts/ops/push_corpus.sh LOCAL_DATA_ROOT`, or feed the filler with `mirror_sources.py push` and build on the
   cluster (`docs/plans/RUN2_corpus.md`). Pushes are resumable and counted in `ops_transfer_ledger.tsv`; one that would
   cross `TRANSFER_BUDGET_GB` is refused.

## Day 1, on the cluster

1. `bsc_queues`, `bsc_project list`, `bsc_quota`; write the account and QOS into `mn5_env.sh`.
2. A 5 minute `acc_debug` job (if the QOS exists): `sbatch --qos=acc_debug --time=00:05:00 --signal=B:USR1@120` of a script
   with `trap 'echo USR1 received' USR1; scontrol show config | grep -i killwait; sleep 240`. Confirms the `--signal`
   trap works and prints KillWait. The training job does not rely on it: it also sets its own save timer from `squeue`
   (20 minutes before the limit).
3. Image smoke inside a job: `srun ... singularity exec --nv --bind $REPO run2.sif python scripts/slurm/container/smoke.py`.
4. NCCL on 2 nodes: `NODES=2` run of the 2 GPU check below at `--nodes=2 --ntasks-per-node=1`, or `all_reduce` of a
   tensor under `srun`; `NCCL_DEBUG=WARN` (set `NCCL_SOCKET_IFNAME` if it picks the wrong interface). Alternative launch
   form if per GPU srun tasks misbehave: one task per node running `torchrun` (`rank_exec.sh` would be replaced).
5. The 2 GPU against 1 GPU check. Needs the seed `ckpts/inject/seed_run2_grounded.pt` and the corpus **rows stage** done
   (the rows cache holds entries for world sizes 1, 2, 4, 8 and 16, `ROWS_WORLD_SIZES` in `build_run2_corpus.py`; without
   the entry the trainer exits 2, which now halts the chain with "rows cache missing"). Exactly:
   `NODES=1 GPUS_PER_NODE=2 RUN=run2_chk2 bash scripts/slurm/submit_chain.sh TRAIN 1`, wait for step lines in
   `ckpts/inject/run2_chk2.log` (about 100 steps, 52M tokens, 7 minutes at 2 GPUs), then `touch ckpts/evidence_run2_chk2/STOP`
   (polled every 10 steps; the trainer saves and exits 10, the chain ends as `stop`, no HALT). Then the same with
   `GPUS_PER_NODE=1 RUN=run2_chk1` (STOP after the same number of steps). Compare the loss lines and the per rank counts.
   The watchdog cannot kill this: its save limit is derived from the GPU count (`SAVE_MIN` 386 minutes at 2 GPUs, plus 30
   for the first save; a 1B save takes about 2 hours there) and the step limit (30 minutes between step lines, 30 at startup)
   is far above the seconds a step takes. Do not shorten `SAVE_MIN` by hand for this check.
5a. Memory at the worst case, before the warm-up: on one node at batch 8 run `sft.py --probe-peak` (one synthetic worst case
   evidence batch; it prints the peak in GB per rank) through the same container and `TINY_LLM_CONFIG=config_run2.yaml`, and
   compare with the 64 GB of an H100. If the peak is above about 58 GB, lower `max_evidence_tokens` in `config_run2.yaml` (and
   rebuild nothing: the cap only closes rows earlier) before any warm-up job; an out of memory in hour 20 costs a job.
5b. One real planned stop: start a short job (the 2 GPU check's line is enough), wait for step lines, then `scancel
   --signal=TERM JOBID.0` on its step by hand. Confirm every rank logs a save and exits 20, `logs/rc_JOBID.rank0` holds 20, and
   the chain decides `resubmit`. This is the path the 20 minute time limit stop and every interruption take.
5c. Run `python scripts/ops/supervise.py` on the PC without `--act` until the first chain has started and its first poll
   reads sensibly (one action seen in `squeue` is the Day 1 test of `--act`, after that); then turn `--act` on.
6. The 10B control: `bash scripts/slurm/submit_chain.sh CONTROL 1` (about 4 hours at 8 GPUs).
7. Corpus stages if built on the cluster (each job takes a whole node: exclusive, all memory, 80 CPUs): `STAGES=... bash
   scripts/slurm/submit_chain.sh CORPUS 3` per stage group, in this order: filler (only through the mirror, "The filler through the mirror" below: no login node has internet), qa, bios_main (with the
   chains in parallel), val_clean, bios_full, bios_warm, tail_text, prune_filler, qa_shuffle (after the chains), order, merges
   (main, warm, control), prune_bios, merge_tail, rows (CPU only, after merge_tail), prune_sources, the held-out stages, stores.
   The filler stage also builds the tail phase's plain text (its fourth command, fresh files at the higher quality mix), so
   the mirror pushes about 490 GB instead of 444 GB; `tail_text` windows it, `merge_tail` writes `run2_tail_train` (the last 5%
   of the run, RUN2_corpus.md section 2b). `prune_bios` must run before `merge_tail` (the disk peak), `prune_sources` after it.
8. Warm-up: `STOP_AFTER='4000M' bash scripts/slurm/submit_chain.sh TRAIN 4` (the watcher drops `STOP` after the first save
   at or past 4B: saves land a log step past their grid point, `tok4003M`, so the rule is numeric, never an exact tag; the
   run passes `--plan-split run2_main_train` and, in the warm-up, `--max-epochs-this-split 1`, and the log line "warm split
   exhausted" ends the chain like a stop and asks for the switch decision).
   Read each save: `RUN=run2 PAIRS="ckptA:2B ckptB:4B" sbatch --job-name=run2_reads_run2 scripts/slurm/reads.sbatch`.
   Each read set ends with the 13-task benchmark suite (`BENCH=1`, the default; `BENCH=0` skips it): language quality is read at every save, offline from `$HF_HOME/datasets` (staged by `stage-small`), output `ckpts/inject/run2_TAG_bench.json`. A failed suite fails the job but never the supervisor's read counting (`reads_done` keys on `rank_none`).
9. The switch, once the criterion holds: `SPLIT=run2_main_train bash scripts/slurm/submit_chain.sh TRAIN 4` (a fresh
   chain: clears `STOP` and `HALT`, adds `--switch-split`). On the main split `STOP_AFTER` defaults to `760000M`: the chain
   stops by itself after the first save at or past 760B (the start of the tail phase) and waits for the tail switch below;
   `STOP_AFTER=''` (set and empty) turns that stop off, any other tag moves it. The stop is recorded in the chain state
   (`stop_after`, `split`), so a chain the supervisor resubmits (`START_INDEX=1`) keeps it.
10. **The tail switch** (the last 5%: 40B tokens from 760B, on `run2_tail_train`). At the 760B stop the chain ends with
   `STOP` kept and the supervisor alerts "tail switch due" (a decision for you, not a failure; it never resubmits past it).
   To go on: `rm ckpts/evidence_run2/STOP`, then `SPLIT=run2_tail_train bash scripts/slurm/submit_chain.sh TRAIN 3`. The fresh
   chain adds `--switch-split --switch-from-start --end-at-plan` and no stop: the trainer reads the tail from its first
   document in the main split's last epoch, the optimizer, the one cosine (step count), the token count and the RNG carry on,
   and the run ends on its own at the plan's last step (the tail holds a 5% surplus), with a final save and exit 0. Refused with
   exit 30 (the chain halts with the reason) unless the checkpoint is in the last epoch and was planned on `run2_main_train`,
   or when the tail holds fewer optimizer steps than the plan has left (a switch far too early). The tail's row estimate
   (`rows` stage, 8 ranks) must exist, as the main one does. A tail run that is stopped and resumed later is a plain resume.

Day 1 checks that the jobs depend on (none was ever run under real Slurm):

- **The save path.** The time limit signal (and the watchdog's stall verdict, and a cancel) stops the trainer with
  `scancel --signal=TERM JOBID.0` on the step, never a signal to `srun` (srun treats SIGTERM as a forced termination and may
  SIGKILL the step: no save, no rc file, exit 255). Check that the signal reaches every task: in the 2 GPU check `touch`
  nothing, wait for step lines, then `scancel --signal=TERM JOBID.0` by hand: each rank must log its save and exit 20, rank 0's
  code must be in `logs/rc_JOBID.rank0`, and a `singularity exec` wrapper must forward the TERM (`rank_exec.sh` forwards it
  to the container process). `squeue -s -j JOBID` must list the step as `JOBID.0`.
- **GPU visibility.** `srun` passes `--gres=gpu:N --gpu-bind=none` (the same gres style as the `#SBATCH` lines; a step mixing
  gres and `--gpus-per-node` is rejected on some Slurm versions): every task sees the node's GPUs and `LOCAL_RANK` picks one. Print `CUDA_VISIBLE_DEVICES` per task once.
- **flock on the project disk.** A job takes `ckpts/evidence_RUN/.trainer.lock` (and a corpus job
  `$DATA_ROOT/slurm_state/<stages>.lock`) with `flock`; a file system without it makes every job exit 2 at once.
- **The transfer socket**, below.

## The filler through the mirror

No node, login nodes included, reaches the Hub, so the filler stage reads its shards from an inbox that the PC fills through
the transfer nodes (`scripts/mirror_sources.py`, demand driven: the PC sends a file only while the build keeps asking).
1. PC, once: `python scripts/mirror_sources.py list --out listing.json` (the driver's plan flags as for the build) and
   `python scripts/mirror_sources.py stage-small --hf-home DIR`, copy `DIR` to `$HF_HOME` on the cluster (step 2 of Day 0).
   Copy `listing.json` to `$DATA_ROOT/listing.json`.
2. Cluster, the filler job: `STAGES=filler CORPUS_ARGS="--mirror-inbox $DATA_ROOT/inbox --mirror-listing
   $DATA_ROOT/listing.json --offline-stage $HF_HOME" bash scripts/slurm/submit_chain.sh CORPUS 3`.
3. PC, at the same time, inside the transfer master socket's reach: `python scripts/mirror_sources.py push --listing
   listing.json --remote USER@transfer1.bsc.es:$DATA_ROOT/inbox --window-gb 120 --transfer-limit-gb 2000`. It ends when the
   driver drops its done mark into the inbox. If the push stops, the job's log ends in `mirror: waiting Ns` lines and the job
   halts after `MIRROR_WAIT_MAX` minutes with its own reason; restart the push (it resumes) and resubmit the stage.
The corpus job's stall watchdog judges by `$DATA_ROOT/run2_logs/last_child_output`, which the driver touches whenever a child
process prints a line or a stage starts or ends (its 300 s heartbeat does not), `STALL_MIN` (120) minutes old means a hung
child; before the file exists the log and file times are used.

## Chains and what protects the budget

`submit_chain.sh KIND N` queues N jobs with `--dependency=afterany` from the login node (default `CHAIN_MODE=presubmitted`:
no job calls sbatch; `CHAIN_MODE=self` restores the old in-job resubmission). Each job runs the precheck first (final
checkpoint, STOP, HALT, ended, job cap, two jobs without a new save) and a surplus job exits at once: about 10 to 30
seconds of the allocation, 0.02 to 0.07 GPU hours at 8 GPUs. 800B is about 6 to 8 jobs of 71:50 at 8 GPUs; submit 4 at a time and resubmit.

Two trainers on one run directory is the failure the chain is built against: every job of a chain carries the chain id
(`submit_chain.sh` creates it, the first job writes it into `chain_state`), a job of another chain exits at once, a job takes
`flock` on the run directory before it touches any state, `submit_chain.sh` refuses while `squeue -n NAME` lists a job of that
name or when squeue fails, and `supervise.py` exits 2 without acting when squeue failed. Job names are explicit and matched
exactly: the run name (`run2`, `run2_control`, `RUN=...`), `run2_corpus_<stages>`, `run2_reads_<run>`. **End a chain by hand
only with `bash scripts/slurm/cancel_chain.sh [RUN]`** (or drop `STOP`). It writes `ended=cancelled` and a history line into the
run's `chain_state` first, then scancels every job of that name, pending and running. A bare `scancel` of a job is not read as
a cancel: a TERM reaching a job (time limit, node failure, admin kill, requeue, a hand scancel) is an interruption, the job
writes an `interrupted` history line, asks the ranks to save, and the next afterany job of the chain goes on. A chain
marked cancelled makes the running job save and the afterany successors exit at once, and `supervise.py` never
resubmits it: it alerts, and a person who wants it restarted clears it on the login node:
`sed -i '/^ended=cancelled$/d' ckpts/evidence_RUN/chain_state && sed -i '$ s/^\([0-9]*\) cancelled /\1 ok /' ckpts/evidence_RUN/chain_history`.
A failure every rerun would repeat (the environment: no container runtime, no tokenizer, a bad `STOP_AFTER`, unfilled site
values, a lock that cannot be taken; the trainer's exit 3, a deliberate refusal, or exit 2, a missing rows cache or a bad
argument) writes `HALT` with the reason, and the supervisor reports it. The trainer's exit 1 is an uncaught exception, a crash
that may be transient (a file system error, an Xid, a dead peer, an out of memory): the chain resubmits, and halts after two
jobs in a row without a new save (or by the crash loop rule below). A chain that ends writes
`ended=...` into `chain_state` (done, stop, halted, max-jobs, stalled, warm-exhausted) and, for any end that is not a done run
or a STOP, a `HALT` with the reason; `supervise.py` never resubmits a chain that names an end. The trainer's own kill rule
(exit 10 with no STOP file) halts the chain with the `KILL:` line, exit 30 halts with the trainer's last error line, and rank 0
exit 0 without `checkpoint_evidence_final.pt` halts (never read as finished). A watchdog that cannot start (its temporary file
lives in `logs/`) stops the job before the trainer starts.

Per job, in the batch shell (no GPU): a **progress watchdog** every 60 s (`watchdog.sh`): no new step line for `STALL_MIN` (30, enough for the
longest eval, since the trainer prints an eval only when it ends; doubled while the log's last line is not a step line, 30
during startup), or no new save for `SAVE_MIN` (derived by
`train_job.sh`: max(45, ceil(2.5 x save interval / (GPUs x 54k tokens per second) / 60)) minutes, so 49 at 16 GPUs, 97 at 8,
386 at 2; plus the startup allowance for the job's first save; stored in `chain_state` and reported by `status.py`, so
`supervise.py` judges by the same number) -> SIGTERM to the ranks, kill after 3 minutes, the job is marked `stalled`;
**divergence**: the last `DIVERGE_K` (3) loss values nan, inf or above `LOSS_CEILING` (20), or the relative rule (a loss that
is finite but climbing: the median CE of the last `RISE_K` 30 step lines, taken once per poll, sits more than `RISE_MARGIN`
1.0 nats above the lowest such median of the job for `RISE_POLLS` 5 polls in a row, and only after `RISE_MIN` 15 polls of
grace; a falling loss, step noise of about 0.02 nats, an eval line and the warm to main switch, which is a new job, never come
near it) -> SIGKILL of the step with no save (a SIGTERM would make the trainer save the diverged state, and the rolling
retention would then delete the older healthy saves) and HALT (never restarted); nothing is deleted. `supervise.py` alerts
earlier on the median CE that `status.py` reports (`last_ce`, `best_ce`): +0.3 since the previous poll or +0.5 over the log
tail's best. Recovery after a halt on divergence: `keep_local_checkpoints: 2` leaves only the newest two rolling saves
(about 32 minutes of tokens), which may already be past the onset, so the safe point is the last 10B kept save
(`$DATA_ROOT/pinned` holds hard links to them); resume from a rolling save only after reading the log. If a save with
`lossnan` in its name exists anyway, status reports it and it must never be resumed from. **Crash loop** (`chain_history` in the run directory): 3 failed, stalled or diverged jobs among the last 5, two jobs in
a row that died within 10 minutes of start, or two consecutive jobs with no new save -> HALT with the reason. A
save timer from `squeue` plus `--signal=B:USR1@1200` stops the trainer with a save 20 minutes before the limit
(exit 20). NCCL: `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, heartbeat 900 s, and `sft.py` already sets a 30 minute collective timeout
(`InitProcessGroupKwargs`), so a dead rank ends the job. The corpus job stops a builder whose `run2_logs/last_child_output`
(child output, not heartbeat) is older than `STALL_MIN` (120) minutes, and one whose log ends in a mirror `waiting Ns` line with N over
`MIRROR_WAIT_MAX` (120) minutes (the mirror logs every 300 s, which keeps the file looking alive), and writes its `.halt` with
the reason. `HALT` holds the reason; `rm` it only after reading the log.

Per job outputs: `ckpts/inject/run2.log` (all ranks, appended), `logs/run2_JOBID.out`, `ckpts/inject/run2_gpu_hours.log`,
`ckpts/evidence_run2/{chain_state,chain_history,watchdog_last,HALT,STOP}` (a failure before the run directory is known writes
its HALT under `$REPO`, or under the submit directory when `$REPO` does not resolve). Accumulation is `16 / GPUs`; a count that
does not divide 16 is refused.

Hours: `status.py` computes the account's GPU hours from `sacct -X -P -o JobID,JobName,ElapsedRaw,AllocTRES,State` since
`--since` (every job that billed: killed at the limit, node failures, surplus, reads, corpus, the control; a job bills the larger of its gres count and 4 per node, so the exclusive one GPU
corpus job counts 4) and falls back to the per job log (`ckpts/inject/RUN_gpu_hours.log`) when sacct gives nothing.
GPU hours per job = nodes x 4 x hours. At 8 GPUs (the allocation's cap, the default since 2026-10-27) and 67.3k tokens per second per GPU, 800B is
about 413 hours (17 days) and 3,302 H100 hours; the floor is about 54k tokens per second per GPU (about 4,100 hours,
21 days). The GPU hours do not depend on the GPU count, the wall time does. Charging is probably per whole node
(unconfirmed): a one GPU job then bills 4 GPUs, so reads are batched four to a node.

## Supervision (the PC, in WSL)

The login node has no outgoing internet and the supervisor must work with one hand-opened ssh session. In a **WSL**
terminal (inside `tmux`, so it survives closing the window):

```
tmux new -s mn5
ssh -M -S ~/.ssh/mn5.sock -o ControlPersist=yes -o ServerAliveInterval=60 USER@alogin1.bsc.es
```

Log in once (password, 2FA if asked); leave it. Transfers (`push_code.sh`, `push_corpus.sh`, `pull_exports.sh`) go to
`transfer1.bsc.es` and need their **own** master socket (a ControlPath of the login host would send the rsync through
alogin1): in a second tmux window `ssh -M -S ~/.ssh/mn5-transfer.sock -o ControlPersist=yes USER@transfer1.bsc.es`
(`SSH_SOCK_TRANSFER` in `ops.env`). Every supervisor and transfer call reuses its socket with BatchMode and
never prompts; with a dead socket `supervise.py` exits 2. Transfers record the bytes in the ledger on failure too (from
`rsync --stats`, or the dry run estimate when rsync was killed) and keep interrupted files in `.rsync-partial`. After a PC reboot or sleep the socket is dead: repeat the
command (`ssh -S ~/.ssh/mn5.sock -O check USER@alogin1.bsc.es` tells). The run is unaffected. Then:

- `python scripts/ops/supervise.py [--act] [--summary]`, one ssh call (`status.sh`), exit 0 ok, 1 attention, 2 unreachable,
  last line `notify: ...` for a phone push. Rules: chain broken (no job listed, no `ended` in the chain state) -> resubmit,
  at most once an hour; rate under 54k per GPU for an hour; no save in the job's own limit while running (the chain state's save_min, 45 at
  least, plus the startup allowance for a first save); the CE median rising (+0.3 since the last poll, +0.5 over the log tail best);
  an action that exited nonzero; a chain cancelled by hand (never resubmitted); HALT, exit 30, an ended chain, a watchdog verdict or a silent watchdog (reported, never restarted);
  a pending job whose reason cannot clear (QOS or association limits, held) or pending a day; a read that ran and left no result (attempted
  once, never retried; a failed submission is retried); reads are batched, one reads job when four saves are due, the oldest
  has waited 12 hours or the run is over; a corpus `.halt`; a nan save; disk over 90% of the quota or unknown; projected GPU hours over
  4,500 (with the token target that fits); squeue failing (exit 2, no action). Milestone saves are the first save at or past
  each multiple of 10B (the retention rule), not a tag that is a multiple. One poll at a time (a lock beside the history);
  history, log and the attempted-reads record are per run name under `ops_state/`. `--act` does only the safe actions. Settings: flags, then the environment, then `scripts/ops/ops.env` (read directly, no export needed). Standing
  instructions for the local Claude session that calls it every 30 to 60 minutes: `scripts/ops/SUPERVISOR.md`.
- `bash scripts/slurm/status.sh` on the login node for the same JSON by hand.

## Saves and copying weights out

Saves are 4.6 GB every 1B tokens; `keep_local_checkpoints: 2` keeps the newest two and `keep_every_tokens: 10B` the first
save at or past each 10B and the final (about 80 kept, 370 GB). `--act` hard links each 10B multiple into
`$DATA_ROOT/pinned` (free on one filesystem). A save you want at another count (warm-up reads) is deleted about two
saves later: `ln` it into `pinned` at once. Weights only: `reads.sbatch` with `EXPORT=1` writes `$DATA_ROOT/exports/run2_TAG.pt`
(bf16 model, about 0.67 GB, `.sha256`, strict load check); by hand `python scripts/export_weights.py -c CKPT --out ...`. Then
on the PC: `bash scripts/ops/pull_exports.sh [LOCAL_DIR] [--hf]` pulls exports and the run's read json and logs, verifies
every sha256, and with `--hf` uploads to the private `HF_EXPORT_REPO`; the bytes go in the ledger.

## Ask support@bsc.es

The charging unit (whole node or GPU) and whether a one GPU job bills 4; KillWait and whether `--signal=B:USR1` is
delivered; whether `sbatch` works from a compute node (not needed: the chain is presubmitted); the scratch purge rule
for `$TMPDIR` and project disk; the inode quota (corpus shards, 80 saves); the transfer allowance's counting (both
directions?); whether a CPU partition can run corpus stages; the `acc_debug` limits for this project.

## Untested

Everything under real Slurm: the signal trap, `scancel --signal` on a step, the `squeue %L` save timer, `afterany`
chains of exclusive 4 node jobs, srun with `--cpus-per-task` repeated (needed on Slurm 22+; `SLURM_CPU_BIND=none` if
binding fails), the container on the cluster, `status.sh` against real `squeue` and `sacct`, and whether the trainer's
`Tokens/sec` and `evidence tokens` are the global figures (the watchdog and rate rules assume so).
