# Run 2 on MareNostrum 5: handover

Written 2026-10-27. What you do, and how the cluster work goes. The details are in
[scripts/slurm/README.md](../../scripts/slurm/README.md) (the runbook), [RUN2_corpus.md](RUN2_corpus.md) (the corpus)
and [scripts/ops/SUPERVISOR.md](../../scripts/ops/SUPERVISOR.md) (the supervising session).

## The plan in five lines

- 800B tokens: 266.67B unique prompt tokens per split, 3 epochs, on 8 H100 (2 nodes, the allocation's cap), as a chain
  of 72 hour jobs (about 6 to 8 of them).
- About 3,300 GPU hours at the expected rate (67k tokens per second per GPU), about 413 hours (17 days) of wall time,
  21 days at the floor rate, plus queue waits between jobs; the run fits
  4,500 hours as long as the rate stays above about 54k per GPU (about 59k if the corpus build's estimated 400 hours
  come out of the same budget).
- Order: corpus, the 2 GPU check, the 10B control, the warm-up (stop at 4B, read), the switch to the main split, the run.
- No node has internet. Everything goes in through the transfer nodes from your PC (code, corpus sources, container).
- A small always-on VM (Hetzner CPX22) watches the run, pulls the results and uploads them to a private HF repo, and
  alerts your phone through ntfy; the PC need not stay on after day 1's pushes (see "The relay VM").

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
2. The 10B control: `bash scripts/slurm/submit_chain.sh CONTROL 1` (about 4 hours).
3. The warm-up: `STOP_AFTER='4000M' bash scripts/slurm/submit_chain.sh TRAIN 4`.
4. Check the first hour's rate against 54k tokens per second per GPU (the supervisor alerts below it).
5. At the 4B stop: read the save, decide the switch. Then
   `SPLIT=run2_main_train bash scripts/slurm/submit_chain.sh TRAIN 4`.
6. Enable the relay timer (The relay VM). The fallback, if no relay runs: a local Claude Code session in the repo,
   `/loop 45m` with SUPERVISOR.md as its standing instruction, Remote Control on for the phone; `/loop` expires after
   7 days, so re-arm it weekly. Never run both.

**Your calls, when they come up**
- The switch at the warm-up stop (and whether to read again at 12B).
- Anything after a HALT, an exit 30, or a chain you cancelled: read the reason, then restart by hand.
- Pulling weights out is automatic (the relay pulls them and uploads them to the HF repo; counted against the 2 TB
  ledger). By hand: `bash scripts/ops/pull_exports.sh`.
- Still open from the plan: the control's exposure (probe tiers or a denser build) and 6f's abstention reward.

## How the cluster work goes

- **Jobs.** Each training job runs up to 72 hours on 2 nodes (8 GPUs, gradient accumulation 2, the same 524,288
  tokens per update as planned for 16). Twenty minutes before its limit it saves and exits; the next job,
  already queued behind it, resumes from that save. Saves every 1B tokens; the newest two and the first one at or past
  each 10B are kept (about 370 GB).
- **On the cluster side.** A watchdog in every job kills and halts it on any of these: a stall (no step for 15 minutes), no
  save within the job's limit (97 minutes at 8 GPUs), a loss that turns NaN or climbs and stays up, a crash loop (3 bad jobs
  in the last 5, 2 early deaths, 2 jobs with no new save). Any setup error that would repeat (container, paths, config,
  missing row estimates) halts the chain on the first try, with the reason in `HALT`. Nothing is ever deleted on a halt.
- **On the relay side** (the PC when no relay runs). The supervisor reads one status JSON per poll. It resubmits a broken chain at most once an hour,
  pins the 10B saves, batches the evaluation reads and alerts you on everything else. It never cancels, never deletes,
  never switches the split and never restarts after a HALT. Its alerts include a slow rate, a missing save, a rising loss,
  the disk, and GPU hours projected over 4,500.
- **If the relay loses its connection** the run goes on untouched. You get one alert per 6 hours ("cannot reach").
  The socket service retries every 5 minutes by itself; if the cluster asks for a new login or blocks the key, fix it
  by hand from the PC.
- **Getting results out.** Weights are exported as bf16 (about 0.67 GB each) and pulled by the relay with
  `pull_exports.sh` every poll, then uploaded to the private HF repo. The full checkpoints stay on the project disk.

## The relay VM

A small server that is always on: it polls the cluster every 45 minutes, resubmits a broken chain, pulls exports, reads
and logs, uploads them to a private HF repo and sends phone alerts through ntfy. No Claude runs on it. It uses
about 55 GB of its 80 GB disk over the run (about 80 exports of 0.67 GB).

0. **Before buying, ask BSC support two questions.** (a) Is unattended ssh key login (no 2FA prompt) allowed from a
   fixed server IP to alogin1 and transfer1? (b) Is copying the project's own model weights to an external server
   within the allocation's terms? If key login is not allowed, the relay cannot work and the PC `/loop` stays the plan.
1. **Create the server.** Hetzner Cloud CPX22 (2 vCPU, 4 GB RAM, 80 GB NVMe), Ubuntu 24.04, your own ssh key, a German
   or Finnish location. Hetzner firewall: inbound ssh only.
2. **Harden.** In `/etc/ssh/sshd_config` set `PasswordAuthentication no`, restart ssh. Create a non-root user, then
   `sudo apt install unattended-upgrades rsync python3 python3-venv curl`.
3. **Code.** As that user: `git clone -b ir-train-build https://github.com/ArminBurkhardt/tiny-moe-llm ~/tiny-llm`. The
   repo is private, so use a read-only deploy key (GitHub, repo settings, Deploy keys) or rsync `scripts/` from the PC
   instead. The relay needs only `scripts/ops` (the cluster side `scripts/slurm/status.sh` runs on MareNostrum).
4. **Python and HF.** `python3 -m venv ~/relay-venv`, then `~/relay-venv/bin/pip install huggingface_hub==1.32.0` (pinned:
   `relay_poll.sh` uses `hf upload-large-folder`, which newer versions may rename; the CLI `hf` comes with the package).
   `~/relay-venv/bin/hf auth login` with a fine-grained token that can write only to the one private repo.
5. **A key for the cluster.** `ssh-keygen -t ed25519 -f ~/.ssh/mn5_relay`. From your PC session (where you are logged in
   with 2FA), append `~/.ssh/mn5_relay.pub` to `~/.ssh/authorized_keys` on MareNostrum. Test from the VM:
   `ssh -i ~/.ssh/mn5_relay -o BatchMode=yes USER@alogin1.bsc.es hostname`, and the same with `transfer1.bsc.es`.
6. **Settings and phone.** Copy `scripts/ops/ops.env` from the PC (same paths) and add `NTFY_TOPIC` (`openssl rand -hex 12`),
   `HF_EXPORT_REPO`, and absolute paths for `SSH_SOCK` and `SSH_SOCK_TRANSFER` (for example `/home/USER/.ssh/mn5.sock`
   and `/home/USER/.ssh/mn5-transfer.sock`). Copy `scripts/ops/relay/relay.env.example` to `relay.env` and fill it with
   the same user and socket paths. Install the ntfy app on your phone, subscribe to the topic, and send a test:
   `curl -d "test" https://ntfy.sh/TOPIC`.
7. **Start it.** `mkdir -p ~/.config/systemd/user`, copy the three unit files (`mn5-sock@.service`, `relay.service`,
   `relay.timer`) there, `loginctl enable-linger USER`, then
   `systemctl --user daemon-reload` and
   `systemctl --user enable --now mn5-sock@login mn5-sock@transfer relay.timer`. Check with
   `systemctl --user list-timers`, `journalctl --user -u relay`, and one manual run from `~/tiny-llm`:
   `bash scripts/ops/relay/relay_poll.sh`. The poll log is `ops_state/relay.log`.
8. **Move the ledger.** The transfer ledger `ops_transfer_ledger.tsv` lives on whichever machine pulls. Copy it from the
   PC to `~/tiny-llm/` on the VM and stop pulling from the PC, otherwise the 2 TB count splits across two files.
9. **From the phone.** An ssh app (for example Termius) to the VM, then `python3 scripts/ops/supervise.py --summary`
   from `~/tiny-llm`.

If the VM is compromised: revoke the HF token, remove the `mn5_relay` line from `authorized_keys` on MareNostrum, and
change the ntfy topic. Cost: about 24 euro a month at the price you quoted, billed hourly; delete the server after the
run (after the last weights are in the HF repo).
