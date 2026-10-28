# Run 2 on MareNostrum 5: handover

Written 2026-10-27, revised 2026-10-28 (the relay VM, the benchmark reads, serving retrieval, post-training on the PC).
What you do, and how the cluster work goes. The details are in
[scripts/slurm/README.md](../../scripts/slurm/README.md) (the runbook), [RUN2_corpus.md](RUN2_corpus.md) (the corpus)
and [scripts/ops/SUPERVISOR.md](../../scripts/ops/SUPERVISOR.md) (the supervising session).

## The whole plan, start to a usable model

1. **Before access:** ask BSC support three questions (below), commit, build the container, stage the small inputs.
2. **Cluster, days 1 to about 5:** the day 1 checks, the corpus build.
3. **Cluster, about 17 to 21 days:** the 2 GPU check, the 10B control, the warm-up and the switch, then the run to 800B.
   The relay VM watches it, reads every kept save (the closed-book and chain reads plus the 13-task benchmark suite)
   and uploads exports and reads to your HF repo.
4. **PC, during the run:** prove the post-training recipe on the pilot model (running since 2026-10-28), and build a
   general-knowledge store (not built yet; see "During the run, on the PC"). The tail phase is built (below): the run stops
   by itself at 760B and you start the last 40B.
5. **PC, after the run:** the consolidation SFT on the final weights (under 5 hours on the 5090), its reads, then chat
   with the store attached (`inference.py --store`).

What the run alone gives you is a base model. It chats only after step 5, and it knows things mainly through the
store: closed-book answers at this size stay weak by design.

## The plan in five lines

- 800B tokens: 266.67B unique prompt tokens per split, 3 epochs, on 8 H100 (2 nodes, the allocation's cap), as a chain
  of 72 hour jobs (about 6 to 8 of them).
- About 3,300 GPU hours at the expected rate (67k tokens per second per GPU), about 413 hours (17 days) of wall time,
  21 days at the floor rate, plus queue waits between jobs; the run fits
  4,500 hours as long as the rate stays above about 54k per GPU (about 59k if the corpus build's estimated 400 hours
  come out of the same budget).
- Order: corpus, the 2 GPU check, the 10B control, the warm-up (stop at 4B, read), the switch to the main split, the run
  to 760B, the switch to the tail split (the last 40B), the end at 800B.
- No node has internet. Everything goes in through the transfer nodes from your PC (code, corpus sources, container).
- A small always-on VM (Hetzner CPX22) watches the run, pulls the results and uploads them to a private HF repo, and
  alerts your phone through ntfy; the PC need not stay on after day 1's pushes (see "The relay VM").

## What you have to do

**Before access (PC, WSL)**
0. Ask BSC support: (a) is unattended ssh key login (no 2FA prompt) allowed from a fixed server IP to alogin1 and
   transfer1 (the relay depends on it); (b) is copying the project's own model weights to an external server within the
   allocation's terms; (c) is the 8 GPU cap per job or for all jobs at once (if at once, the reads jobs only run in the
   gaps between training jobs).
1. Commit the tree (the one line command from the session).
2. Decide where the corpus is built: on the cluster (about 400 GPU hours of node time, an estimate; about 450 GB of
   filler sources pushed in, the listing in RUN2_corpus.md) or
   on the PC (needs about 3 TB of local disk; D: has about 125 GB free). The cluster is the default.
3. Build the container: `bash scripts/slurm/container/build.sh` (or `build.sh docker`). Check that the smoke lines pass.
4. Stage the tokenizer, bge and the benchmark datasets: `python scripts/mirror_sources.py stage-small --hf-home DIR`
   (needs your HF token; without the benchmark data every benchmark read on the cluster fails). If you staged
   `HF_HOME` before commit 20f17aa (2026-10-28 review), stage it again: the job scripts now export
   `TINY_LLM_TOKENIZER` from `$HF_HOME/tokenizer` and fail fast if it is missing, and the per save reads need the
   benchmark datasets offline.
5. Copy `scripts/ops/ops.env.example` to `scripts/ops/ops.env` and fill in user and paths.

**Day 1 (cluster)**
1. Open the two ssh master sockets in `tmux` in WSL (README, "Supervision"). Log in by hand once; 2FA is yours alone.
2. `bash scripts/ops/push_code.sh`, then copy the container and the staged models over.
3. On the cluster: copy `scripts/slurm/mn5_env.local.sh.example` to `mn5_env.local.sh` and fill in the account,
   `acc_ehpc`, the paths, `SIF` and the `module load singularity` line.
4. Run the day 1 checks in the README, in order: queues and quota, the 5 minute signal test, the container smoke, NCCL
   on 2 nodes.
5. **Memory probe (2026-10-28).** The cards are H100 64 GB and RUN2's worst case sum was written for 80 GB. On one node
   at batch 8 run `sft.py --evidence --probe-peak` (README Day 1, step 5a; `--probe-chunk-tokens` and
   `--probe-conversations` set the synthetic layouts). It prints the per rank peak GB for a QA like and a chain like
   worst case, exits 0 and saves nothing. If the peak is above about 58 GB, lower `max_evidence_tokens` in
   `config_run2.yaml` before the warm-up; after the first save the value is frozen (a resume with another value exits 3).
6. **A real `scancel` test.** The planned stop path (Slurm's `scancel --signal=TERM` on a step) never ran under Slurm; the
   review found it killed the DataLoader workers. Submit a short job, `scancel --signal=TERM` its step as the chain does,
   and check that it saves and exits 10 (README Day 1).
7. Run the supervisor without `--act` (`python scripts/ops/supervise.py` in WSL) until a chain exists; it resubmits
   only a chain that has started, carrying its split, `STOP_AFTER` and nodes, never past a reached `STOP_AFTER`. Then
   run one action by hand (`--act`) and check that the job shows up in `squeue`. The tests could not cover the real ssh
   shell.
8. Hand cancels go through `bash scripts/slurm/cancel_chain.sh` only (it writes ended=cancelled, then scancels and
   verifies). Any other TERM counts as an interruption and the chain continues.

**Corpus (days 1 to about 5)**
1. Push the filler sources: `mirror_sources.py push` from the PC through the transfer node (RUN2_corpus.md).
2. Submit the stage groups in the README's order (`STAGES=... bash scripts/slurm/submit_chain.sh CORPUS 3`).
   Every long stage resumes after a kill.
3. **Before the `prune_sources` stage** (it deletes the QA builds): cut the post-training samples on the cluster,
   `python scripts/build_post_sft_mix.py qa --profile run2 --data-root /data/run2 --out-dir /data/run2/post` and the
   same with `replay`. Missed here, the run 2 post-training has no QA slice without rebuilding QA.

**Training**
1. The 2 GPU against 1 GPU check: README Day 1 step 5. Touch STOP after about 100 steps; compare the loss lines.
2. The 10B control: `bash scripts/slurm/submit_chain.sh CONTROL 1` (about 4 hours).
3. The warm-up: `STOP_AFTER='4000M' bash scripts/slurm/submit_chain.sh TRAIN 4`.
4. Check the first hour's rate against 54k tokens per second per GPU (the supervisor alerts below it).
5. At the 4B stop: read the save, decide the switch. Then
   `SPLIT=run2_main_train bash scripts/slurm/submit_chain.sh TRAIN 4`. This chain stops by itself after the first save at or
   past 760B (95% of the way in, at day 16 of about 17 at 67k tokens per second per GPU).
5b. **At the 760B stop (the tail phase).** The last 5% of the run, 40B tokens, is read from its own split, built with the
   corpus: 65% plain text from the best sources (dclm, finepdfs, math, code, wikipedia, a little fineweb, files the main run
   never read), 12% a sample of the main split (the biographies with the same swap and masking rules, so the leak reads
   stay comparable), 15% the evidence QA and chat rows, 8% the chains. The cosine does not change: it was fitted to 800B from
   step 0 and the tail is just where it ends. Nothing runs by itself past 760B: the chain ends, the supervisor alerts
   "tail switch due", and you start the tail (the supervisor never does):
   `rm ckpts/evidence_run2/STOP; SPLIT=run2_tail_train bash scripts/slurm/submit_chain.sh TRAIN 3`. Read the 760B save
   first if you want a before and after (the read set labels it 760B). The run then ends by itself at the plan's last step.
   If you would rather not have a tail, start the main split without the stop (`STOP_AFTER=''`) or restart the main chain
   after the stop with `SPLIT=run2_main_train STOP_AFTER=''`; the main split is then read to its end as before.
6. Enable the relay timer (The relay VM). The fallback, if no relay runs: a local Claude Code session in the repo,
   `/loop 45m` with SUPERVISOR.md as its standing instruction, Remote Control on for the phone; `/loop` expires after
   7 days, so re-arm it weekly. Never run both.

**Your calls, when they come up**
- The switch at the warm-up stop (and whether to read again at 12B).
- Anything after a HALT, an exit 30, or a chain you cancelled: read the reason, then restart by hand.
- Pulling weights out is automatic (the relay pulls them and uploads them to the HF repo; counted against the 2 TB
  ledger). By hand: `bash scripts/ops/pull_exports.sh`.
- The switch at the 760B stop to the tail split (above); nothing runs by itself past it.
- The general-knowledge store: which source and how big (same section).
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
   Since 2026-10-28 `OPS_LEDGER` in `ops.env` names one absolute ledger path, which the mirror pusher also writes; the
   PC and the relay VM still keep separate ledgers, so budget the 2 TB across them together.
9. **Relay settings added 2026-10-28** (in `ops.env`, defaults in `ops.env.example`): `HEARTBEAT_HOURS` (default 24, 0 =
   off; one low priority message a day with the status line, so silence means the relay is down), `LOG_UPLOAD_HOURS`
   (default 6; the logs go to the HF repo this often, exports and json reads at every poll), `HF_PYTHON` (the python with
   `huggingface_hub`, for example `~/relay-venv/bin/python`). The relay uploads only verified exports (a sha256 file
   written last, a 2 hour hash grace), refuses a public HF repo, keeps alert keys stable while numbers drift, records an
   ntfy message only after it was sent, and retries milestones.
9. **From the phone.** An ssh app (for example Termius) to the VM, then `python3 scripts/ops/supervise.py --summary`
   from `~/tiny-llm`.

If the VM is compromised: revoke the HF token, remove the `mn5_relay` line from `authorized_keys` on MareNostrum, and
change the ntfy topic. Cost: about 24 euro a month at the price you quoted, billed hourly; delete the server after the
run (after the last weights are in the HF repo).

## During the run, on the PC

- **The post-training rehearsal.** The same SFT recipe as step 5 below, on the pilot model (arm P's final, 97M
  parameters), 150M tokens of mix (chat 40%, evidence QA 25%, pretraining replay 35%), launched 2026-10-28 on the 5090
  (`ckpts/inject/postsft_pilot.log`, saves every 25M, read with `scripts/reads/read_post_sft_save.sh`). Done the same
  day: 140.7M tokens in 62 minutes, every save read with verdict ok. Seed to final: stops on its own 0.33 to 0.53 of 40
  chat prompts, replies that loop 0.90 to 0.80, gold-evidence exact match 0.044 to 0.091, abstention with the gold
  passage attached 0.82 to 0.70, clean-text CE 3.267 to 3.292 (the kill line is +0.15), gold answer CE on the fixed
  QA set 1.84 to 1.71. The recipe works without damage, but it fixes too little: the model still abstains on 70% of
  answerable questions with the answer in front of it, and long replies still loop. Before the run 2 post-training,
  lower the share of abstention rows in the QA slice (half its rows teach "the passage doesn't cover that" today) and
  read the rehearsal again. The answers themselves stay poor at this size (97M on 1B tokens); that part says nothing
  about the 800B model. Balancing is loss-free as in run 2; the shift of picks onto the dense slots (0.05 to about
  0.5 within 100 steps) happens with or without the aux term and did not cost clean-text CE.
- **The tail phase (built 2026-10-28, your decision 2026-10-28: the last 5%, 40B tokens from 760B).** Most current models
  end on a short phase of their best data while the learning rate falls; this run does the same. The tail split is built
  with the corpus (days 1 to 5; stages `tail_text`, `merge_tail`, plan in RUN2_corpus.md section 2b, 42B tokens so that the run
  reaches its last step before the split ends), so nothing has to be done on the PC. What to know: the tail's plain text is
  fresh files at a higher quality mix, but wikipedia and code are almost used up by the main filler, so they are a small part
  of it (about 1.5B of the 27B); the evidence QA and the chat rows are the same builds the main run read, 4 more passes (the
  QA slice is 4.1 passes of its build, the chains 1.2); the biography slice is a sample of the main split, so it is mostly
  the old filler windows (the persons' own renderings are about 0.01% of it). Training in the tail is about 10% slower per
  token (evidence on about 44% of tokens against 4.5% in the main run), about 12 GPU hours in all (an estimate).
- **A general-knowledge store (not built).** Serving retrieval works (`inference.py --store DIR`, `gradio_app.py --store
  DIR`, 4 chunks per question, checked against the evaluation store: the right chunk in the top 4 for 16 of 20), but the
  only stores today are the evaluation stores (passages of about 3,000 QA questions) and the fictional biographies.
  A store over Wikipedia or a similar source is what makes the finished model useful; its size and source are open.

## After the run: post-training on the PC

The run's weights do not chat yet: they answer, abstain and stop only after a short consolidation SFT, which runs on your
RTX 5090 in under 5 hours. A single-format SFT erased every other ability at pilot size (R5), so the mix carries the
pretraining slices beside the chat and QA rows, and every save is read against the seed's own reads.

1. Pull the final bf16 export from the HF repo (`scripts/export_weights.py` made it, about 0.67 GB). The trainer loads it
   as it is, with `-c`: the file has a `model_state_dict` and the strict load takes bf16, so no conversion is needed; it
   restarts the optimizer and reseeds the fp32 masters from the bf16 weights.
2. The replay and QA samples were cut on the cluster during the corpus build (Corpus step 3: a seeded sample of
   `run2_main_train` and of the corpus's QA build, with their val splits). Copy `/data/run2/post/post_replay_*` and
   `/data/run2/post/postqa_*` to the PC's `data/prepared_post_run2/`.
3. On the PC, from the repo in WSL: `python scripts/build_post_sft_mix.py chat --profile run2` (needs `manifest.json`
   with the smoltalk2 holdout list in the repo root, or `--pull-manifest`), then `merge`, then
   `verify` (it must print `verify: ok`: shares within 3%, every assistant turn ending in a supervised EOS, and at least
   0.15 of the QA slice from SQuAD or HotpotQA). Also copy
   `evidence_fixed.*` (cluster `prepared`) and `inject_val_clean.*` (cluster `prepared_run2`) into `data/prepared_post_run2/`. Also copy `chains_eval.*` and `chains_heldout_tmpl.*` (cluster `prepared`) into the same
   directory: the bridge retention read needs them (without them it fails on every save; non-fatal, but the read is lost).
4. Read the seed first: `RUN=postsft_run2 CLEAN_DIR=data/prepared_post_run2 CHAINS_DIR=data/prepared_post_run2
   READ_CONFIG=config_run2_postsft.yaml bash scripts/reads/read_post_sft_save.sh EXPORT seed` (the baseline the verdict
   compares against). Delete any old seed abstention json first if it was read at a different `MAX_EXAMPLES`: the read now
   refuses with exit 4 on a mismatch. Export the same `CHAINS_DIR` for the per save reads in step 6.
5. Launch: `DATA_DIR=data/prepared_post_run2 FIXED_DIR=data/prepared_post_run2 bash scripts/launch_post_sft.sh config_run2_postsft.yaml EXPORT
   post_sft_train postsft_run2`, and watch `ckpts/inject/postsft_run2.log`. Check the peak memory in minute one: batch 2 is a start.
6. At every save (250M tokens) run `read_post_sft_save.sh CKPT TAG` with the same env as step 4; the last line is `verdict ok` or
   `verdict KILL`. Since 2026-10-28 the verdict reads 2000 examples: KILL on gold EM down with paired z at or under -3, on
   clean CE at the last exit up more than 0.15, or on an abstention kill (the non-abstain rate under none or distractors
   up with z at or over +3 and by 0.02 or more); a missing seed baseline exits 4 (read the seed first, step 4); the bridge
   retention read is reported only. SQuAD comes from a local directory: set `SQUAD_DIR` (squad_v2 validation parquet shards;
   default the benchmark scratch copy under `data/benchmarks`) and, for the bridge read, `CHAINS_DIR` (default
   `data/prepared`; `DATA_ROOT/prepared` on the cluster). The post-training config runs with
   `conversation_loss_weighting: false`. On KILL stop the run (`touch ckpts/evidence_postsft_run2/STOP`), lower the learning rate or the chat share, and
   read the generations in `ckpts/inject/postsft_run2_TAG_chat.json` yourself: the numbers say whether it stops and repeats,
   not whether the answers are good.
7. Talk to it: `python scripts/inference.py -c CKPT --store data/index/STORE --show-retrieval -p "QUESTION"` or the
   Gradio app with the same `--store`. Without `--store` it answers from memory alone, which this model was trained not
   to rely on.

Untested so far: the `replay` and `qa` stages of the run2 profile (step 2 and 3 need the cluster's corpus and the Hub)
and batch 2 at shape L with this mix.
