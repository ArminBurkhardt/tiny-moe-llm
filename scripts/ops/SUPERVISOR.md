# Supervisor instructions (standing, for a local Claude Code session)

On the relay VM the same rules run without a Claude session: `scripts/ops/relay/relay_poll.sh` runs `supervise.py
--act`, notifies through ntfy on exit 1, on exit 2 (at most once per 6 hours) and on a milestone, pulls the exports and
reads and uploads them (see [the handover](../../docs/plans/RUN2_handover.md), "The relay VM"). The PC `/loop` below is
the fallback when no relay runs; never run both at once.

Relay rules, each in `relay_poll.sh`:

- An alert is keyed by `supervise.py`'s `keys:` line: its text with only the drifting figures (rate, minutes, CE,
  projected and pending hours) wiped, so a persistent condition pushes once per 6 hour window while an exit 20 and an
  exit 30, or the reads of two different saves, are different alerts; the message sent carries the current numbers. A different alert pushes at once. A send that failed is not recorded and is tried again at
  the next poll.
- A milestone is pushed on its own, once per milestone, whatever the exit code, so an alert in force never swallows it;
  one whose send failed waits in `ops_state/relay_milestones_pending.txt` and goes out at a later poll.
- Heartbeat: one low priority message per `HEARTBEAT_HOURS` (ops.env, default 24, 0 = off) saying the relay is alive,
  with the latest status line. Silence for longer than that means the relay, its timer or the VM is down.
- Uploads to the private HF repo happen only after a complete pull (`pull_exports.sh` exit 0, or 5 for a hash problem:
  then the bad file is deleted and alerted, and every export without a `.sha256` is excluded from the upload while the
  verified ones and the json reads go on). A `.pt` with no `.sha256` is only skipped for `MISSING_SHA_GRACE_MIN` (120)
  minutes, a problem after that. At startup the relay checks that `HF_PYTHON` (ops.env, default python3; the python of
  the hf CLI) imports huggingface_hub and that `hf` is on PATH, and alerts once with the real cause. A pair whose hash does not match is deleted on the relay so the next pull refetches it. The repo's visibility
  is checked before every upload (`hf repo create --private --exist-ok` leaves an existing public repo public): a public
  or unreadable repo gets an urgent ntfy and no upload. Exports and json reads go up at every poll, the growing logs
  (`*.log`, `joblogs/`) every `LOG_UPLOAD_HOURS` (default 6).
- Landed reads and exports are announced once each, and a name counts as announced only when its message went out.

Transfer ledger: `OPS_LEDGER` (ops.env; default `ops_transfer_ledger.tsv` in the repo root, always an absolute path) is
the one ledger of `push_code.sh`, `push_corpus.sh`, `pull_exports.sh` and `mirror_sources.py push` on a machine, whatever
the working directory, and the 2 TB budget is checked against it. The PC and the relay VM keep separate repo copies and
so separate ledgers: the allowance is shared across both (both directions count), so leave the relay's pulls out of what
the PC pushes (about 1 TB of corpus and filler leaves the rest to pulls) or point both at one file.

You supervise the real run on MareNostrum 5 from the user's Windows PC. You run every 30 to 60 minutes under `/loop`,
driven from the phone through Remote Control. Everything runs inside WSL: Linux ssh, rsync and python. No credential
goes to any cloud service; the cluster is reached only through the ssh master socket the user opened by hand
(`scripts/slurm/README.md`, "Supervision").

## One poll

1. In WSL, from the repo root: `python scripts/ops/supervise.py --act` (`--run run2_control` defaults the kind to CONTROL and reads with `READ_CONFIG=config_run2_control.yaml`; settings from flags `--user`, `--repo`,
   `--sock`, then the environment, then `scripts/ops/ops.env`, which the script reads itself, no `export` needed). It prints the verdict and a last line `notify: ...`.
2. Read the exit code: 0 ok, 1 attention needed, 2 cannot reach the cluster.
3. Append one line to the local ops log `ops_state/RUN_ops.log` (the script already does; add your own note only when you
   did something beyond it).
4. Push a phone notification only on exit 1, on exit 2, or on a milestone (the `milestone:` part of the notify line:
   every 100B tokens, run finished) and on the manual milestones below. Send the `notify:` line as the message. On exit 0
   with no milestone, say nothing.

Manual milestones to notify: the warm-up read at its stop (the `STOP_AFTER` save exists and the job ended: the switch
decision is the user's), the "tail switch due" alert (the main chain stopped by itself after the first save at or past
760B, the start of the tail phase: the switch to `run2_tail_train` is the user's, never yours), the 10B control finished, any new `run2_*B_rank_*.json` read (one line with the tag).

## Exit 2 (cannot reach the cluster)

The socket is dead (PC slept or rebooted, or the cluster closed it). Notify once, then do not retry more often than the
poll interval. The user reopens the socket; the run itself is not affected.

## What `--act` does by itself

Resubmits a broken chain, only after the chain has started (its first job wrote a chain id into the chain state, or the
run has a save: before the first launch, during the corpus build and during the 10B control there is no train job and
nothing is broken, a note says "chain not started"), with the chain's recorded split, node count and `STOP_AFTER`
(`START_INDEX=1`, so STOP, HALT and the stall rules still apply; a chain state with no split is an alert, not a
resubmit), hard links the 10B multiple
saves into `$DATA_ROOT/pinned` (the first save at or past each multiple of 10B), submits one batched reads job with weight
exports for kept saves not yet attempted, only once four are due, the oldest has waited 12 hours or the run is over. An action that exits nonzero is an alert and is not recorded as done. Nothing else. It never resubmits a chain a person cancelled (`scripts/slurm/cancel_chain.sh` writes `ended=cancelled` before it scancels, so the chain is reported, never resubmitted; a bare scancel counts as an interruption and the chain goes on; the person clears the end, README) or whose state names an end (halted,
max-jobs, stalled, warm-exhausted, a kill rule exit), never retries a failed read (report it), and exits 2 without acting
when squeue failed. A warm split that ran out ("warm split" alert) and the 760B stop ("tail switch due" alert) are switch decisions: notify the user, do nothing.

## Never without the user

- cancel a job (`scancel`), change a config or a yaml, delete data or checkpoints, switch the split (the warm-up to
  main switch, the main to tail switch), change the token target;
- restart after a HALT (read the reason in the verdict and report it) or after exit 30 (resume verification failed);
- edit `chain_state`, remove STOP or HALT, touch the run directory by hand;
- raise `--chain-n` beyond 6 or resubmit more than once per hour: if the chain keeps breaking, report it.

For an alert, report the verdict line and the one fact that explains it (the last error line, the rate, the quota),
and wait. A `watchdog` verdict or a HALT reason is already the cluster side's judgment: relay it, do not fix it.

## Recording

Each poll appends to `ops_state/RUN_history.jsonl` (state) and `ops_state/RUN_ops.log` (one line), RUN the run name (the control has its own
files); `ops_state/RUN_attempted.json` lists the reads already tried (delete an entry to allow one more try). Keep both. Pull weights and
logs out with `bash scripts/ops/pull_exports.sh` when the user asks, not on your own.

## Staying alive

`/loop` recurring tasks expire after 7 days. Re-arm the loop yourself on day 6 (and say so in the log), or the user
restarts it weekly. A session that lost its loop must be restarted by the user: tell them in the next notification.
