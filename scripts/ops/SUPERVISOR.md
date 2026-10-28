# Supervisor instructions (standing, for a local Claude Code session)

On the relay VM the same rules run without a Claude session: `scripts/ops/relay/relay_poll.sh` runs `supervise.py
--act`, notifies through ntfy on exit 1, on exit 2 (at most once per 6 hours) and on a milestone, pulls the exports and
reads and uploads them (see [the handover](../../docs/plans/RUN2_handover.md), "The relay VM"). The PC `/loop` below is
the fallback when no relay runs; never run both at once.

You supervise the real run on MareNostrum 5 from the user's Windows PC. You run every 30 to 60 minutes under `/loop`,
driven from the phone through Remote Control. Everything runs inside WSL: Linux ssh, rsync and python. No credential
goes to any cloud service; the cluster is reached only through the ssh master socket the user opened by hand
(`scripts/slurm/README.md`, "Supervision").

## One poll

1. In WSL, from the repo root: `python scripts/ops/supervise.py --act` (settings from flags `--user`, `--repo`,
   `--sock`, then the environment, then `scripts/ops/ops.env`, which the script reads itself, no `export` needed). It prints the verdict and a last line `notify: ...`.
2. Read the exit code: 0 ok, 1 attention needed, 2 cannot reach the cluster.
3. Append one line to the local ops log `ops_state/RUN_ops.log` (the script already does; add your own note only when you
   did something beyond it).
4. Push a phone notification only on exit 1, on exit 2, or on a milestone (the `milestone:` part of the notify line:
   every 100B tokens, run finished) and on the manual milestones below. Send the `notify:` line as the message. On exit 0
   with no milestone, say nothing.

Manual milestones to notify: the warm-up read at its stop (the `STOP_AFTER` save exists and the job ended: the switch
decision is the user's), the 10B control finished, any new `run2_*B_rank_*.json` read (one line with the tag).

## Exit 2 (cannot reach the cluster)

The socket is dead (PC slept or rebooted, or the cluster closed it). Notify once, then do not retry more often than the
poll interval. The user reopens the socket; the run itself is not affected.

## What `--act` does by itself

Resubmits a broken chain (`START_INDEX=1`, so STOP, HALT and the stall rules still apply), hard links the 10B multiple
saves into `$DATA_ROOT/pinned` (the first save at or past each multiple of 10B), submits one batched reads job with weight
exports for kept saves not yet attempted, only once four are due, the oldest has waited 12 hours or the run is over. An action that exits nonzero is an alert and is not recorded as done. Nothing else. It never resubmits a chain a person cancelled (alert; the person clears it, README) or whose state names an end (halted,
max-jobs, stalled, warm-exhausted, a kill rule exit), never retries a failed read (report it), and exits 2 without acting
when squeue failed. A warm split that ran out ("warm split" alert) is the switch decision: notify the user, do nothing.

## Never without the user

- cancel a job (`scancel`), change a config or a yaml, delete data or checkpoints, switch the split (the warm-up to
  main switch), change the token target;
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
