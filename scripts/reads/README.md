# scripts/reads

Tracked copies of the read, ladder and launcher scripts the measurement notes depend on. Until
2026-10-26 they lived only in the gitignored `ckpts/inject/`. The copies are verbatim, the record:
they still `cd` into the local WSL checkout, `source env_init`, and name pilot and micro paths, so
they run on the local box only. The real run's scripts are `read_run2_save.sh` and
`stop_after_save_run2.sh`, which take `TINY_LLM_ROOT` and `TINY_LLM_ENV_INIT` as
`tests/run_tests.sh` does.

The micro launchers read yamls that are still gitignored: `launch_review_arms.sh` reads
`ckpts/inject/config_micro_infonce0_e3.yaml`, `config_micro_infonce_l0_e3.yaml` and
`config_micro_segsel_e3.yaml`; `read_clean_val.sh` reads `ckpts/inject/config_pilot_e4.yaml`. Copy
them here before relying on a rerun from a fresh checkout.

| script | what it reads | quoted by | flags that fix comparability |
|---|---|---|---|
| `read_pilot_save.sh CKPT ARM TAG [CHAINS]` | one pilot save: `closed_book_rank.py bios` under `none gold swapped prompt` (arm C: `none` only), then `eval_chains.py` | `chain_continuation_pilot.md`, `expert_count_pilot.md`, `PILOT.md` | `--form both`, `--batch-size 512`, facts, pools and store of `data/prepared_pilot` and `data/index/pilot_bios`; chains `--splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 1,2,3 --sites all --max-questions 2000 --batch-size 256`, then `chains_hop4 --depths 3,4`; `e4` loads `ckpts/inject/config_pilot_e4.yaml`; `compare` is run by hand (the logs `compare_pilot_c_vs_p.log`, `compare_pilot_e4_vs_p.log` show the argument order used) |
| `ladder_pilot.sh` | arm P and C finals, 2026-10-14: ceiling on `evidence_fixed` (2000 questions), `eval_exit.py` on `inject_val` (40 batches), rank `none` at `--n-loops 1` and `2`, store recall, pathway and edit (model query route, void since the packer fix), abstention with the port, the 13-task suite | `PILOT.md`, `written_bridge_pilot.md`, `r5_bridge_ladder.md` | `TINY_LLM_CONFIG=config_pilot.yaml`; rank `--batch-size 512`; store reads `--store data/index/openqa --buffer 4 --loop 1`; abstention `--evidence-condition gold,counterfactual,distractors,none --max-examples 2000 --batch-size 16` (generation is comparable only at that batch size); suite `--compare docs/measurements/benchmarks/*.json` |
| `ladder_pilot_redo.sh` | arm P final, 2026-10-24: the pathway and abstention reads redone on the fixed evidence packer | `written_bridge_pilot.md`, `r5_bridge_ladder.md` | the same flags as `ladder_pilot.sh` for those two reads; outputs `pilot_p_ladder2_*`; the A4 edit on the bge route is `eval_store.py edit ... --route bge` (in `r5_bridge_ladder.sh`) |
| `r5_bridge_ladder.sh` | the written-bridge final, 2026-10-25: rank `none gold swapped`, ceiling, abstention, pathway, A4 edit `--route bge`, `eval_exit.py` on `inject_val_clean`, the suite | `r5_bridge_ladder.md`, `written_bridge_pilot.md`, `RUN2.md` | rank `--batch-size 512`; abstention batch 16, 2000 examples; exit `--max-batches 40 --batch-size 4`; each read skipped when its json exists |
| `read_clean_val.sh` | `eval_exit.py` on the arm P, C and E4 finals, `inject_val_clean` and `inject_val`, no evidence | `clean_validation_slice.md`, `infonce_micro.md`, `PILOT.md` | `--data-dir data/prepared_pilot --max-batches 40 --batch-size 4`, split order clean then `inject_val`; E4 under its own yaml |
| `read_bridge_save.sh CKPT TAG` | one written-bridge save: `eval_chains.py --generate-bridge` on `chains_eval,chains_heldout_tmpl` at depth 3 and `chains_hop4` at depth 4 | `written_bridge_pilot.md` | `--max-questions 1000 --batch-size 64` (the decode is left padded and comparable only at a fixed batch), `--sites all` at depth 3 and `--sites 4` at depth 4; decoded after the packer fix `af24c70` only |
| `launch_leak_arm.sh m1\|g1\|cfb` | a micro leak arm: warm-up to the 100M save, rank `none` and `gold` there, switch, read the final, `compare` against the warm-up arm | `leak_candidates_micro.md` | `config_micro.yaml`, seed `seed_micro.pt`, `--reader-kv`, rank `--batch-size 1024 --form both`, splits `inject_retrieval_s100a50{,m1,g1}_train` then `inject_retrieval_train` or `inject_retrieval_s15a50{m1,g1}_train`; `cfb` is the flagless same-code twin |
| `launch_review_arms.sh [w0e3b\|l0e3\|seg_e3]` | the three micro retriever arms of the 2026-10-24 review (three passes over `evidence_nomany_train`), then store recall at loop 1 | `infonce_micro.md` | one yaml per arm (gitignored, see above), the twin `w0e3b` first; `eval_exit.py --pair` is run by hand against the twin |
| `stop_after_save.sh ARM PATTERN` | none: waits for a matching save line in `ckpts/inject/pilot_ARM.log`, then touches `ckpts/evidence_pilot_ARM/STOP` | `expert_count_pilot.md`, `PILOT.md`, `NEXT.md` | the pattern is a range on the token tag (`'30[0-4]M'`), not a single number, because saves landed on a shifting grid before 2026-10-24 |
| `cf_likelihood.sh` | counterfactual likelihood only (`--counterfactual-likelihood-only`) on the seed and the two 10M arms, 2026-10-06 | `seed_instruments.md` (the `cfll_*` outputs) | `ckpts/instruments/cfll_{seed,armA,armB}`; paths of the repair-era checkpoints |
| `read_run2_save.sh CKPT TAG [CHAINS] [EXIT]` | one real run save, the `read_pilot_save.sh` set on `DATA_ROOT`, optionally the clean-slice per exit CE | `RUN2.md` (Reads and gates) | `config_run2.yaml`; `--form both`, the four evidence modes, `--max-questions 2000`, depths `1,2,3` and `3,4`, the pilot's split names; `RANK_BS` 1024 and `CHAINS_BS` 512 are speed only and untested on the 80 GB card; facts, pools and store from `DATA_ROOT/prepared_run2` and `DATA_ROOT/index/run2_bios`, chains from `DATA_ROOT/prepared`; the `compare` against the control is run by hand on the same tokens (the 10B save of each), in the order of the pilot logs |
| `stop_after_save_run2.sh RUN PATTERN` | none: the same wait on `ckpts/inject/RUN.log`, STOP in `ckpts/evidence_RUN/` | `scripts/launch_run2.sh` | the grid is fixed in tokens, so `'2000M'` style exact tags are safe; `FROM` (a log byte offset) ignores older text, set by the Slurm job (`scripts/slurm/`) |

Launchers beside this directory: `scripts/launch_run2.sh` and `scripts/launch_run2_control.sh`
(`--dry-run` prints the command). `tests/test_run2_launchers.py` runs both dry.

On the cluster (`scripts/slurm/README.md`) `read_run2_save.sh` runs under `scripts/slurm/reads.sbatch`, which sources `mn5_env.sh` (`TINY_LLM_ROOT`, `DATA_ROOT`, offline flags).
