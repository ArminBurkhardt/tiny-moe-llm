#!/usr/bin/env bash
# the per-save read set for one real run checkpoint (shape L), the read_pilot_save.sh set on the
# run's corpus root. usage: bash scripts/reads/read_run2_save.sh CKPT TAG [CHAINS] [EXIT]
#   CKPT   the checkpoint file
#   TAG    token count or "final", used in the output names
#   CHAINS 1 (default) runs eval_chains.py too; 0 skips it
#   EXIT   1 also reads the per exit CE on the clean slice (eval_exit.py, 40 batches of 4, no evidence); default 0
# outputs: ckpts/inject/run2_${TAG}_rank_{none,gold,swapped,prompt}.{json,log},
#          ckpts/inject/run2_${TAG}_chains_{d123,hop4}.{json,log}, run2_${TAG}_exit_clean.{json,log}; every read is timed
#          and skipped when its json exists, so a killed read set relaunches where it stopped.
# env: DATA_ROOT (default data/run2: merges and facts in DATA_ROOT/prepared_run2, chains in
#      DATA_ROOT/prepared, the bios store in DATA_ROOT/index/run2_bios), RANK_BS (default 1024: the
#      pilot read at 512 on 32 GB; batch size does not change a rank read, only its speed),
#      CHAINS_BS (default 512, the pilot's 256), TINY_LLM_ROOT and TINY_LLM_ENV_INIT as tests/run_tests.sh.
# what fixes comparability with the pilot reads: --form both, the four evidence modes, --max-questions
# 2000, the same split names and depths (1,2,3 and 3,4); batch sizes are speed only. both batch
# defaults are untested on the 80 GB card at width 768: lower them on an OOM, not the question counts.
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}" || exit 1
source "${TINY_LLM_ENV_INIT:-env_init}"
C=$1; TAG=$2; CHAINS=${3:-1}; EXIT=${4:-0}
export TINY_LLM_CONFIG=config_run2.yaml
DATA_ROOT=${DATA_ROOT:-data/run2}
RANK_BS=${RANK_BS:-1024}
CHAINS_BS=${CHAINS_BS:-512}
B=ckpts/inject/run2_${TAG}
R=$DATA_ROOT/prepared_run2
FACTS="--facts $R/inject_facts.jsonl --pools $R/inject_pools.json --store $DATA_ROOT/index/run2_bios"
run() {  # run OUT CMD...   skip when OUT.json exists
  local out=$1; shift
  [ -f $out.json ] && { echo "skip $out"; return; }
  local t0=$(date +%s)
  "$@" --json-out $out.json > $out.log 2>&1
  echo "$(basename $out) exit $? in $(( $(date +%s) - t0 ))s"
}
for ev in none gold swapped prompt; do
  run ${B}_rank_$ev python scripts/closed_book_rank.py bios -c $C --form both --evidence $ev --batch-size $RANK_BS $FACTS
done
if [ "$CHAINS" = 1 ]; then
  run ${B}_chains_d123 python scripts/eval_chains.py -c $C --data-dir $DATA_ROOT/prepared --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 1,2,3 --sites all --max-questions 2000 --batch-size $CHAINS_BS
  run ${B}_chains_hop4 python scripts/eval_chains.py -c $C --data-dir $DATA_ROOT/prepared --splits chains_hop4 --depths 3,4 --sites all --max-questions 2000 --batch-size $CHAINS_BS
fi
if [ "$EXIT" = 1 ]; then
  run ${B}_exit_clean python scripts/eval_exit.py -c $C --data-dir $R --split inject_val_clean --max-batches 40 --batch-size 4
fi
echo "reads complete for $C"
