#!/usr/bin/env bash
# the ladder's final reads on the pilot arms, in sequence, one log per read. usage: bash ladder_pilot.sh
# outputs under ckpts/inject/pilot_{p,c}_ladder_*; every read is timed and skipped when its json exists.
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
export TINY_LLM_CONFIG=config_pilot.yaml
P=ckpts/evidence_pilot_p/checkpoint_evidence_final.pt
C=ckpts/evidence_pilot_c/checkpoint_evidence_final.pt
B=ckpts/inject
run() {  # run NAME CMD...
  local name=$1; shift
  local out=$B/$name
  [ -f $out.json ] && { echo "skip $name"; return; }
  local t0=$(date +%s)
  "$@" --json-out $out.json > $out.log 2>&1
  echo "$name exit $? in $(( $(date +%s) - t0 ))s"
}
run pilot_p_ladder_ceiling_fixed python scripts/evidence_ceiling_probe.py -c $P --fixed-split evidence_fixed --data-dir data/prepared_pilot --max-questions 2000
run pilot_p_ladder_exit python scripts/eval_exit.py -c $P --data-dir data/prepared_pilot --split inject_val --max-batches 40
run pilot_c_ladder_exit python scripts/eval_exit.py -c $C --data-dir data/prepared_pilot --split inject_val --max-batches 40
FACTS="--facts data/prepared_pilot/inject_facts.jsonl --pools data/prepared_pilot/inject_pools.json --store data/index/pilot_bios"
for d in 1 2; do
  run pilot_p_ladder_rank_none_d$d python scripts/closed_book_rank.py bios -c $P --form both --evidence none --batch-size 512 --n-loops $d $FACTS
  run pilot_c_ladder_rank_none_d$d python scripts/closed_book_rank.py bios -c $C --form both --evidence none --batch-size 512 --n-loops $d $FACTS
done
run pilot_p_ladder_store_recall python scripts/eval_store.py recall -c $P --store data/index/openqa --buffer 4 --loop 1
run pilot_p_ladder_store_pathway python scripts/eval_store.py pathway -c $P --store data/index/openqa --buffer 4 --loop 1
run pilot_p_ladder_store_edit python scripts/eval_store.py edit -c $P --store data/index/openqa --edit-store data/index/openqa_edit --buffer 4 --loop 1
run pilot_p_ladder_abstention python scripts/eval_abstention.py -c $P --evidence-port --evidence-condition gold,counterfactual,distractors,none --max-examples 2000 --batch-size 16
run pilot_p_ladder_bench python scripts/eval_benchmarks.py -c $P --compare docs/measurements/benchmarks/*.json
run pilot_c_ladder_bench python scripts/eval_benchmarks.py -c $C --compare docs/measurements/benchmarks/*.json
echo "ladder complete"
