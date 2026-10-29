#!/usr/bin/env bash
# R5 (the winner, eval only), run 2026-10-25: the A-gate ladder on the written-bridge final
# (ckpts/evidence_pilot_bridge, arm P's final after R9's SFT, the form 6f builds on), one read
# per process, each skipped when its json exists, so the queue can be killed between reads and
# relaunched. arm P's own ladder is complete (ladder_pilot.sh, ladder_pilot_redo.sh, the bge edit).
# outputs ckpts/inject/pilot_bridge_ladder_*. usage: bash r5_bridge_ladder.sh
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
export TINY_LLM_CONFIG=config_pilot.yaml
K=ckpts/evidence_pilot_bridge/checkpoint_evidence_final.pt
B=ckpts/inject
L=$B/r5_bridge_ladder.log
run() {  # run NAME CMD...
  local name=$1; shift
  local out=$B/$name
  [ -f $out.json ] && { echo "skip $name" | tee -a $L; return; }
  local t0=$(date +%s)
  "$@" --json-out $out.json > $out.log 2>&1
  echo "$name exit $? in $(( $(date +%s) - t0 ))s" | tee -a $L
}
echo "r5 bridge ladder start $(date -Is)" | tee -a $L
FACTS="--facts data/prepared_pilot/inject_facts.jsonl --pools data/prepared_pilot/inject_pools.json --store data/index/pilot_bios"
for ev in none gold swapped; do
  run pilot_bridge_ladder_rank_$ev python scripts/closed_book_rank.py bios -c $K --form both --evidence $ev --batch-size 512 $FACTS
done
run pilot_bridge_ladder_ceiling_fixed python scripts/evidence_ceiling_probe.py -c $K --fixed-split evidence_fixed --data-dir data/prepared_pilot --max-questions 2000
run pilot_bridge_ladder_abstention python scripts/eval_abstention.py -c $K --evidence-port --evidence-condition gold,counterfactual,distractors,none --max-examples 2000 --batch-size 16
run pilot_bridge_ladder_store_pathway python scripts/eval_store.py pathway -c $K --store data/index/openqa --buffer 4 --loop 1
run pilot_bridge_ladder_store_edit_bge python scripts/eval_store.py edit -c $K --store data/index/openqa --edit-store data/index/openqa_edit --buffer 4 --loop 1 --route bge
run pilot_bridge_ladder_exit_clean python scripts/eval_exit.py -c $K --data-dir data/prepared_pilot --split inject_val_clean --max-batches 40 --batch-size 4
run pilot_bridge_ladder_bench python scripts/eval_benchmarks.py -c $K --compare docs/measurements/benchmarks/*.json
echo "r5 bridge ladder complete $(date -Is)" | tee -a $L
