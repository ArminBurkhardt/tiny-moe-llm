#!/usr/bin/env bash
# the two ladder reads that decode with evidence attached (left padded), redone on arm P's final
# after the evidence packer fix of 2026-10-24: the abstention eval with the port (A3 in generation)
# and the store pathway (A7). outputs ckpts/inject/pilot_p_ladder2_{abstention,store_pathway}.*
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
export TINY_LLM_CONFIG=config_pilot.yaml
P=ckpts/evidence_pilot_p/checkpoint_evidence_final.pt
B=ckpts/inject
run() {  # run NAME CMD...
  local name=$1; shift
  local out=$B/$name
  [ -f $out.json ] && { echo "skip $name"; return; }
  local t0=$(date +%s)
  "$@" --json-out $out.json > $out.log 2>&1
  echo "$name exit $? in $(( $(date +%s) - t0 ))s"
}
run pilot_p_ladder2_store_pathway python scripts/eval_store.py pathway -c $P --store data/index/openqa --buffer 4 --loop 1
run pilot_p_ladder2_abstention python scripts/eval_abstention.py -c $P --evidence-port --evidence-condition gold,counterfactual,distractors,none --max-examples 2000 --batch-size 16
echo "ladder redo complete"
