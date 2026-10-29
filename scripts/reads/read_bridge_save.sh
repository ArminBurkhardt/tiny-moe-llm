cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
# the written bridge read of one R9 save: generate then score on the eval and held-out chain
# splits at depth 3 and on the 4-hop split at depth 4. usage: bash read_bridge_save.sh CKPT TAG
# outputs ckpts/inject/pilot_bridge_${TAG}_gen_{d3,hop4}.{json,log}
export TINY_LLM_CONFIG=config_pilot.yaml
C=$1; TAG=$2
B=ckpts/inject/pilot_bridge_${TAG}
t0=$(date +%s)
python scripts/eval_chains.py -c $C --data-dir data/prepared --splits chains_eval,chains_heldout_tmpl --depths 3 --sites all --max-questions 1000 --batch-size 64 --generate-bridge --json-out ${B}_gen_d3.json > ${B}_gen_d3.log 2>&1
echo "gen d3 exit $? in $(( $(date +%s) - t0 ))s"
t0=$(date +%s)
python scripts/eval_chains.py -c $C --data-dir data/prepared --splits chains_hop4 --depths 4 --sites 4 --max-questions 1000 --batch-size 64 --generate-bridge --json-out ${B}_gen_hop4.json > ${B}_gen_hop4.log 2>&1
echo "gen hop4 exit $? in $(( $(date +%s) - t0 ))s"
echo "bridge reads complete for $C"
