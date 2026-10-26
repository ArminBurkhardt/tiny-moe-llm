cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
# the per-save read set for one pilot checkpoint. usage: bash read_pilot_save.sh CKPT ARM TAG [CHAINS]
#   CKPT   the checkpoint file
#   ARM    p or c (arm C skips the card reads, it has no reader to copy with)
#   TAG    token count or "final", used in the output names
#   CHAINS 1 (default) runs eval_chains.py too; 0 skips it (after the switch: every 100M)
# outputs: ckpts/inject/pilot_${ARM}_${TAG}_rank_{none,gold,swapped,prompt}.{json,log},
#          ckpts/inject/pilot_${ARM}_${TAG}_chains_{d123,hop4}.{json,log}; every read is timed.
C=$1; ARM=$2; TAG=$3; CHAINS=${4:-1}
export TINY_LLM_CONFIG=config_pilot.yaml
[ "$ARM" = e4 ] && export TINY_LLM_CONFIG=ckpts/inject/config_pilot_e4.yaml   # the 4 expert arm builds from its own yaml
B=ckpts/inject/pilot_${ARM}_${TAG}
FACTS="--facts data/prepared_pilot/inject_facts.jsonl --pools data/prepared_pilot/inject_pools.json --store data/index/pilot_bios"
if [ "$ARM" = p ] || [ "$ARM" = pi ] || [ "$ARM" = e4 ]; then EVS="none gold swapped prompt"; else EVS="none"; fi
for ev in $EVS; do
  out=${B}_rank_$ev
  [ -f $out.json ] && { echo "skip $out"; continue; }
  t0=$(date +%s)
  python scripts/closed_book_rank.py bios -c $C --form both --evidence $ev --batch-size 512 $FACTS --json-out $out.json > $out.log 2>&1
  echo "rank $ev exit $? in $(( $(date +%s) - t0 ))s"
done
if [ "$CHAINS" = 1 ]; then
  t0=$(date +%s)
  python scripts/eval_chains.py -c $C --data-dir data/prepared --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 1,2,3 --sites all --max-questions 2000 --batch-size 256 --json-out ${B}_chains_d123.json > ${B}_chains_d123.log 2>&1
  echo "chains d123 exit $? in $(( $(date +%s) - t0 ))s"
  t0=$(date +%s)
  python scripts/eval_chains.py -c $C --data-dir data/prepared --splits chains_hop4 --depths 3,4 --sites all --max-questions 2000 --batch-size 256 --json-out ${B}_chains_hop4.json > ${B}_chains_hop4.log 2>&1
  echo "chains hop4 exit $? in $(( $(date +%s) - t0 ))s"
fi
echo "reads complete for $C"
