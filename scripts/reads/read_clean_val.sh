cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
# the three pilot finals' held-out CE on the clean validation slice (inject_val_clean, built
# 2026-10-24 by scripts/prepare_clean_val.py from filler windows the evidence builder never used)
# beside the same read on inject_val, per exit, through eval_exit.py. usage: bash read_clean_val.sh
L=ckpts/inject/read_clean_val.log
for spec in "p config_pilot.yaml ckpts/evidence_pilot_p/checkpoint_evidence_final.pt" "c config_pilot.yaml ckpts/evidence_pilot_c/checkpoint_evidence_final.pt" "e4 ckpts/inject/config_pilot_e4.yaml ckpts/evidence_pilot_e4/checkpoint_evidence_final.pt"; do
  set -- $spec
  ARM=$1; CFG=$2; CKPT=$3
  for split in inject_val_clean inject_val; do
    out=ckpts/inject/pilot_${ARM}_exit_${split}
    echo "read $ARM $split $(date -Is)" | tee -a $L
    TINY_LLM_CONFIG=$CFG python scripts/eval_exit.py -c $CKPT --data-dir data/prepared_pilot --split $split --max-batches 40 --batch-size 4 --json-out $out.json > $out.log 2>&1
    echo "read $ARM $split exit $?" | tee -a $L
  done
done
echo "clean val reads complete $(date -Is)" | tee -a $L
