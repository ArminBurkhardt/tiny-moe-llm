cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
# the three micro retriever arms after the 2026-10-24 review, each the w25e3 / w0e3 recipe (three
# passes over evidence_nomany_train from the micro seed with the key/value reader, about 104M
# tokens), then the store recall read on the final at loop 1:
#   w0e3b   the w0e3 recipe under the reviewed code (the twin every other arm is read against)
#   l0e3    the w25e3 recipe with the contrastive term on the first pass only and the trunk gradient on
#   seg_e3  the w0e3 recipe with the selection loss weighted per query segment
# usage: bash launch_review_arms.sh [w0e3b|l0e3|seg_e3 ...]   (default: all three in that order)
ARMS=${@:-"w0e3b l0e3 seg_e3"}
for ARM in $ARMS; do
  case "$ARM" in
    w0e3b) CFG=ckpts/inject/config_micro_infonce0_e3.yaml;;
    l0e3) CFG=ckpts/inject/config_micro_infonce_l0_e3.yaml;;
    seg_e3) CFG=ckpts/inject/config_micro_segsel_e3.yaml;;
    *) echo "unknown arm $ARM"; exit 1;;
  esac
  R=ckpts/evidence_infonce_$ARM
  L=ckpts/inject/infonce_$ARM.log
  mkdir -p $R
  rm -f $R/STOP
  echo "launch $ARM $(date -Is)" | tee -a $L
  TINY_LLM_CONFIG=$CFG python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name infonce_$ARM --data-dir data/prepared --train-split evidence_nomany_train --val-split evidence_nomany_val 2>&1 | tee -a $L
  echo "trainer exit ${PIPESTATUS[0]}" | tee -a $L
  C=$R/checkpoint_evidence_final.pt
  if [ -f $C ]; then
    out=ckpts/inject/infonce_${ARM}_store_recall
    TINY_LLM_CONFIG=$CFG python scripts/eval_store.py recall -c $C --store data/index/openqa --buffer 4 --loop 1 --json-out $out.json > $out.log 2>&1
    echo "store recall $ARM exit $?" | tee -a $L
  fi
done
echo "review arms complete"
