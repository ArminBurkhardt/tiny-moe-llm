cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
# one leak candidate arm of 2026-10-24 on the copy-first warm-up recipe (the R0b warm-up arm:
# 100M on the swap rate 1.0 split, then the swap rate 0.15 split to the epoch's end, micro shape,
# key/value reader, seed_micro.pt), with the builder flag's splits in place of the originals.
# usage: bash launch_leak_arm.sh m1|g1|cfb
#   cfb the flagless splits (inject_retrieval_s100a50_train, inject_retrieval_train): the recorded
#       warm-up arm's recipe rerun on the code in the tree, the same-code twin of m1 and g1
#   m1  the first name mention masked (inject_retrieval_s100a50m1_train, inject_retrieval_s15a50m1_train)
#   g1  support from the gold card only (inject_retrieval_s100a50g1_train, inject_retrieval_s15a50g1_train)
# reads: closed_book_rank none and gold at the 100M save (the switch criterion) and at the final,
# both forms, batch 1024, into ckpts/inject/rank_leak_${ARM}_{TAG}_{none,gold}.{json,log}
ARM=$1
case "$ARM" in m1|g1|cfb) ;; *) echo "usage: launch_leak_arm.sh m1|g1|cfb"; exit 1;; esac
export TINY_LLM_CONFIG=config_micro.yaml
NAME=leak_$ARM
R=ckpts/evidence_$NAME
L=ckpts/inject/$NAME.log
if [ "$ARM" = cfb ]; then WARM=inject_retrieval_s100a50_train; MAIN=inject_retrieval_train;
else WARM=inject_retrieval_s100a50${ARM}_train; MAIN=inject_retrieval_s15a50${ARM}_train; fi
mkdir -p $R
rm -f $R/STOP
echo "warm-up launch $(date -Is)" | tee -a $L
( until ls $R/checkpoint_evidence_tok100M_*.pt >/dev/null 2>&1; do sleep 5; done; touch $R/STOP ) &
python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name $NAME --data-dir data/prepared_inject --train-split $WARM --val-split inject_val 2>&1 | tee -a $L
echo "warm-up trainer exit ${PIPESTATUS[0]} $(date -Is)" | tee -a $L
rm -f $R/STOP
C100=$(ls -tr $R/checkpoint_evidence_tok100M_*.pt | tail -1)
for ev in none gold; do
  out=ckpts/inject/rank_${NAME}_100M_$ev
  python scripts/closed_book_rank.py bios -c $C100 --form both --evidence $ev --batch-size 1024 --json-out $out.json > $out.log 2>&1
  echo "read 100M $ev exit $?" | tee -a $L
done
echo "main launch $(date -Is)" | tee -a $L
python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name $NAME --data-dir data/prepared_inject --train-split $MAIN --val-split inject_val --switch-split 2>&1 | tee -a $L
echo "main trainer exit ${PIPESTATUS[0]} $(date -Is)" | tee -a $L
C=$R/checkpoint_evidence_final.pt
if [ -f $C ]; then
  for ev in none gold swapped; do
    out=ckpts/inject/rank_${NAME}_final_$ev
    python scripts/closed_book_rank.py bios -c $C --form both --evidence $ev --batch-size 1024 --json-out $out.json > $out.log 2>&1
    echo "read final $ev exit $?" | tee -a $L
  done
  python scripts/closed_book_rank.py compare ckpts/inject/rank_retrieval_kv_cffinal_none.json ckpts/inject/rank_${NAME}_final_none.json --class entity --by form,tier > ckpts/inject/compare_cf_vs_${NAME}.log 2>&1
  echo "compare exit $?" | tee -a $L
fi
echo "leak arm $ARM complete $(date -Is)" | tee -a $L
