cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init
run() {
  name=$1; ckpt=$2
  out=ckpts/instruments/cfll_$name
  [ -f $out.json ] && { echo "skip $name"; return; }
  python scripts/eval_abstention.py -c $ckpt --evidence-port --evidence-condition counterfactual --counterfactual-likelihood-only --json-out $out.json > $out.log 2>&1
  echo "done $name $?"
}
run seed ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt
run armA ckpts/evidence/checkpoint_evidence_tok10M_loss3.7712.pt
run armB ckpts/evidence_norope/checkpoint_evidence_tok10M_loss3.7359.pt
echo "cfll complete"
