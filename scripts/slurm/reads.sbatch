#!/usr/bin/env bash
#SBATCH --job-name=run2_reads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# the per-save read set (scripts/reads/read_run2_save.sh) for up to four saves at once, one per GPU of one node, so a
# node's bill buys four reads. submit from the repo root after `source scripts/slurm/mn5_env.sh` and `mkdir -p logs`,
# with the job named run2_reads_<RUN> (the supervisor and status.py match that name exactly):
#   RUN=run2 PAIRS="ckpts/evidence_run2/a.pt:10B ckpts/evidence_run2/b.pt:20B" sbatch --job-name=run2_reads_run2 scripts/slurm/reads.sbatch
#   RUN=run2 CKPT=ckpts/evidence_run2/checkpoint_evidence_tok10003M_loss3.2000.pt TAG=10B sbatch --job-name=run2_reads_run2 scripts/slurm/reads.sbatch   (one)
# fewer than four pairs: add --gres=gpu:N --cpus-per-task=20N --exclusive=no at submit to bill less where the site charges per GPU.
# env: PAIRS (CKPT:TAG pairs, at most 4) or CKPT and TAG, RUN (default run2: the prefix of every output, reads
#      ckpts/inject/RUN_TAG_*, exports $DATA_ROOT/exports/RUN_TAG.pt, so the control never overwrites the main run),
#      READ_CONFIG (default config_run2.yaml: TINY_LLM_CONFIG of the reads and of the export), CHAINS (default 1),
#      EXIT (default 0), BENCH (default 1: also the 13-task suite, offline from $HF_HOME/datasets), EXPORT=1 (also write the weights export before each read), RANK_BS and CHAINS_BS (batch sizes
#      of the read script; lower them on an out of memory), DRY_RUN=1.
# each save is copied first to the node's temporary disk when one is set (the trainer's retention deletes rolling saves).
# a read is skipped when its json exists, so a killed job is resubmitted with the same line. the job exits nonzero when
# any read or export failed; the supervisor never resubmits a failed read by itself.
SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
source "$SUBMIT/scripts/slurm/mn5_env.sh"
cd "$REPO" || exit 2
source scripts/slurm/lib.sh
[ -z "$CTR_FAIL" ] || exit 2
RUN=${RUN:-run2}
READ_CONFIG=${READ_CONFIG:-config_run2.yaml}
PAIR_LINES=$(expand_read_pairs) || exit 2
[ "${DRY_RUN:-0}" = 1 ] || { require_site_values && require_tokenizer; } || exit 2
mkdir -p logs "$DATA_ROOT/exports"
i=0; PIDS=""; COPIES=""
while read -r CK TG; do
  C=$CK
  if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    cp "$CK" "$TMPDIR/read_${RUN}_$TG.pt" && C=$TMPDIR/read_${RUN}_$TG.pt && COPIES="$COPIES $C"
  fi
  CMD="CUDA_VISIBLE_DEVICES=$i RUN=$RUN TINY_LLM_CONFIG=$READ_CONFIG ${CTR}bash scripts/reads/read_run2_save.sh $C $TG ${CHAINS:-1} ${EXIT:-0} ${BENCH:-1}"
  EXP="CUDA_VISIBLE_DEVICES=$i TINY_LLM_CONFIG=$READ_CONFIG ${CTR}python scripts/export_weights.py -c $C --out $DATA_ROOT/exports/${RUN}_$TG.pt"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    [ "${EXPORT:-0}" = 1 ] && echo "$EXP"
    echo "$CMD"
  else
    ( if [ "${EXPORT:-0}" = 1 ] && [ ! -f "$DATA_ROOT/exports/${RUN}_$TG.pt.sha256" ]; then env $EXP || exit 1; fi
      env $CMD ) > "logs/reads_${SLURM_JOB_ID:-0}_${RUN}_$TG.log" 2>&1 &
    PIDS="$PIDS $!"
  fi
  i=$(( i + 1 ))
done <<< "$PAIR_LINES"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
echo "[$(date -Is)] $i read(s) started"
RC=0
for p in $PIDS; do wait "$p" || RC=$?; done
for c in $COPIES; do rm -f "$c"; done
echo "[$(date -Is)] reads exit $RC"
exit $RC
