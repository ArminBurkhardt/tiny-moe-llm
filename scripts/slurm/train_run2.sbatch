#!/usr/bin/env bash
#SBATCH --job-name=run2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --exclusive
#SBATCH --time=71:50:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/%x_%j.out
# the real run (docs/plans/RUN2.md) on MareNostrum 5, accelerated partition: 4 nodes x 4 H100 = 16 GPUs by default
# (accumulation 1). submit the chain from the login node, from the repo root (it names the job, chains the jobs with
# afterany, gives them one chain id and refuses while a chain of that name is queued):
#   bash scripts/slurm/submit_chain.sh TRAIN 5                                 warm-up split
#   SPLIT=run2_main_train bash scripts/slurm/submit_chain.sh TRAIN 5           the switch (adds --switch-split)
#   NODES=2 bash scripts/slurm/submit_chain.sh TRAIN 5                         8 GPUs, accumulation 2
# a job queues no successor (CHAIN_MODE=self restores that); it saves and exits 20 on the signal 20 minutes before the
# limit, and the chain ends on done, a STOP file, exit 30, a halt or two jobs without a new save. never scancel one job of
# a chain: cancel the whole chain (squeue -n run2) or drop the STOP file. variables and the whole logic:
# scripts/slurm/train_job.sh, the day by day use: scripts/slurm/README.md.
export MODE=run JOB_SCRIPT=scripts/slurm/train_run2.sbatch
source "${SLURM_SUBMIT_DIR:?submit from the repo root}/scripts/slurm/train_job.sh"
