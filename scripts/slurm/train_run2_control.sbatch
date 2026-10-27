#!/usr/bin/env bash
#SBATCH --job-name=run2_control
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --exclusive
#SBATCH --time=71:50:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/%x_%j.out
# the 10B control of the real run (RUN2 row 8): the same chain as train_run2.sbatch on the control split and yaml,
# no retrieval, one pass to the end. submit from the repo root through the chain script (job name run2_control):
#   bash scripts/slurm/submit_chain.sh CONTROL 1
# at 16 GPUs it is about 2 hours (10B at about 1.6M tokens per second without evidence), well inside one job.
export MODE=control JOB_SCRIPT=scripts/slurm/train_run2_control.sbatch
source "${SLURM_SUBMIT_DIR:?submit from the repo root}/scripts/slurm/train_job.sh"
