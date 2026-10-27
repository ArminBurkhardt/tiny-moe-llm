#!/usr/bin/env bash
# the cluster environment (MareNostrum 5, accelerated partition), sourced by every job script, by the status and
# submit scripts and by you before sbatch (source scripts/slurm/mn5_env.sh), so the SBATCH_* variables reach sbatch.
# this file is tracked and holds no site value: every site value (project paths, account, qos, modules, the image)
# lives in scripts/slurm/mn5_env.local.sh, which is gitignored, never pushed by push_code.sh and filled in once on the
# cluster from mn5_env.local.sh.example. a job or a submit with a CHANGE placeholder left refuses to run
# (lib.sh require_site_values).
_MN5_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# a non interactive ssh shell (the supervisor's) may not have `module` defined: the overlay's `module load` would fail
# and the container runtime would be missing. initialise it here, quietly, from the usual places
if ! type module >/dev/null 2>&1; then
  for _mn5_f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    # shellcheck disable=SC1090
    [ -f "$_mn5_f" ] && source "$_mn5_f" >/dev/null 2>&1
    type module >/dev/null 2>&1 && break
  done
  unset _mn5_f
fi
# shellcheck disable=SC1091
[ -f "$_MN5_HERE/mn5_env.local.sh" ] && source "$_MN5_HERE/mn5_env.local.sh"

export REPO=${REPO:-/gpfs/projects/CHANGE_PROJECT/tiny-llm}
export DATA_ROOT=${DATA_ROOT:-/gpfs/projects/CHANGE_PROJECT/run2}
export TINY_LLM_ROOT=$REPO
export TINY_LLM_ENV_INIT=/dev/null

# sbatch reads these as its own options, so the sbatch scripts carry no account or queue. acc_ehpc is the QOS that holds
# a 71:50 job; acc_debug cannot and leaves such a job pending for ever
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-CHANGE_ACCOUNT}
export SBATCH_QOS=${SBATCH_QOS:-acc_ehpc}
export SBATCH_PARTITION=${SBATCH_PARTITION:-acc}

VENV=${VENV:-}                                   # path of the venv, empty = already active (no container)
[ -n "$VENV" ] && source "$VENV/bin/activate"

# the container: when SIF is set every python line of the jobs runs inside it, the repo and the data bind mounted
# (scripts/slurm/container/, no InfiniBand binds needed). empty = the venv above.
export SIF=${SIF:-}

# compute nodes may have no internet: nothing may reach the Hub. ONLINE=1 lifts this for the filler download.
if [ "${ONLINE:-0}" != 1 ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
# models and the tokenizer the jobs need (bge, the DeepSeek tokenizer) must already sit in these, fetched
# on a login node with ONLINE=1:
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_home}

[ -n "$SLURM_CPU_BIND" ] && export SLURM_CPU_BIND   # set to none when srun fails to bind the 20 cpus per task
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1

# BIND and CTR are rebuilt here from this shell's own environment, never inherited (--export=ALL carries the login
# shell's values, and a login shell has no $TMPDIR of the job: reads.sbatch copies a save there and the container
# could not see it). not exported. BIND_EXTRA (local overlay) adds site mounts.
source "$_MN5_HERE/lib.sh"
BIND=$REPO,$DATA_ROOT${TMPDIR:+,$TMPDIR},$HF_HOME${BIND_EXTRA:+,$BIND_EXTRA}
CTR_FAIL=""
CTR=$(container_prefix) || { CTR=""; CTR_FAIL=1; }
if [ -n "$CTR" ]; then CTR="$CTR "; fi
# a job script that finds CTR_FAIL set (SIF named, no singularity or apptainer) exits 2 instead of running host python
if [ -n "$CTR_FAIL" ]; then echo "mn5_env: CTR_FAIL set: SIF=$SIF is named but no container runtime was found; every job script will refuse to run" >&2; fi
# this file is sourced: it must return 0 whatever CTR holds, or a caller chaining `source mn5_env.sh && next` silently
# skips next (the supervisor's actions did)
true
