#!/usr/bin/env bash
# build the training image OFF the cluster (MareNostrum 5 has no outgoing internet and no fakeroot). usage:
#   bash scripts/slurm/container/build.sh [apptainer|docker] [OUT]
#   apptainer (default when apptainer or singularity builds are possible here): builds OUT (default run2.sif) from
#             run2.def, prints size and sha256, runs the smoke test inside it
#   docker:   builds the docker image run2:latest from the Dockerfile, saves OUT (default run2.tar), prints size and
#             sha256, runs the smoke test in the image; on the cluster turn the tar into a .sif in an interactive job:
#             module load singularity; singularity build run2.sif docker-archive://run2.tar
# env: BASE_IMAGE (default nvcr.io/nvidia/pytorch:25.06-py3, docker mode only; the .def names its own),
#      REPO_ROOT (the checkout the smoke test mounts, default two levels above this file).
# then copy the file to the project disk: rsync -P run2.sif USER@transfer1.bsc.es:/gpfs/projects/PROJECT/run2.sif
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/../../.." && pwd)}
MODE=${1:-}
RT=$(command -v apptainer || command -v singularity)
if [ -z "$MODE" ]; then
  if [ -n "$RT" ]; then MODE=apptainer; else MODE=docker; fi
fi
report() { echo "size $(du -h "$1" | cut -f1)"; sha256sum "$1"; }
case "$MODE" in
  apptainer)
    [ -n "$RT" ] || { echo "no apptainer or singularity here; use: build.sh docker" >&2; exit 1; }
    OUT=${2:-run2.sif}
    "$RT" build "$OUT" "$HERE/run2.def" || exit 1
    report "$OUT"
    "$RT" exec --bind "$REPO_ROOT:/repo" --pwd /repo "$OUT" python scripts/slurm/container/smoke.py || { echo "smoke test failed" >&2; exit 1; }
    ;;
  docker)
    OUT=${2:-run2.tar}
    docker build -t run2:latest --build-arg BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.06-py3}" -f "$HERE/Dockerfile" "$HERE" || exit 1
    docker save run2:latest -o "$OUT" || exit 1
    report "$OUT"
    docker run --rm -v "$REPO_ROOT:/repo" -w /repo run2:latest python scripts/slurm/container/smoke.py || { echo "smoke test failed" >&2; exit 1; }
    ;;
  *) echo "usage: build.sh [apptainer|docker] [OUT]" >&2; exit 2;;
esac
echo "built $OUT; copy it to the project disk and set SIF in scripts/slurm/mn5_env.sh"
