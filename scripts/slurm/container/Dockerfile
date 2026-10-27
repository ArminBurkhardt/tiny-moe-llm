# the same image as run2.def, for a machine that has docker and no apptainer (build.sh docker). the output is a tar to
# copy to the cluster and turn into a .sif there: singularity build run2.sif docker-archive://run2.tar
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}
RUN pip install --no-cache-dir \
    "transformers>=5.5.4" "numpy>=1.21.0" "pandas>=3.0.0" "pyarrow>=14.0.0" "huggingface_hub>=0.27.0" \
    "zstandard>=0.23.0" "matplotlib>=3.10.0" "accelerate>=1.13.0"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
