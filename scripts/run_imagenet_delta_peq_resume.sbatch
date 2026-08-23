#!/usr/bin/env bash
#SBATCH --job-name=imagenet-delta-peq-resume
#SBATCH --partition=h100,a100
#SBATCH --exclude=h04,h10,n06,n15,l06
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 /path/to/checkpoint_latest.pt" >&2
    exit 1
fi

DEFAULT_PROJECT_ROOT="/weka/scratch/bhattad/chaohaoyang/new_Archive"
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_ROOT}}}"
cd "${PROJECT_ROOT}"

if [[ ! -f "$1" ]]; then
    echo "Checkpoint not found: $1" >&2
    exit 1
fi

export RESUME="$1"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export PROGRESS="${PROGRESS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHON_BIN="${PYTHON_BIN:-/home/cyang140/.conda/envs/peq-fla/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

echo "stage=resume checkpoint=${RESUME}"
echo "node=${SLURMD_NODENAME:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-none}"
echo "python_bin=${PYTHON_BIN}"
echo "model and training config will be restored from checkpoint"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${GPUS_PER_NODE}" \
        delta_peq.py
else
    "${PYTHON_BIN}" -u delta_peq.py
fi
