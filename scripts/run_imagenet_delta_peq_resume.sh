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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(dirname -- "${SCRIPT_DIR}")"
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_ROOT}}}"
cd "${PROJECT_ROOT}"

if [[ ! -f "$1" ]]; then
    echo "Checkpoint not found: $1" >&2
    exit 1
fi

export RESUME="$1"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PROGRESS="${PROGRESS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHON_BIN="${PYTHON_BIN:-/cis/home/cyang140/.conda/envs/peq-fla/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# Triton caches compiled shared objects. Since the home directory is shared
# across nodes with different glibc versions, isolate the persistent cache by
# hostname and glibc ABI to prevent loading an incompatible cuda_utils.so.
GLIBC_VERSION="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
GLIBC_VERSION="${GLIBC_VERSION#glibc }"
GLIBC_VERSION="${GLIBC_VERSION:-unknown}"
TRITON_CACHE_NODE="${HOSTNAME:-unknown-host}"
TRITON_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/triton"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_CACHE_ROOT}/${TRITON_CACHE_NODE}-glibc${GLIBC_VERSION}}"
mkdir -p -- "${TRITON_CACHE_DIR}"

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    "${PYTHON_BIN}" -c 'import os, torch; count = torch.cuda.device_count(); expected = int(os.environ["GPUS_PER_NODE"]); assert torch.cuda.is_available() and count >= expected, f"CUDA preflight failed: available={torch.cuda.is_available()} count={count} expected={expected}"; torch.ones(1, device="cuda").add_(1); torch.cuda.synchronize(); print(f"cuda_preflight=ok torch={torch.__version__} build_cuda={torch.version.cuda} visible_gpus={[torch.cuda.get_device_name(i) for i in range(count)]}")'
fi

echo "stage=resume checkpoint=${RESUME}"
echo "node=${SLURMD_NODENAME:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-none}"
echo "python_bin=${PYTHON_BIN}"
echo "triton_cache_dir=${TRITON_CACHE_DIR}"
echo "model and training config will be restored from checkpoint"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --local_addr=127.0.0.1 \
        --nnodes=1 \
        --nproc_per_node="${GPUS_PER_NODE}" \
        delta_peq.py
else
    "${PYTHON_BIN}" -u delta_peq.py
fi
