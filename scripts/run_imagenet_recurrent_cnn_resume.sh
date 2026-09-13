#!/usr/bin/env bash
#SBATCH --job-name=imagenet-recurrent-cnn-resume
#SBATCH --account=abhatt40_viztac
#SBATCH --qos=normal
#SBATCH --partition=h200,h100,a100
#SBATCH --exclude=gh102,l40s
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --comment=accept_cost
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 /path/to/checkpoint_latest.pt" >&2
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export PROGRESS="${PROGRESS:-0}"
export DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DELTA_BACKEND="${DELTA_BACKEND:-auto}"
export DELTA_REQUIRE_FLA="${DELTA_REQUIRE_FLA:-1}"
export DELTA_CHUNK_SIZE="${DELTA_CHUNK_SIZE:-64}"
export TORCHRUN_MAX_RESTARTS="${TORCHRUN_MAX_RESTARTS:-2}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export PYTHON_BIN="${PYTHON_BIN:-/home/jhu/cyang140/.conda/envs/peq-fla/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# Keep Triton native caches local to a compatible node/glibc combination.
GLIBC_VERSION="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
GLIBC_VERSION="${GLIBC_VERSION#glibc }"
GLIBC_VERSION="${GLIBC_VERSION:-unknown}"
TRITON_CACHE_NODE="${HOSTNAME:-unknown-host}"
TRITON_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/triton"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_CACHE_ROOT}/${TRITON_CACHE_NODE}-glibc${GLIBC_VERSION}}"
mkdir -p -- "${TRITON_CACHE_DIR}"

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    "${PYTHON_BIN}" -c 'import os, torch; count=torch.cuda.device_count(); expected=int(os.environ["GPUS_PER_NODE"]); assert torch.cuda.is_available() and count >= expected, f"CUDA preflight failed: available={torch.cuda.is_available()} count={count} expected={expected}"; [torch.empty(1, device=f"cuda:{i}") for i in range(expected)]; torch.cuda.synchronize(); print(f"cuda_preflight=ok torch={torch.__version__} visible_gpus={[torch.cuda.get_device_name(i) for i in range(count)]}")'
fi

echo "stage=resume checkpoint=${RESUME}"
echo "python_bin=${PYTHON_BIN} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none} gpus=${GPUS_PER_NODE}"
echo "triton_cache_dir=${TRITON_CACHE_DIR}"
echo "delta_backend=${DELTA_BACKEND} delta_require_fla=${DELTA_REQUIRE_FLA} delta_chunk_size=${DELTA_CHUNK_SIZE}"
echo "progress=${PROGRESS} dataloader_timeout=${DATALOADER_TIMEOUT}s max_restarts=${TORCHRUN_MAX_RESTARTS}"
echo "architecture and training config will be restored from checkpoint"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
    if [[ -t 1 ]]; then
        stty -ixon
    fi
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --local_addr=127.0.0.1 \
        --nnodes=1 \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --max_restarts="${TORCHRUN_MAX_RESTARTS}" \
        recurrent_cnn.py
else
    "${PYTHON_BIN}" -u recurrent_cnn.py
fi
