#!/usr/bin/env bash
#SBATCH --job-name=peq-pretrain
#SBATCH --partition=h100,a100
#SBATCH --exclude=h04,h10,n06,n15,l06
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname -- "${SCRIPT_DIR}")}}"
export DATA_ROOT="${DATA_ROOT:-/cis/project/peq_project/imagenet-1k}"
export PYTHON_BIN="${PYTHON_BIN:-/cis/home/cyang140/.conda/envs/peq-fla/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

cd "${PROJECT_ROOT}"
mkdir -p wandb/pretrain

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ImageNet directory not found: ${DATA_ROOT}" >&2
    exit 1
fi

# Triton caches compiled shared objects. The home directory is shared across
# nodes with different glibc versions, so isolate the cache by host and ABI.
GLIBC_VERSION="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
GLIBC_VERSION="${GLIBC_VERSION#glibc }"
GLIBC_VERSION="${GLIBC_VERSION:-unknown}"
TRITON_CACHE_NODE="${HOSTNAME:-unknown-host}"
TRITON_CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/triton"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TRITON_CACHE_ROOT}/${TRITON_CACHE_NODE}-glibc${GLIBC_VERSION}}"
mkdir -p -- "${TRITON_CACHE_DIR}"

"${PYTHON_BIN}" -c 'import os, torch; count = torch.cuda.device_count(); expected = int(os.environ["GPUS_PER_NODE"]); assert torch.cuda.is_available() and count >= expected, f"CUDA preflight failed: available={torch.cuda.is_available()} count={count} expected={expected}"; torch.ones(1, device="cuda").add_(1); torch.cuda.synchronize(); print(f"cuda_preflight=ok torch={torch.__version__} build_cuda={torch.version.cuda} visible_gpus={[torch.cuda.get_device_name(i) for i in range(count)]}")'

echo "stage=pretrain"
echo "triton_cache_dir=${TRITON_CACHE_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" \
    imagenet_peq_timm_pretrain.py \
    --data-root "${DATA_ROOT}" \
    --output-dir outputs/peq_timm/pretrain \
    --mode tied \
    --seed 0 \
    --lr 1e-3 \
    --batch-size 256 \
    --grad-accum-steps 1 \
    --workers 16 \
    --epochs 400 \
    --weight-decay 0.03 \
    --steps 12 \
    --n-reg 64 \
    --attention sequential \
    --sdpa-backend flash \
    --delta-backend fla \
    --delta-chunk-size 64 \
    --readout reg \
    --wandb-project peq_imagenet_pretrain \
    --wandb-entity "" \
    --wandb-name pretrain \
    --wandb-group "" \
    --wandb-dir wandb/pretrain
