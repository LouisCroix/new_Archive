#!/usr/bin/env bash
#SBATCH --job-name=convnext-loop-eval
#SBATCH --account=abhatt40_viztac
#SBATCH --qos=jhu
#SBATCH --partition=h200,h100,a100,l40s
#SBATCH --exclude=gh102
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --comment=accept_cost
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname -- "${SCRIPT_DIR}")}}"
PYTHON_BIN="${PYTHON_BIN:-/home/jhu/cyang140/.conda/envs/peq-fla/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
DATA_ROOT="${DATA_ROOT:-/home/jhu/cyang140/scratch_abhatt40/cyang140/datasets/imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
BS_PER_GPU="${BS_PER_GPU:-256}"
WORKERS="${WORKERS:-4}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
LIMIT_VAL="${LIMIT_VAL:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "CHECKPOINT must point to an official recurrent ConvNeXt checkpoint" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/val" ]]; then
    echo "ImageNet validation directory not found: ${DATA_ROOT}/val" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer, got ${GPUS_PER_NODE}" >&2
    exit 1
fi
if ! [[ "${BS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BS_PER_GPU must be a positive integer, got ${BS_PER_GPU}" >&2
    exit 1
fi
if ! [[ "${WORKERS}" =~ ^[0-9]+$ && "${LIMIT_VAL}" =~ ^[0-9]+$ ]]; then
    echo "WORKERS and LIMIT_VAL must be non-negative integers" >&2
    exit 1
fi
if [[ "${AMP}" != "0" && "${AMP}" != "1" ]]; then
    echo "AMP must be 0 or 1, got ${AMP}" >&2
    exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" && "${AMP_DTYPE}" != "float16" ]]; then
    echo "AMP_DTYPE must be bfloat16 or float16, got ${AMP_DTYPE}" >&2
    exit 1
fi
if [[ "${OVERWRITE}" != "0" && "${OVERWRITE}" != "1" ]]; then
    echo "OVERWRITE must be 0 or 1, got ${OVERWRITE}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

EVAL_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --data-root "${DATA_ROOT}"
    --batch-size "${BS_PER_GPU}"
    --workers "${WORKERS}"
    --limit-val "${LIMIT_VAL}"
    --device cuda
    --dist-backend nccl
    --amp-dtype "${AMP_DTYPE}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
    EVAL_ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${AMP}" == "1" ]]; then
    EVAL_ARGS+=(--amp)
else
    EVAL_ARGS+=(--no-amp)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    EVAL_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node "${GPUS_PER_NODE}" \
    imagenet_recurrent_cnn_loop_eval.py \
    "${EVAL_ARGS[@]}"
