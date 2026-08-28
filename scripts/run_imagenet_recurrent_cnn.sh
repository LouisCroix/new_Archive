#!/usr/bin/env bash
#SBATCH --job-name=imagenet-recurrent-cnn
#SBATCH --partition=h100,a100,nvl,l40s
#SBATCH --exclude=h04,h10,n06,n15,l06
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(dirname -- "${SCRIPT_DIR}")"
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_ROOT}}}"
cd "${PROJECT_ROOT}"

# Architecture selection only. Naive uses native first-stage blocks; pro uses
# native early stages and recurrent third-stage blocks. Widths, kernels, norms,
# expansion ratios, LayerScale, and initialization remain native.
export BLOCK_TYPE="${BLOCK_TYPE:-convnext}"  # resnet or convnext
if [[ "${BLOCK_TYPE}" != "resnet" && "${BLOCK_TYPE}" != "convnext" ]]; then
    echo "Unsupported BLOCK_TYPE=${BLOCK_TYPE}; use resnet or convnext" >&2
    exit 1
fi
export MODE="${MODE:-pro}"  # naive or pro
if [[ "${MODE}" != "naive" && "${MODE}" != "pro" ]]; then
    echo "Unsupported MODE=${MODE}; use naive or pro" >&2
    exit 1
fi
export BLOCK_DEPTH="${BLOCK_DEPTH:-1}"
if [[ ! "${BLOCK_DEPTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLOCK_DEPTH must be a positive integer, got ${BLOCK_DEPTH}" >&2
    exit 1
fi
export T="${T:-12}"

export DATA_ROOT="${DATA_ROOT:-/cis/project/peq_project/imagenet-1k}"
export IMG="${IMG:-224}"
export RESIZE="${RESIZE:-256}"
export EPOCHS="${EPOCHS:-22}"
export BS="${BS:-512}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export WORKERS="${WORKERS:-4}"
export MAX_LR="${MAX_LR:-5e-4}"
export MIN_LR="${MIN_LR:-1e-6}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export SEEDS="${SEEDS:-0}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
export PROGRESS="${PROGRESS:-0}"
export MEMORY_PROBE="${MEMORY_PROBE:-0}"
export DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-120}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/imagenet_recurrent_v3_${MODE}_${BLOCK_TYPE}_depth${BLOCK_DEPTH}_T${T}_img${IMG}_epochs${EPOCHS}_BS${BS}_accum${GRAD_ACCUM_STEPS}_lr${MAX_LR}_minlr${MIN_LR}}"
export PYTHON_BIN="${PYTHON_BIN:-/cis/home/cyang140/.conda/envs/peq-fla/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    "${PYTHON_BIN}" -c 'import os, torch; count=torch.cuda.device_count(); expected=int(os.environ["GPUS_PER_NODE"]); assert torch.cuda.is_available() and count >= expected, "CUDA preflight failed: available={} count={} expected={} CUDA_VISIBLE_DEVICES={}".format(torch.cuda.is_available(), count, expected, os.environ.get("CUDA_VISIBLE_DEVICES")); [torch.empty(1, device=f"cuda:{i}") for i in range(expected)]; torch.cuda.synchronize(); print(f"cuda_preflight=ok torch={torch.__version__} visible_gpus={[torch.cuda.get_device_name(i) for i in range(count)]}")'
fi

echo "node=${SLURMD_NODENAME:-none} job_id=${SLURM_JOB_ID:-none}"
echo "python_bin=${PYTHON_BIN} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none} gpus=${GPUS_PER_NODE}"
echo "block_type=${BLOCK_TYPE} mode=${MODE} block_depth=${BLOCK_DEPTH} T=${T} architecture=native recurrent_norm=v2 tied_cell=1 activation_checkpointing=0"
echo "epochs=${EPOCHS} BS_per_gpu=${BS} accum=${GRAD_ACCUM_STEPS} workers_per_rank=${WORKERS} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "progress=${PROGRESS} dataloader_timeout=${DATALOADER_TIMEOUT}s nccl_trace_buffer=${TORCH_NCCL_TRACE_BUFFER_SIZE}"
echo "max_lr=${MAX_LR} min_lr=${MIN_LR} warmup_epochs=${WARMUP_EPOCHS}"
echo "output_dir=${OUTPUT_DIR}"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
    if [[ -t 1 ]]; then
        stty -ixon
    fi
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --local_addr=127.0.0.1 \
        --nnodes=1 \
        --nproc_per_node="${GPUS_PER_NODE}" \
        recurrent_cnn.py
else
    "${PYTHON_BIN}" -u recurrent_cnn.py
fi
