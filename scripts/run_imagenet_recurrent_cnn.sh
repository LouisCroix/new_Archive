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

# ARR1 is the number of unique blocks in each native ConvNeXt stage. ARR2 is
# the number of times the complete stage is repeated with shared parameters.
# V=1 uses the current ConvNeXt block; V=2 uses ConvNeXt V2 with GRN.
export ARR1="${ARR1:-1,1,1,0}"
export ARR2="${ARR2:-3,3,6,0}"
export REG_MODE="${REG_MODE:-0,0,1,0}"
export N_REG="${N_REG:-8,8,64,8}"
export DELTA_MODE="${DELTA_MODE:-0}"
export REG_HEAD="${REG_HEAD:-0}"
export V="${V:-2}"
if [[ "${V}" != "1" && "${V}" != "2" ]]; then
    echo "V must be 1 or 2, got ${V}" >&2
    exit 1
fi
if [[ ! "${ARR1}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "ARR1 must contain exactly four comma-separated non-negative integers" >&2
    exit 1
fi
if [[ ! "${ARR2}" =~ ^[0-9]+(,[0-9]+){3}$ ]]; then
    echo "ARR2 must contain exactly four comma-separated non-negative integers" >&2
    exit 1
fi
if [[ ! "${REG_MODE}" =~ ^[01](,[01]){3}$ ]]; then
    echo "REG_MODE must contain exactly four comma-separated 0/1 values" >&2
    exit 1
fi
if [[ ! "${N_REG}" =~ ^[1-9][0-9]*(,[1-9][0-9]*){3}$ ]]; then
    echo "N_REG must contain exactly four comma-separated positive integers" >&2
    exit 1
fi
if [[ ! "${DELTA_MODE}" =~ ^[01]$ || ! "${REG_HEAD}" =~ ^[01]$ ]]; then
    echo "DELTA_MODE and REG_HEAD must each be 0 or 1" >&2
    exit 1
fi

export DATA_ROOT="${DATA_ROOT:-/cis/project/peq_project/imagenet-1k}"
export IMG="${IMG:-224}"
export RESIZE="${RESIZE:-256}"
export BS="${BS:-512}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export WORKERS="${WORKERS:-4}"
export MAX_LR="${MAX_LR:-5e-4}"
export MIN_LR="${MIN_LR:-1e-6}"
export EPOCHS="${EPOCHS:-100}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export SEEDS="${SEEDS:-0}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
export PROGRESS="${PROGRESS:-0}"
export MEMORY_PROBE="${MEMORY_PROBE:-0}"
export DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-600}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
ARR1_SLUG="${ARR1//,/-}"
ARR2_SLUG="${ARR2//,/-}"
REG_MODE_SLUG="${REG_MODE//,/-}"
N_REG_SLUG="${N_REG//,/-}"
if [[ "${REG_MODE}" == "0,0,0,0" ]]; then
    REG_SUFFIX=""
    EXPERIMENT_VERSION=6
else
    REG_SUFFIX="_REG-${REG_MODE_SLUG}_NREG-${N_REG_SLUG}"
    EXPERIMENT_VERSION=7
fi
if [[ "${DELTA_MODE}" == "1" ]]; then
    REG_SUFFIX+="_DELTA1"
    EXPERIMENT_VERSION=8
fi
if [[ "${REG_HEAD}" == "1" ]]; then
    REG_SUFFIX+="_REGHEAD1"
    EXPERIMENT_VERSION=8
fi
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/imagenet_recurrent_v${EXPERIMENT_VERSION}_convnextV${V}_ARR1-${ARR1_SLUG}_ARR2-${ARR2_SLUG}${REG_SUFFIX}_img${IMG}_epochs${EPOCHS}_BS${BS}_accum${GRAD_ACCUM_STEPS}_lr${MAX_LR}_minlr${MIN_LR}}"
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
echo "model=convnext V=${V} ARR1=${ARR1} ARR2=${ARR2} REG_MODE=${REG_MODE} N_REG=${N_REG} DELTA_MODE=${DELTA_MODE} REG_HEAD=${REG_HEAD} architecture=four_stage_array_tied activation_checkpointing=0"
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
