#!/usr/bin/env bash
#SBATCH --job-name=imagenet_delta_peq
#SBATCH --partition=h100,a100,nvl,l40s
#SBATCH --exclude=h04,h10,n06,n15,l06
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(dirname -- "${SCRIPT_DIR}")"
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_ROOT}}}"
cd "${PROJECT_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/cis/project/peq_project/imagenet-1k}"
export IMG="${IMG:-224}"
export RESIZE="${RESIZE:-256}"
export D="${D:-384}"
export N_REG="${N_REG:-64}"  # 16,16,24,24,32,32,48,48,64,64,64,64 or 64
export DELTAREG="${DELTAREG:-0}"
export ATTN="${ATTN:-sequential}"  # sequential or rats
if [[ "${ATTN}" != "sequential" && "${ATTN}" != "rats" ]]; then
    echo "Unsupported ATTN=${ATTN}; use sequential or rats" >&2
    exit 1
fi
export SDPA_BACKEND="${SDPA_BACKEND:-flash}"  # flash or auto; ordinary softmax attention only
PATCH_ATTN="delta"
STAGE_LAYOUT="${ATTN}"
if [[ "${ATTN}" == "rats" ]]; then
    COMPRESS_ATTN="rats_shared_qkv"
    REFINE_ATTN="rats_identity_registers"
    BROADCAST_ATTN="rats_identity_register_kv"
else
    COMPRESS_ATTN="softmax"
    REFINE_ATTN="softmax"
    BROADCAST_ATTN="softmax"
fi
export DELTA_BACKEND="${DELTA_BACKEND:-fla}"  # fla, chunk, fused_recurrent, auto, or naive
DELTA_BACKEND_LABEL="${DELTA_BACKEND}"
SDPA_BACKEND_LABEL="${SDPA_BACKEND}"
export DELTA_CHUNK_SIZE="${DELTA_CHUNK_SIZE:-64}"  # 16, 32, or 64
export READOUT="${READOUT:-reg}"  # reg, weighted, patch, sum, or concat
export MIDOUT="${MIDOUT:-none}"  # none or untied; untied adds 0.5 * midpoint CE loss
if [[ "${MIDOUT}" != "none" && "${MIDOUT}" != "untied" ]]; then
    echo "Unsupported MIDOUT=${MIDOUT}; use none or untied" >&2
    exit 1
fi
export RMSNORM="${RMSNORM:-0}"
export LAYERSCALE="${LAYERSCALE:-0}"
export LS_INIT="${LS_INIT:-1e-4}"
export BS="${BS:-256}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export MAX_LR="${MAX_LR:-5e-4}"
export MIN_LR="${MIN_LR:-1e-6}"
export WORKERS="${WORKERS:-4}"
export EPOCHS="${EPOCHS:-22}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
export T="${T:-12}"
export SKIPATTN="${SKIPATTN:-2}"  # attention on 1-based iterations divisible by SKIPATTN
if ! [[ "${SKIPATTN}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SKIPATTN=${SKIPATTN} must be a positive integer" >&2
    exit 1
fi
N_REG_LABEL="${N_REG//,/x}"
N_REG_LABEL="${N_REG_LABEL// /}"
N_REG_LABEL="${N_REG_LABEL//\[/}"
N_REG_LABEL="${N_REG_LABEL//\]/}"
export GAMMA_D="${GAMMA_D:-0.5}"
export LREC="${LREC:-0.3}"
export SEEDS="${SEEDS:-0}"
export MODES="${MODES:-tied}"
export AMP="${AMP:-1}"
export PROGRESS="${PROGRESS:-1}"
export MEMORY_PROBE="${MEMORY_PROBE:-0}"
export AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/imagenet_deltareg${DELTAREG}_delta_peq_patch${PATCH_ATTN}_${STAGE_LAYOUT}_refine${REFINE_ATTN}_${DELTA_BACKEND_LABEL}_sdpa${SDPA_BACKEND_LABEL}_c${DELTA_CHUNK_SIZE}_skipattn${SKIPATTN}_readout${READOUT}_midout${MIDOUT}_D${D}_NREG${N_REG_LABEL}_T${T}_img${IMG}_epochs${EPOCHS}_BS${BS}_accum${GRAD_ACCUM_STEPS}_rms${RMSNORM}_LS${LAYERSCALE}_lr${MAX_LR}_minlr${MIN_LR}}"
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

echo "node=${SLURMD_NODENAME:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none}"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-none}"
echo "python_bin=${PYTHON_BIN}"
echo "triton_cache_dir=${TRITON_CACHE_DIR}"
echo "stage_layout=${STAGE_LAYOUT} patch_attn=${PATCH_ATTN} compress_attn=${COMPRESS_ATTN} refine_attn=${REFINE_ATTN} broadcast_attn=${BROADCAST_ATTN}"
echo "delta_backend=${DELTA_BACKEND_LABEL} sdpa_backend=${SDPA_BACKEND_LABEL} delta_chunk_size=${DELTA_CHUNK_SIZE} readout=${READOUT} midout=${MIDOUT}"
echo "n_reg=${N_REG} T=${T} skipattn=${SKIPATTN}"
echo "epochs=${EPOCHS} warmup_epochs=${WARMUP_EPOCHS} grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "memory_probe=${MEMORY_PROBE}"
echo "max_lr=${MAX_LR} min_lr=${MIN_LR}"
echo "output_dir=${OUTPUT_DIR}"

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
