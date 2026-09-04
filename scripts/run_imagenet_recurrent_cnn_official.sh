#!/usr/bin/env bash
#SBATCH --job-name=convnext-official-300
#SBATCH --partition=h100,a100,nvl,l40s
#SBATCH --exclude=h04,h10,n06,n15,l06
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname -- "${SCRIPT_DIR}")}}"
PYTHON_BIN="${PYTHON_BIN:-/cis/home/cyang140/.conda/envs/peq-fla/bin/python}"
DATA_ROOT="${DATA_ROOT:-/cis/project/peq_project/imagenet-1k}"

V="${V:-2}"
ARR1="${ARR1:-1,1,2,0}"
ARR2="${ARR2:-3,3,6,0}"
REG_MODE="${REG_MODE:-0,0,0,0}"
N_REG="${N_REG:-8,8,8,8}"
DELTA_MODE="${DELTA_MODE:-0}"
REG_HEAD="${REG_HEAD:-1}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.1}"
EPOCHS="${EPOCHS:-300}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
BS_PER_GPU="${BS_PER_GPU:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4096}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SAVE_EVERY="${SAVE_EVERY:-25}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-recurrent-convnext-imagenet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_DIR="${WANDB_DIR:-wandb/recurrent-convnext-official}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

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
if (( GPUS_PER_NODE < 1 || BS_PER_GPU < 1 || GLOBAL_BATCH_SIZE < 1 )); then
    echo "GPUS_PER_NODE, BS_PER_GPU, and GLOBAL_BATCH_SIZE must be positive" >&2
    exit 1
fi
if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPOCHS must be a positive integer, got ${EPOCHS}" >&2
    exit 1
fi
if ! [[ "${WARMUP_EPOCHS}" =~ ^[0-9]+$ ]] || (( 10#${WARMUP_EPOCHS} > 10#${EPOCHS} )); then
    echo "WARMUP_EPOCHS must be an integer in [0, EPOCHS], got ${WARMUP_EPOCHS}" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE != 4096 )); then
    echo "The strict official recipe requires GLOBAL_BATCH_SIZE=4096" >&2
    exit 1
fi

MICRO_GLOBAL_BATCH=$((GPUS_PER_NODE * BS_PER_GPU))
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}"
else
    if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
        echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by GPUS_PER_NODE*BS_PER_GPU=${MICRO_GLOBAL_BATCH}" >&2
        exit 1
    fi
    GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))
fi
if (( GRAD_ACCUM_STEPS < 1 )); then
    echo "GRAD_ACCUM_STEPS must be positive" >&2
    exit 1
fi
EFFECTIVE_BATCH_SIZE=$((MICRO_GLOBAL_BATCH * GRAD_ACCUM_STEPS))
if (( EFFECTIVE_BATCH_SIZE != GLOBAL_BATCH_SIZE )); then
    echo "Effective batch ${EFFECTIVE_BATCH_SIZE} does not equal requested GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

ARR1_SLUG="${ARR1//,/-}"
ARR2_SLUG="${ARR2//,/-}"
REG_MODE_SLUG="${REG_MODE//,/-}"
N_REG_SLUG="${N_REG//,/-}"
REG_SUFFIX=""
if [[ "${REG_MODE}" != "0,0,0,0" ]]; then
    REG_SUFFIX="_REG-${REG_MODE_SLUG}_NREG-${N_REG_SLUG}"
fi
if [[ "${DELTA_MODE}" == "1" ]]; then
    REG_SUFFIX+="_DELTA1"
fi
if [[ "${REG_HEAD}" == "1" ]]; then
    REG_SUFFIX+="_REGHEAD1"
fi
RUN_SCHEDULE_SLUG="ep${EPOCHS}_warmup${WARMUP_EPOCHS}"
WANDB_NAME="${WANDB_NAME:-convnext-official-ep${EPOCHS}-warmup${WARMUP_EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/imagenet_recurrent_official_convnextV${V}_ARR1-${ARR1_SLUG}_ARR2-${ARR2_SLUG}${REG_SUFFIX}_${RUN_SCHEDULE_SLUG}_gbs${GLOBAL_BATCH_SIZE}_seed${SEED}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ImageNet directory not found: ${DATA_ROOT}" >&2
    exit 1
fi
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "Resume checkpoint not found: ${RESUME}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

TRAIN_ARGS=(
    --data-root "${DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --arr1 "${ARR1}"
    --arr2 "${ARR2}"
    --reg-mode "${REG_MODE}"
    --n-reg "${N_REG}"
    --convnext-version "${V}"
    --drop-path-rate "${DROP_PATH_RATE}"
    --batch-size "${BS_PER_GPU}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --workers "${WORKERS}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --base-lr 4e-3
    --reference-batch-size 4096
    --warmup-lr 1e-6
    --min-lr 1e-6
    --weight-decay 0.05
    --mixup 0.8
    --cutmix 1.0
    --smoothing 0.1
    --reprob 0.25
    --aa rand-m9-mstd0.5-inc1
    --color-jitter 0.4
    --ema-decay 0.9999
    --amp
    --amp-dtype "${AMP_DTYPE}"
    --save-every "${SAVE_EVERY}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-name "${WANDB_NAME}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
)
if [[ "${DELTA_MODE}" == "1" ]]; then
    TRAIN_ARGS+=(--delta-mode)
fi
if [[ "${REG_HEAD}" == "1" ]]; then
    TRAIN_ARGS+=(--reg-head)
fi
if (( 10#${EPOCHS} == 300 && 10#${WARMUP_EPOCHS} == 20 )); then
    TRAIN_ARGS+=(--strict-official-recipe)
else
    TRAIN_ARGS+=(--no-strict-official-recipe)
fi
if [[ -n "${RESUME}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME}")
fi

COMMAND=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --local_addr=127.0.0.1
    --nnodes=1
    --nproc_per_node="${GPUS_PER_NODE}"
    imagenet_recurrent_cnn_official.py
    "${TRAIN_ARGS[@]}"
)

echo "model=convnext V=${V} ARR1=${ARR1} ARR2=${ARR2} REG_MODE=${REG_MODE} N_REG=${N_REG} DELTA_MODE=${DELTA_MODE} REG_HEAD=${REG_HEAD} drop_path_rate=${DROP_PATH_RATE}"
echo "gpus=${GPUS_PER_NODE} batch_per_gpu=${BS_PER_GPU} accum=${GRAD_ACCUM_STEPS} effective_batch_size=${EFFECTIVE_BATCH_SIZE}"
if (( 10#${EPOCHS} == 300 && 10#${WARMUP_EPOCHS} == 20 )); then
    RECIPE_EXACT=1
else
    RECIPE_EXACT=0
fi
echo "epochs=${EPOCHS} warmup_epochs=${WARMUP_EPOCHS} peak_lr=4e-3 min_lr=1e-6 recipe_exact=${RECIPE_EXACT}"
echo "amp_dtype=${AMP_DTYPE} wandb_mode=${WANDB_MODE} output_dir=${OUTPUT_DIR} resume=${RESUME:-none}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p logs "${WANDB_DIR}"

if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    export EXPECTED_GPUS="${GPUS_PER_NODE}"
    "${PYTHON_BIN}" -c 'import os, torch; expected=int(os.environ["EXPECTED_GPUS"]); count=torch.cuda.device_count(); assert torch.cuda.is_available() and count >= expected, f"CUDA preflight failed: available={torch.cuda.is_available()} count={count} expected={expected}"; [torch.empty(1, device=f"cuda:{index}") for index in range(expected)]; torch.cuda.synchronize(); print(f"cuda_preflight=ok torch={torch.__version__} visible_gpus={[torch.cuda.get_device_name(index) for index in range(count)]}")'
fi

"${COMMAND[@]}"
