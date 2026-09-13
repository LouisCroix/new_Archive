#!/usr/bin/env bash
#SBATCH --job-name=atto-fcmae-ft
#SBATCH --account=abhatt40_viztac
#SBATCH --qos=jhu
#SBATCH --partition=h200,h100,a100
#SBATCH --exclude=gh102
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --comment=accept_cost
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname -- "${SCRIPT_DIR}")}}"
PYTHON_BIN="${PYTHON_BIN:-/home/jhu/cyang140/.conda/envs/peq-fla/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/jhu/cyang140/scratch_abhatt40/cyang140/datasets/imagenet}"
STAGE_REPEATS="${STAGE_REPEATS:-2,2,2,2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
BS_PER_GPU="${BS_PER_GPU:-256}"
MAX_GLOBAL_BATCH_SIZE="${MAX_GLOBAL_BATCH_SIZE:-1024}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-600}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
BASE_LR="${BASE_LR:-2e-4}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-256}"
MIN_LR="${MIN_LR:-1e-6}"
LAYER_DECAY="${LAYER_DECAY:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.3}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.1}"
REPROB="${REPROB:-0.25}"
MIXUP="${MIXUP:-0.0}"
CUTMIX="${CUTMIX:-0.0}"
SMOOTHING="${SMOOTHING:-0.2}"
AA="${AA:-rand-m9-mstd0.5-inc1}"
EMA_DECAY="${EMA_DECAY:-0.9999}"
AMP_DTYPE="${AMP_DTYPE:-float16}"
SEED="${SEED:-0}"
SAVE_EVERY="${SAVE_EVERY:-25}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
RESUME="${RESUME:-}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-convnextv2-atto-fcmae-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_DIR="${WANDB_DIR:-wandb/convnextv2-atto-fcmae}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

if [[ ! "${STAGE_REPEATS}" =~ ^[1-9][0-9]*(,[1-9][0-9]*){3}$ ]]; then
    echo "STAGE_REPEATS must contain exactly four comma-separated positive integers" >&2
    exit 1
fi
for value_name in GPUS_PER_NODE BS_PER_GPU MAX_GLOBAL_BATCH_SIZE EPOCHS REFERENCE_BATCH_SIZE; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
        exit 1
    fi
done
if ! [[ "${WARMUP_EPOCHS}" =~ ^[0-9]+$ ]] || (( 10#${WARMUP_EPOCHS} > 10#${EPOCHS} )); then
    echo "WARMUP_EPOCHS must be an integer in [0, EPOCHS]" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ImageNet directory not found: ${DATA_ROOT}" >&2
    exit 1
fi
if [[ -n "${PRETRAINED_CHECKPOINT}" && ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Pretrained checkpoint not found: ${PRETRAINED_CHECKPOINT}" >&2
    exit 1
fi
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "Resume checkpoint not found: ${RESUME}" >&2
    exit 1
fi
if [[ -n "${RESUME}" && -n "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "RESUME and PRETRAINED_CHECKPOINT are mutually exclusive" >&2
    exit 1
fi

MICRO_GLOBAL_BATCH_SIZE=$((GPUS_PER_NODE * BS_PER_GPU))
if (( MICRO_GLOBAL_BATCH_SIZE > MAX_GLOBAL_BATCH_SIZE )); then
    echo "GPUS_PER_NODE*BS_PER_GPU=${MICRO_GLOBAL_BATCH_SIZE} exceeds MAX_GLOBAL_BATCH_SIZE=${MAX_GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$((MAX_GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH_SIZE))
EFFECTIVE_BATCH_SIZE=$((MICRO_GLOBAL_BATCH_SIZE * GRAD_ACCUM_STEPS))
PEAK_LR="$(awk -v base_lr="${BASE_LR}" -v batch_size="${EFFECTIVE_BATCH_SIZE}" -v reference="${REFERENCE_BATCH_SIZE}" 'BEGIN { printf "%.10g", base_lr * batch_size / reference }')"
REPEAT_SLUG="${STAGE_REPEATS//,/-}"
WANDB_NAME="${WANDB_NAME:-convnextv2-atto-fcmae-r${REPEAT_SLUG}-ep${EPOCHS}-gbs${EFFECTIVE_BATCH_SIZE}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/convnextv2_atto_fcmae_repeats-${REPEAT_SLUG}_ep${EPOCHS}_gbs${EFFECTIVE_BATCH_SIZE}_lr${PEAK_LR}_seed${SEED}}"

cd "${PROJECT_ROOT}"
export HF_HOME HF_HUB_OFFLINE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

TRAIN_ARGS=(
    --stage-repeats "${STAGE_REPEATS}"
    --data-root "${DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BS_PER_GPU}"
    --max-global-batch-size "${MAX_GLOBAL_BATCH_SIZE}"
    --workers "${WORKERS}"
    --epochs "${EPOCHS}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --base-lr "${BASE_LR}"
    --reference-batch-size "${REFERENCE_BATCH_SIZE}"
    --min-lr "${MIN_LR}"
    --layer-decay "${LAYER_DECAY}"
    --weight-decay "${WEIGHT_DECAY}"
    --drop-path-rate "${DROP_PATH_RATE}"
    --reprob "${REPROB}"
    --mixup "${MIXUP}"
    --cutmix "${CUTMIX}"
    --smoothing "${SMOOTHING}"
    --aa "${AA}"
    --ema-decay "${EMA_DECAY}"
    --amp --amp-dtype "${AMP_DTYPE}"
    --seed "${SEED}"
    --save-every "${SAVE_EVERY}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-name "${WANDB_NAME}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
)
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")
fi
if [[ -n "${RESUME}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME}")
fi

COMMAND=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --local_addr=127.0.0.1
    --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}"
    imagenet_recurrent_convnextv2_atto_fcmae_finetune.py "${TRAIN_ARGS[@]}"
)

echo "model=convnextv2_atto.fcmae stage_repeats=${STAGE_REPEATS}"
echo "gpus=${GPUS_PER_NODE} batch_per_gpu=${BS_PER_GPU} max_global_batch=${MAX_GLOBAL_BATCH_SIZE} accum=${GRAD_ACCUM_STEPS} effective_batch=${EFFECTIVE_BATCH_SIZE}"
echo "epochs=${EPOCHS} warmup=${WARMUP_EPOCHS} base_lr=${BASE_LR} peak_lr=${PEAK_LR} layer_decay=${LAYER_DECAY}"
echo "hf_home=${HF_HOME} offline=${HF_HUB_OFFLINE} pretrained=${PRETRAINED_CHECKPOINT:-timm-cache} resume=${RESUME:-none}"
echo "output_dir=${OUTPUT_DIR} wandb_mode=${WANDB_MODE}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p logs "${WANDB_DIR}" "${HF_HOME}"
if [[ "${REQUIRE_CUDA}" == "1" ]]; then
    export EXPECTED_GPUS="${GPUS_PER_NODE}"
    "${PYTHON_BIN}" -c 'import os, torch; n=int(os.environ["EXPECTED_GPUS"]); assert torch.cuda.is_available() and torch.cuda.device_count() >= n, f"CUDA preflight failed: count={torch.cuda.device_count()} expected={n}"; print(f"cuda_preflight=ok torch={torch.__version__}")'
fi

# Resolve and cache the 13.6 MB safetensors once before DDP ranks start.
if [[ -z "${RESUME}" && -z "${PRETRAINED_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" -c 'import timm; timm.create_model("convnextv2_atto.fcmae", pretrained=True, num_classes=0); print("fcmae_pretrained_cache=ready")'
fi

"${COMMAND[@]}"
