#!/usr/bin/env bash
#SBATCH --job-name=peq-pre-resume
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

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 /path/to/checkpoint_latest.pt" >&2
    exit 1
fi

cd /weka/scratch/bhattad/chaohaoyang/new_Archive
mkdir -p wandb/pretrain-resume

if [[ ! -f "$1" ]]; then
    echo "Checkpoint not found: $1" >&2
    exit 1
fi
if [[ ! -d /weka/scratch/bhattad/datasets/imagenet ]]; then
    echo "ImageNet directory not found: /weka/scratch/bhattad/datasets/imagenet" >&2
    exit 1
fi

echo "stage=pretrain-resume checkpoint=$1 output-base=$(dirname "$1")"
conda run --no-capture-output -n peq python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=gpu \
    imagenet_peq_timm_pretrain.py \
    --data-root /weka/scratch/bhattad/datasets/imagenet \
    --output-dir "$(dirname "$1")" \
    --resume "$1" \
    --workers 16 \
    --wandb-dir wandb/pretrain-resume
