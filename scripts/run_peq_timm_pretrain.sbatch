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

cd /weka/scratch/bhattad/chaohaoyang/new_Archive
mkdir -p wandb/pretrain

if [[ ! -d /weka/scratch/bhattad/datasets/imagenet ]]; then
    echo "ImageNet directory not found: /weka/scratch/bhattad/datasets/imagenet" >&2
    exit 1
fi

echo "stage=pretrain"
conda run --no-capture-output -n peq python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=gpu \
    imagenet_peq_timm_pretrain.py \
    --data-root /weka/scratch/bhattad/datasets/imagenet \
    --output-dir outputs/peq_timm/pretrain \
    --mode tied \
    --seed 0 \
    --lr 4e-3 \
    --batch-size 512 \
    --grad-accum-steps 4 \
    --workers 16 \
    --epochs 400 \
    --weight-decay 0.03 \
    --drop-path 0.0 \
    --steps 12 \
    --n-reg 64 \
    --lora-rank 0 \
    --lora-alpha 16 \
    --lora-compose cumul \
    --wandb-project peq_imagenet_pretrain \
    --wandb-entity "" \
    --wandb-name pretrain_tied \
    --wandb-group "" \
    --wandb-dir wandb/pretrain
