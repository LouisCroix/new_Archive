# start tmux
/usr/bin/tmux new -s peq1

# list all tmux sessions
/usr/bin/tmux ls

# reattach to the session
/usr/bin/tmux attach -d -t peq1

# delta run
BS=128 ATTN=sequential CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 bash scripts/run_imagenet_delta_peq.sh

# cnn run
BS=128 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 bash scripts/run_imagenet_recurrent_cnn.sh

# timm pretrain
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 bash scripts/run_peq_timm_pretrain.sh

# timm pretrain resume
CUDA_VISIBLE_DEVICES=0,1 \
GPUS_PER_NODE=2 \
bash scripts/run_peq_timm_pretrain_resume.sh \
/绝对路径/checkpoint_latest.pt

# timm finetune
CUDA_VISIBLE_DEVICES=0,1 \
GPUS_PER_NODE=2 \
bash scripts/run_peq_timm_finetune.sh \
/绝对路径/checkpoint_final.pt