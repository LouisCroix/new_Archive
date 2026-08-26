# start tmux
/usr/bin/tmux new -s peq1

# list all tmux sessions
/usr/bin/tmux ls

# reattach to the session
/usr/bin/tmux attach -t peq1

CUDA_VISIBLE_DEVICES=0,1 \
GPUS_PER_NODE=2 \
bash scripts/run_imagenet_delta_peq.sh