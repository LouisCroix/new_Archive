# start tmux on cpu node
/home/cyang140/.conda/envs/tmux-tools/bin/tmux new -s peq1

salloc \
  --job-name=intr1 \
  --partition=h100,a100 \
  --exclude=h04,h10,n06,n15,l06 \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=3-00:00:00

# if currently cpu node
srun --pty bash -i

# now on gpu node
nvidia-smi
bash scripts/run_imagenet_delta_peq.sbatch

# detach from the session
# ctrl+b then d

# list all tmux sessions
/home/cyang140/.conda/envs/tmux-tools/bin/tmux ls

# reattach to the session
/home/cyang140/.conda/envs/tmux-tools/bin/tmux attach -t peq1