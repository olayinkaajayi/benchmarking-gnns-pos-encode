#!/bin/bash


# bash script_tensorboard.sh


log_path=~/codes/benchmarking-gnns/out/WikiCS_node_classification_NAPE/


tmux new -s tensorboard -d
tmux send-keys "conda activate yinka_env" C-m
tmux send-keys "python3 -m tensorboard.main --logdir=$log_path --port 8889" C-m
