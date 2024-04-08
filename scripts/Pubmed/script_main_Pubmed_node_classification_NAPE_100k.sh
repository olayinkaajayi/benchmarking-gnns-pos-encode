#!/bin/bash


############
# Usage
############

# bash script_main_Pubmed_node_classification_100k.sh



############
# GNNs
############

#MLP
#GCN
#GraphSage
#GAT
#MoNet
#GIN



############
# Pubmed - 4 RUNS
############

# visible device
id=2
seed0=41
# seed1=95
# seed2=12
# seed3=35
code=main_Pubmed_node_classification.py
tmux new -s benchmark_Pubmed -d
# tmux send-keys "source activate benchmark_gnn" C-m
tmux send-keys -t benchmark_Pubmed "cd codes/benchmarking-gnns" C-m
tmux send-keys -t benchmark_Pubmed "conda activate yinka_env" C-m
dataset=Pubmed

####
#### Consider turning off sign-flip later on
####

tmux send-keys -t benchmark_Pubmed "
CUDA_VISIBLE_DEVICES=$id python $code --dataset $dataset --gpu_id $id --seed $seed0 --config 'configs/Pubmed_node_classification_MoNet_NAPE_100k.json'
" C-m
