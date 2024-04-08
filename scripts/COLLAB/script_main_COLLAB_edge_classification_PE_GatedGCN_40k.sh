#!/bin/bash

# check :
# bash script.sh
# tmux attach -t script_tsp
# tmux detach
# pkill python

# bash script_main_COLLAB_edge_classification_PE_GatedGCN_40k.sh




############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT


############
# OGBL-COLLAB - 4 RUNS
############

# visible device
id=2
seed0=41
seed1=42
seed2=9
seed3=23
code=main_COLLAB_edge_classification.py
dataset=OGBL-COLLAB
tmux new -s benchmark_COLLAB_edge_classification -d
tmux send-keys -t benchmark_COLLAB_edge_classification "cd codes/benchmarking-gnns" C-m
tmux send-keys -t benchmark_COLLAB_edge_classification "conda activate yinka_env" C-m

tmux send-keys -t benchmark_COLLAB_edge_classification "
CUDA_VISIBLE_DEVICES=$id python $code --dataset $dataset --gpu_id $id --seed $seed0 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json'
wait" C-m

tmux send-keys -t benchmark_COLLAB_edge_classification "
CUDA_VISIBLE_DEVICES=$id python $code --dataset $dataset --gpu_id $id --seed $seed1 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json'
wait" C-m

tmux send-keys -t benchmark_COLLAB_edge_classification "
CUDA_VISIBLE_DEVICES=$id python $code --dataset $dataset --gpu_id $id --seed $seed2 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json'
wait" C-m

tmux send-keys -t benchmark_COLLAB_edge_classification "
CUDA_VISIBLE_DEVICES=$id python $code --dataset $dataset --gpu_id $id --seed $seed3 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json'
wait" C-m

# Compute central Statistics
code_avg=zz_average_of_mul_simulation.py
folder_name=zz_ogb_collab_acc
train_filename=train_@50Hits.csv
test_filename=test_@50Hits.csv
tmux send-keys -t benchmark_COLLAB_edge_classification "
CUDA_VISIBLE_DEVICES=$id python $code_avg --dataset $dataset --folder $folder_name \
--best_train_filename $train_filename --best_test_filename $test_filename
wait" C-m











# tmux send-keys "source activate benchmark_gnn" C-m
# tmux send-keys "
# python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json' &
# python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json' &
# python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json' &
# python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json' &
# wait" C-m
# tmux send-keys "tmux kill-session -t benchmark_COLLAB_edge_classification" C-m
