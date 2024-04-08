#!/bin/bash

id0=0
id1=1
id2=2
seed0=41
seed1=42
seed2=9
seed3=23
code=main_WikiCS_node_classification.py
home=/dcs/pg20/u2034358/codes/benchmarking-gnns
config_file_lap="$home/configs/WikiCS_node_classification_MoNet_PE_100k.json"
config_file_no_PE="$home/configs/WikiCS_node_classification_MoNet_100k.json"


function run_experiments {

  config_file=$1
  dataset=$(jq '.dataset' $config_file)

  # Run the experiments for each config file: with Lap Eig PE, and no PE
  tmux send-keys -t benchmark_COLLAB_edge_classification_none_NAPE "
  python $code --dataset $dataset --gpu_id $id1 --seed $seed0 --config $config_file &
  sleep 10
  python $code --dataset $dataset --gpu_id $id1 --seed $seed1 --config $config_file &
  sleep 10
  python $code --dataset $dataset --gpu_id $id2 --seed $seed2 --config $config_file &
  sleep 10
  python $code --dataset $dataset --gpu_id $id2 --seed $seed3 --config $config_file &
  wait" C-m

  # Compute central Statistics
  code_avg=zz_average_of_mul_simulation.py
  folder_name=$(jq '.save_file.folder' $config_file)
  train_filename=$(jq '.save_file.train_file' $config_file)
  test_filename=$(jq '.save_file.test_file' $config_file)
  exp_name="none_NAPE"

  tmux send-keys -t benchmark_COLLAB_edge_classification_none_NAPE "
  python $code_avg --dataset $dataset --folder $folder_name \
  --best_train_filename $train_filename --best_test_filename $test_filename \
  --exp $exp_name -s
  wait" C-m
}


# Create a new tmux session
tmux new -s benchmark_COLLAB_edge_classification_none_NAPE -d

# Send commands to the tmux session
tmux send-keys -t benchmark_COLLAB_edge_classification_none_NAPE "cd codes/benchmarking-gnns" C-m
tmux send-keys -t benchmark_COLLAB_edge_classification_none_NAPE "conda activate yinka_env" C-m

configs=($config_file_no_PE $config_file_lap)

# Run the experiments for each value of eta
for config in "${configs[@]}"; do
  run_experiments $config
done

echo "All commands pushed!"

# Kill the tmux session
# tmux send-keys "tmux kill-session -t benchmark_COLLAB_edge_classification_none_NAPE" C-m
