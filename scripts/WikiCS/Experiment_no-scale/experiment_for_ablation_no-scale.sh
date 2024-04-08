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
config_file="$home/configs/WikiCS_node_classification_MoNet_NAPE-ns_100k.json"
exp_name="eta"
# "pos_enc_name": "betaNAPE_WikiCS.pt"

function run_experiments {

  eta=$1
  config_file=$2
  exp_name=$3
  dataset=$(jq '.dataset' $config_file)

  # To solve this problem, I had to create seperate config files for each call.
  # It appears the tmux holds on to a resource once requested.
  config_file_mod="$home/scripts/WikiCS/Experiment_no-scale/modified_config_$eta.json"

  # Read the original JSON file.
  original_data=$(cat $config_file)
  # Replace the entries of dictionary for key "save_file", and change scale in net_params dictionary
  new_data=$(echo "$original_data" | jq --arg eta "$eta" '.net_params.scale = $eta | .save_file.train_file = "train_accuracy\($eta).csv" | .save_file.test_file = "test_accuracy_\($eta).csv"')
  # Rename folder in save_file dictionary
  new_data2=$(echo "$new_data" | jq --arg experiment $exp_name '.save_file.folder = "zz_WikiCS_acc--no-scale-NAPE_\($experiment)"')
  # Save the modified JSON data to file_new.json
  echo "$new_data2" > $config_file_mod

  # Run the experiments
  tmux send-keys -t benchmark_WikiCS-ns "
  python $code --dataset $dataset --gpu_id $id1 --seed $seed0 --config $config_file_mod &
  sleep 10
  python $code --dataset $dataset --gpu_id $id1 --seed $seed1 --config $config_file_mod &
  sleep 10
  python $code --dataset $dataset --gpu_id $id2 --seed $seed2 --config $config_file_mod &
  sleep 10
  python $code --dataset $dataset --gpu_id $id2 --seed $seed3 --config $config_file_mod &
  wait" C-m

  # Compute central Statistics
  code_avg=zz_average_of_mul_simulation.py
  folder_name=$(jq '.save_file.folder' $config_file_mod)
  train_filename=$(jq '.save_file.train_file' $config_file_mod)
  test_filename=$(jq '.save_file.test_file' $config_file_mod)

  tmux send-keys -t benchmark_WikiCS-ns "
  python $code_avg --dataset $dataset --folder $folder_name \
  --best_train_filename $train_filename --best_test_filename $test_filename \
  --exp $exp_name -s
  wait" C-m
}


# Create a new tmux session
tmux new -s benchmark_WikiCS-ns -d

# Send commands to the tmux session
tmux send-keys -t benchmark_WikiCS-ns "cd codes/benchmarking-gnns" C-m
tmux send-keys -t benchmark_WikiCS-ns "conda activate yinka_env" C-m

etas=(1e3 3e3 5e3 7e3 9e3 1e4 5e4 8e4 10e4 11e4 12e4 5e5)

# Run the experiments for each value of eta
for eta in "${etas[@]}"; do
  run_experiments $eta $config_file $exp_name
done

# Get the plots for the ablation
dataset=$(jq '.dataset' $config_file)
code_avg=zz_average_of_mul_simulation.py
folder_name="$(jq '.save_file.folder' $config_file)_$exp_name"

echo "Getting ablation plots!"
tmux send-keys -t benchmark_WikiCS-ns "
python $code_avg --dataset $dataset --folder $folder_name --exp $exp_name --plot -c
wait" C-m

# remove unused file to avoid redundancy
for eta in "${etas[@]}"; do
  tmux send-keys -t benchmark_WikiCS-ns "rm $home/scripts/WikiCS/Experiment_no-scale/modified_config_$eta.json" C-m
done

# Kill the tmux session
# tmux send-keys "tmux kill-session -t benchmark_WikiCS-ns" C-m
