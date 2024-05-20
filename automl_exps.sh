#!/bin/bash

# Define the datasets
datasets=("all" "optuna" "gridsearch")

# Create the output directory if it doesn't exist
mkdir -p ./final/automl

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
  # Define the command and log file
  command="python -u regression_with_automl.py --trials $dataset"
  log_file="./final/automl/logs/${dataset}.log"
  
  # Run the command and wait for it to complete
  echo "Running: nohup $command > $log_file 2>&1 &"
  nohup $command > $log_file 2>&1
  wait
done

echo "All tasks completed."
