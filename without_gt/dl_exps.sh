#!/bin/bash

datasets=("optuna" "gridsearch" "all")

mkdir -p ./dl

for dataset in "${datasets[@]}"; do
  command="python -u regression_with_dl.py --trials $dataset"
  log_file="./dl/logs/${dataset}.log"
  
  echo "Running: nohup $command > $log_file 2>&1 &"
  nohup $command > $log_file 2>&1 &
  # wait
done

echo "All tasks completed."
