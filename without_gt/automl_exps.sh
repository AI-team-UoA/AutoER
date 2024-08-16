#!/bin/bash

datasets=("optuna" "gridsearch" "all")

mkdir -p ./automl

for dataset in "${datasets[@]}"; do
  command="python -u regression_with_automl.py --trials $dataset"
  log_file="./automl/logs/${dataset}.log"
  
  echo "Running: nohup $command > $log_file 2>&1 &"
  nohup $command > $log_file 2>&1 &
  # wait
done

echo "All tasks completed."
