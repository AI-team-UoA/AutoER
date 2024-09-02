#!/bin/bash

datasets=("optuna" "gridsearch" "all")
ds=("D1" "D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" "D10")

mkdir -p ./automl/logs

for dataset in "${datasets[@]}"; do
  for d in "${ds[@]}"; do

    command="python -u regression_with_automl.py --trials $dataset --d $d "
    log_file="./automl/logs/${dataset}_${d}.log"
    
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$start_time] Starting process for dataset: $dataset, d: $d" | tee -a $log_file

    echo "Running: nohup $command > $log_file 2>&1 &"
    nohup $command >> $log_file 2>&1 &
    sleep 120

  done
  wait
done

echo "All tasks completed."
