#!/bin/bash

datasets=("optuna" "gridsearch" "all")
ds=("D1" "D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" "D10")
# datasets=("gridsearch")


mkdir -p ./automl

for dataset in "${datasets[@]}"; do
  for d in "${ds[@]}"; do

    command="python -u regression_with_automl.py --trials $dataset --d $d "
    log_file="./automl/logs/${dataset}_${d}.log"
    
    echo "Running: nohup $command > $log_file 2>&1 &"
    nohup $command > $log_file 2>&1 &
    sleep 15

  done
  # wait
done

echo "All tasks completed."
