#!/bin/bash

datasets=("optuna" "gridsearch" "all")
regressors=("LASSO" "RIDGE" "LINEAR" "RF" "XGB")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF" "XGB" "SVR")

mkdir -p ./sklearn

for regressor in "${regressors[@]}"; do
  for dataset in "${datasets[@]}"; do
    log_file="./sklearn/logs/${dataset}_${regressor}.log"
    
    echo "Running: nohup python -u regression_with_sklearn.py --dataset $dataset --regressor $regressor > $log_file 2>&1 &"
    nohup python -u regression_with_sklearn.py --trials "$dataset" --regressor "$regressor" > "$log_file" 2>&1 &
    # wait
  done
  wait
done

echo "All tasks completed."
