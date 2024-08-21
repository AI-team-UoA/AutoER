#!/bin/bash

datasets=("optuna" "gridsearch" "all")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF" "XGB")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF" "XGB" "SVR")
regressors=("RF")

mkdir -p ./sklearn

for regressor in "${regressors[@]}"; do
  for dataset in "${datasets[@]}"; do
    log_file="./sklearn/logs/${dataset}_${regressor}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: dataset=$dataset, regressor=$regressor" | tee -a "$log_file"

    echo "Running: nohup python -u regression_with_sklearn.py --dataset $dataset --regressor $regressor > $log_file 2>&1 &"
    nohup python -u regression_with_sklearn.py --trials "$dataset" --regressor "$regressor" > "$log_file" 2>&1 &
    # wait
  done
  wait
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: regressor=$regressor" | tee -a "$log_file"
done

echo "All tasks completed."
