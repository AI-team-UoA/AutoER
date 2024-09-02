#!/bin/bash

datasets=("optuna" "gridsearch" "all")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF" "XGB" "SVR")
regressors=("XGB")

mkdir -p ./sklearn

for regressor in "${regressors[@]}"; do
  for dataset in "${datasets[@]}"; do
    folder="./sklearn/logs/${regressor}"
    mkdir -p "$folder"

    log_file="./sklearn/logs/${regressor}/${dataset}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: dataset=$dataset, regressor=$regressor" | tee -a "$log_file"

    echo "Running: nohup python -u regression_with_sklearn.py --dataset $dataset --regressor $regressor > $log_file 2>&1 &"
    nohup python -u regression_with_sklearn.py --trials "$dataset" --regressor "$regressor" > "$log_file" 2>&1 &
    # wait
    sleep 10
  done
  # wait
  sleep 10
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: regressor=$regressor" | tee -a "$log_file"
done

echo "All tasks completed."
