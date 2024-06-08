#!/bin/bash

# Define the datasets and regressors
datasets=("optuna" "gridsearch" "all")
regressors=("LASSO" "RIDGE" "LINEAR" "RF")
# regressors=("XGB" "SVR")

# Create the output directory if it doesn't exist
mkdir -p ./sklearn

# Iterate over each combination of dataset and regressor
for dataset in "${datasets[@]}"; do
  for regressor in "${regressors[@]}"; do
    # Define the log file name
    log_file="./sklearn/logs/${dataset}_${regressor}.log"
    
    # Run the command and wait for it to complete
    echo "Running: nohup python -u regression_with_sklearn.py --dataset $dataset --regressor $regressor > $log_file 2>&1 &"
    nohup python -u regression_with_sklearn.py --trials "$dataset" --regressor "$regressor" > "$log_file" 2>&1
    wait
  done
done

echo "All tasks completed."
