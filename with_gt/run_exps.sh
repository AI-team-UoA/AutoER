#!/bin/bash

# Function to get current timestamp
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log_file="./logs/with_gt_exps.log"

echo "Script started at: $(timestamp)" | tee -a $log_file

# Running GRIDSEARCH experiments
echo "Running: GRIDSEARCH experiments" | tee -a $log_file
echo "Start time: $(timestamp)" | tee -a $log_file
nohup python -u ./scripts/autoconf_gridsearch.py > ./logs/gridsearch.out 2>&1
wait
echo "End time: $(timestamp)" | tee -a $log_file

# Running OPTUNA experiments
echo "Running: OPTUNA experiments" | tee -a $log_file
echo "Start time: $(timestamp)" | tee -a $log_file
nohup python -u ./scripts/autoconf_sampling.py --ntrials 200 > ./logs/samplers.out 2>&1
wait
echo "End time: $(timestamp)" | tee -a $log_file

# Concatenating results
echo "Running: Concatenate results" | tee -a $log_file
echo "Start time: $(timestamp)" | tee -a $log_file
python ./scripts/concatenate.py
echo "End time: $(timestamp)" | tee -a $log_file

echo "Script ended at: $(timestamp)" | tee -a $log_file
echo "Done" | tee -a $log_file
