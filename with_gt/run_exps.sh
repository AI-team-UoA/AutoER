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
for d in {1..10}
do
    echo "Running with --d=$d" | tee -a $log_file
    nohup python -u ./scripts/autoconf_gridsearch.py --did $d > ./logs/gridsearch/D${d}.out 2>&1 &
    sleep 60

    if [ $d -eq 5 ]; then
        echo "Waiting after --d=$d" | tee -a $log_file
        wait
    fi
done
echo "End time: $(timestamp)" | tee -a $log_file

echo "Running: OPTUNA experiments" | tee -a $log_file
echo "Start time: $(date)" | tee -a $log_file

for d in {1..10}
do
    echo "Running with --d=$d" | tee -a $log_file
    nohup python -u ./scripts/autoconf_sampling.py --ntrials 100 --d $d > ./logs/samplers_d${d}.out 2>&1 &
    sleep 10

    if [ $d -eq 7 ]; then
        echo "Waiting after --d=$d" | tee -a $log_file
        wait
    fi
done

wait

echo "All OPTUNA experiments completed." | tee -a $log_file
echo "Overall end time: $(date)" | tee -a $log_file

# Concatenating results
echo "Running: Concatenate results" | tee -a $log_file
echo "Start time: $(timestamp)" | tee -a $log_file
python ./scripts/concatenate.py
echo "End time: $(timestamp)" | tee -a $log_file

echo "Script ended at: $(timestamp)" | tee -a $log_file
echo "Done" | tee -a $log_file
