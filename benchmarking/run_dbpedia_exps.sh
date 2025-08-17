#!/bin/bash

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

trials_types=("optuna" "gridsearch" "all")
log_file="./logs/"

echo "Script started at: $(timestamp)"
echo "Running: DBPEDIA experiments"
echo "Start time: $(timestamp)"

for t in "${trials_types[@]}";
do
    echo "Starting process for trials type: $t"

    t_capitalized=$(echo $t | tr '[:lower:]' '[:upper:]')

    command="python eteer_evaluate.py --config ./configs/ablation/D1D10_DBPEDIA_${t_capitalized}.json"
    mkdir -p "./logs/ablation/"
    log_file="./logs/ablation/D1D10_DBPEDIA_${t_capitalized}_ablation.log"
    
    if [ -f $log_file ]; then
      rm $log_file
    fi

    echo "Command: $command"
    echo "Log file: $log_file"

    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$start_time] Starting process for trials type: $t"

    echo "Running: nohup $command > $log_file 2>&1 &"
    nohup $command >> $log_file 2>&1 &
    echo "Process started for trials type: $t"
    sleep 10
done
echo "Script ended at: $(timestamp)"
echo "Done"