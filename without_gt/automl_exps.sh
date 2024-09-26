#!/bin/bash

trials_types=("gridsearch" "all")
hidden_datasets=("D1" "D2" "D3" "D4" "D5" "D6" "D7" "D8" "D9" "D10")

# read from command line a config file
config_file=$1

echo "JSON Config file: $config_file"

specs=$(basename $config_file .json)

mkdir -p ./automl/logs/$specs

for t in "${trials_types[@]}"; do
  for d in "${hidden_datasets[@]}"; do

    command="python -u regression_with_automl.py --trials_type $t --hidden_dataset $d --config $config_file"
    mkdir -p ./automl/logs/$specs/$t
    log_file="./automl/logs/${specs}/${t}/${d}.log"
    
    if [ -f $log_file ]; then
      rm $log_file
    fi

    echo "Command: $command"
    echo "Log file: $log_file"

    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$start_time] Starting process for trials type: $t, d: $d" | tee -a $log_file

    echo "Running: nohup $command > $log_file 2>&1 &"
    nohup $command >> $log_file 2>&1 &
    sleep 10

  done
done

echo "All tasks completed."
