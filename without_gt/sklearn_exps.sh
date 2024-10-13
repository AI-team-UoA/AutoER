#!/bin/bash

# datasets=("optuna" "gridsearch" "all")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF")
# regressors=("LASSO" "RIDGE" "LINEAR" "RF" "XGB" "SVR")
# regressors=("XGB")

datasets=("optuna" "gridsearch" "all")
regressors=("LINEAR")

# read input cmd flag --ablation

if [ "$1" == "--ablation" ]; then
  ablation=true
  with_data_features=0
 echo "Running ablation study."
else
  ablation=false
  with_data_features=1
  echo "Running all experiments."
fi

mkdir -p ./sklearn

for regressor in "${regressors[@]}"; do
  for dataset in "${datasets[@]}"; do
    if [ "$ablation" = true ]; then
      log_file="./sklearn/logs/ablation/${regressor}/${dataset}.log"
      mkdir -p "./sklearn/logs/ablation"
      mkdir -p "./sklearn/logs/ablation/${regressor}"
    else
      log_file="./sklearn/logs/${regressor}/${dataset}.log"
      mkdir -p "./sklearn/logs"
      mkdir -p "./sklearn/logs/${regressor}"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: dataset=$dataset, regressor=$regressor" | tee -a "$log_file"

    #  if file like $dataset_$regressor_ablation.db exists, rm and ablation active
    if [ "$ablation" = true ]; then
      if [ -f "./${dataset}_${regressor}_ablation.db"  ]; then
        rm "./${dataset}_${regressor}_ablation.db"
        echo "Removed existing database file: ${dataset}_${regressor}_ablation.db"
      fi
    else
      if [ -f "./${dataset}_${regressor}.db"  ]; then
        rm "./${dataset}_${regressor}.db"
        echo "Removed existing database file: ${dataset}_${regressor}.db"
      fi
    fi


    cmd="nohup python -u regression_with_sklearn.py --trials $dataset --regressor $regressor --with_data_features $with_data_features > $log_file 2>&1 &"

    echo "Running command: $cmd" | tee -a "$log_file"

    # run the command
    eval $cmd
    # nohup python -u regression_with_sklearn.py --trials "$dataset" --regressor "$regressor" > "$log_file" 2>&1 &
    # wait
    sleep 10
  done
  # wait
  sleep 10
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: regressor=$regressor" | tee -a "$log_file"
done

echo "All tasks completed."
