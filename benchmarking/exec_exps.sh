#!/bin/bash
D=$1

if [ -z "$D" ]; then
  echo "Error: No dataset provided. Please provide a dataset name."
  echo "Usage: $0 <dataset_name>"
  exit 1
fi


nohup python python create_test_trials.csv --data $D  > ${D}_trials.log 2>&1 &
wait
python evaluate.py --data $D
wait
# nohup python python predict_on_dbpedia_data.py --testdata $D  > ${D}.log 2>&1 &
# wait