#!/bin/bash

nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_OPTUNA_ablation.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_OPTUNA_ablation.log 2>&1 & 
sleep 10
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_GRIDSEARCH_ablation.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_GRIDSEARCH_ablation.log 2>&1 & 
sleep 10
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_ALL_ablation.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_ALL_ablation.log 2>&1 & 
