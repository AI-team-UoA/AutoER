#!/bin/bash
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_OPTUNA.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_OPTUNA.log 2>&1 & 
sleep 10
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_GRIDSEARCH.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_GRIDSEARCH.log 2>&1 & 
sleep 10
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_ALL.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_ALL.log 2>&1 & 