#!/bin/bash
# nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_OPTUNA_RFR.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_OPTUNA_RFR.log 2>&1 & 
# sleep 10
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_GRIDSEARCH_RF.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_GRIDSEARCH_RF.log 2>&1 & 
wait
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_ALL_RFR.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_ALL_RFR.log 2>&1 & 