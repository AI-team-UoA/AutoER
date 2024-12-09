nohup python -u eteer_evaluate_ind_regressors.py --config ./configs/RFR/D1D10_DBPEDIA_OPTUNA_RFR.json > ./logs/D1D10_DBPEDIA_OPTUNA_RFR.log 2>&1 & 
echo "Process started for trials type: OPTUNA"
wait
nohup python -u eteer_evaluate_ind_regressors.py --config ./configs/RFR/D1D10_DBPEDIA_GRIDSEARCH_RFR.json > ./logs/D1D10_DBPEDIA_GRIDSEARCH_RFR.log 2>&1 &
echo "Process started for trials type: GRIDSEARCH"
wait
nohup python -u eteer_evaluate_ind_regressors.py --config ./configs/RFR/D1D10_DBPEDIA_ALL_RFR.json > ./logs/D1D10_DBPEDIA_ALL_RFR.log 2>&1 &
echo "Process started for trials type: ALL"
wait