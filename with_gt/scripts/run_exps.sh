echo "Running: GRIDSEARCH experiments"
nohup python -u autoconf_gridsearch.py > ../logs/gridsearch.out 2>&1
wait
echo "Running: OPTUNA experiments"
nohup python -u autoconf_sampling.py --ntrials 200 > ../logs/samplers.out 2>&1
wait
python concatenate.py
echo "Done"
