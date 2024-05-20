# pyJedAI Auto Configuration
Auto Configuration experiments for pyJedAI


## AutoML Experiments

To run one experiment:
```
nohup python -u regression_with_automl.py --dataset gridsearch  > ./final/automl/grid.log  2>&1 &
```

To run all one-by-one:
```
nohup ./automl_exps.sh > ./final/automl/automl_exps.log 2>&1 &
```

## Sklearn Experiments

To run one experiment:
```
python -u regression_with_sklearn.py --dataset optuna  --regressor LASSO
```

To run all one-by-one:
```
nohup ./sklearn_exps.sh > ./final/sklearn/sklearn_exps.log 2>&1 &
```

## DL Experiments

To run one experiment:
```
python regression_with_dl.py --trials optuna
```


