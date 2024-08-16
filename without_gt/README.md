# Problem 2: **Without** Ground-Truth file

## AutoML Approach

To run one experiment:
```
python regression_with_automl.py --dataset gridsearch
```

To run all one-by-one:
```
nohup ./automl_exps.sh > ./final/automl/automl_exps.log 2>&1 &
```

## Regressors Experiments - SKLEARN & DL Approach

### SKLEARN

To run one experiment:
```
python regression_with_sklearn.py --dataset optuna  --regressor LASSO
```

To run all one-by-one:
```
nohup ./sklearn_exps.sh > ./final/sklearn/sklearn_exps.log 2>&1 &
```

### DL

To run one experiment:
```
python regression_with_dl.py --trials optuna
```

To run all one-by-one:
```
nohup ./dl_exps.sh > ./final/dl/dl_exps.log 2>&1 &
```
