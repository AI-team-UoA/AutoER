# pyJedAI Auto Configuration
Auto Configuration experiments for pyJedAI

# Problem 1: **With** Ground-Truth file

## Build

Create a conda env 3.10, pip install optuna and pyjedai.

## Execution

Go to `/with_gt/scripts/` and run 

```
nohup ./run_exps.sh 2>&1 & 
```

# Problem 2: **Without** Ground-Truth file

## Classic Regressors

### AutoML Experiments

To run one experiment:
```
python regression_with_automl.py --dataset gridsearch
```

To run all one-by-one:
```
nohup ./automl_exps.sh > ./final/automl/automl_exps.log 2>&1 &
```

### Sklearn Experiments

To run one experiment:
```
python regression_with_sklearn.py --dataset optuna  --regressor LASSO
```

To run all one-by-one:
```
nohup ./sklearn_exps.sh > ./final/sklearn/sklearn_exps.log 2>&1 &
```

## DL

### LinearNN Experiments

To run one experiment:
```
python regression_with_dl.py --trials optuna
```

To run all one-by-one:
```
nohup ./dl_exps.sh > ./final/dl/dl_exps.log 2>&1 &
```

# Resources

| Spec    | Exp. P1                                  | Exp. P2                                                            |
|---------|------------------------------------------|--------------------------------------------------------------------|
| OS      | Ubuntu 22.04 jammy                       | Ubuntu 22.04 jammy                                                 |
| Kernel  | x86_64 Linux 6.2.0-36-generic            | x86_64 Linux 6.5.0-18-generic                                      |
| CPU     | Intel Core i7-9700K @ 8x 4.9GHz [46.0°C] | Intel Xeon E5-4603 v2 @ 32x 2.2GHz [31.0°C]                        |
| GPU     | NVIDIA GeForce RTX 2080 Ti               | Matrox Electronics Systems Ltd. G200eR2                            |
| RAM     | 6622MiB / 64228MiB                       | 4381MiB / 128831MiB                                                |
