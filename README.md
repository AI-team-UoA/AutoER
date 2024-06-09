# pyJedAI Auto Configuration
Auto Configuration experiments for pyJedAI

# 1st case; **With** Ground-Truth file

All exps run at 
```python
konstantinos@snorlax
OS: Ubuntu 22.04 jammy
Kernel: x86_64 Linux 6.2.0-36-generic
Uptime: 112d 32m
Packages: 1630
Shell: zsh 5.8.1
Disk: 946G / 2,3T (44%)
CPU: Intel Core i7-9700K @ 8x 4,9GHz [46.0°C]
GPU: NVIDIA GeForce RTX 2080 Ti
RAM: 6622MiB / 64228MiB
```

# 2nd case; **Without** Ground-Truth file

## Classic Regressors

### Resources
```python
nikoletos@pyravlos5
OS: Ubuntu 22.04 jammy
Kernel: x86_64 Linux 6.5.0-18-generic
Uptime: 39d 19h 48m
Packages: 1745
Shell: zsh 5.8.1
Disk: 47G / 1,7T (3%)
CPU: Intel Xeon E5-4603 v2 @ 32x 2,2GHz [31.0°C]
GPU: Matrox Electronics Systems Ltd. G200eR2
RAM: 4381MiB / 128831MiB
```

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

### Resources
```python
konstantinos@snorlax
OS: Ubuntu 22.04 jammy
Kernel: x86_64 Linux 6.2.0-36-generic
Uptime: 112d 32m
Packages: 1630
Shell: zsh 5.8.1
Disk: 946G / 2,3T (44%)
CPU: Intel Core i7-9700K @ 8x 4,9GHz [46.0°C]
GPU: NVIDIA GeForce RTX 2080 Ti
RAM: 6622MiB / 64228MiB
```

### LinearNN Experiments

To run one experiment:
```
python regression_with_dl.py --trials optuna
```

To run all one-by-one:
```
nohup ./dl_exps.sh > ./final/dl/dl_exps.log 2>&1 &
```


