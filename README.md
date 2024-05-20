# pyJedAI Auto Configuration
Auto Configuration experiments for pyJedAI

## Classic Regressors

### Resources
```python
                          ./+o+-       nikoletos@pyravlos5
                  yyyyy- -yyyyyy+      OS: Ubuntu 22.04 jammy
               ://+//////-yyyyyyo      Kernel: x86_64 Linux 6.5.0-18-generic
           .++ .:/++++++/-.+sss/`      Uptime: 39d 19h 48m
         .:++o:  /++++++++/:--:/-      Packages: 1745
        o:+o+:++.`..```.-/oo+++++/     Shell: zsh 5.8.1
       .:+o:+o/.          `+sssoo+/    Disk: 47G / 1,7T (3%)
  .++/+:+oo+o:`             /sssooo.   CPU: Intel Xeon E5-4603 v2 @ 32x 2,2GHz [31.0°C]
 /+++//+:`oo+o               /::--:.   GPU: Matrox Electronics Systems Ltd. G200eR2
 \+/+o+++`o++o               ++////.   RAM: 4381MiB / 128831MiB
  .++.o+++oo+:`             /dddhhh.  
       .+.o+oo:.          `oddhhhh+   
        \+.++o+o``-````.:ohdhhhhh+    
         `:o+++ `ohhhhhhhhyo++os:     
           .o:`.syhhhhhhh/.oo++o`     
               /osyyyyyyo++ooo+++/    
                   ````` +oo+++o\:    
                          `oo++.   
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
                          ./+o+-       konstantinos@snorlax
                  yyyyy- -yyyyyy+      OS: Ubuntu 22.04 jammy
               ://+//////-yyyyyyo      Kernel: x86_64 Linux 6.2.0-36-generic
           .++ .:/++++++/-.+sss/`      Uptime: 112d 32m
         .:++o:  /++++++++/:--:/-      Packages: 1630
        o:+o+:++.`..```.-/oo+++++/     Shell: zsh 5.8.1
       .:+o:+o/.          `+sssoo+/    Disk: 946G / 2,3T (44%)
  .++/+:+oo+o:`             /sssooo.   CPU: Intel Core i7-9700K @ 8x 4,9GHz [46.0°C]
 /+++//+:`oo+o               /::--:.   GPU: NVIDIA GeForce RTX 2080 Ti
 \+/+o+++`o++o               ++////.   RAM: 6622MiB / 64228MiB
  .++.o+++oo+:`             /dddhhh.  
       .+.o+oo:.          `oddhhhh+   
        \+.++o+o``-````.:ohdhhhhh+    
         `:o+++ `ohhhhhhhhyo++os:     
           .o:`.syhhhhhhh/.oo++o`     
               /osyyyyyyo++ooo+++/    
                   ````` +oo+++o\:    
                          `oo++.  
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


