# Benchmarking on DBPedia dataset

## Build

Create a conda env 3.10, pip install optuna and pyjedai.

## Execution

Concatenate dbpedia features with ground-truth trials configurations:
```python
nohup python create_test_trials.csv --data $D  > ${D}_trials.log 2>&1 &
```

Create a model with AutoML build upon D1-D10 datasets and predict the gridsearch confs:
```python
nohup python predict_on_dbpedia_data.py --testdata $D > ${D}.log 2>&1 &     
```

Evaluate the best topk predicted configurations with pyJedAI to get the real value:
```python
nohup python ./evaluate.py --topk 1 --data dbpedia > ./dbpedia.log 2>&1 &  
```
