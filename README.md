# AutoER: Auto Configuring Entity Resolution pipelines

This is the repository of the relevant submitted paper.

Contains:
- `data/`: datasets used for this paper.
- `figures/`: contains all figures created for paper
- `sheets/`: csv and spearsheets containing results
- `without_gt/`: all code and scripts to build P1s results
- `with_gt/`: all code and scripts to build P2s results (AutoML & LinearRegression)
- `baseline/`: code used to replicate ZeroER
- `benchmarking/`: code used for evaluating ETEER pipeline in DBpedia
- `results.ipynd`: a view in results, figures and tables generator

Bellow, you'll find instructions to build & execute this project, experiment by experiment.

# Datasets

Please in the initial directory execute commands to download and prepare datasets:

```
chmod +x prepare_datasets.sh
./prepare_datasets.sh
```

# Problem 1: **With** Ground-Truth file

## Build

Create conda env:

```
conda env create -f autoconf_env_p1_p2.yml
conda activate autoconf_p1_p2
```

## Execution

Go to `/with_gt/scripts/` and run 

```
nohup ./run_exps.sh 2>&1 & 
```

in the end a concatenation is made to get the files in the appropriate format. 

# Problem 2: **Without** Ground-Truth file

## AutoML Approach

### Build

Create conda env:

```
conda env create -f autoconf_env_automl.yml
conda activate autoconf_automl
```

### Execute

To run one experiment:
```
python -u regression_with_automl.py --trials_type $t --hidden_dataset $d --config $config_file
```

where:
- `--trials_type` stands for training instances type
- `--hidden_dataset` stands for training with Di..j and holding Dx us hidden for testing
-  `--config` specifies experiment type

To run all one-by-one:
```
nohup ./automl_exps.sh ./automl/configs/12_4_0.json > ./automl/logs/EXPS_12_4_0.log  2>&1 &\n
```

the config file specifies experiments characteristics, like overall/per model hours for auto-sklearn, etc. 

and in the end, you need to conactenate all results into a format that can be read by the notebook, for merging purposes. 

Execute:

```
python concatenate.py --exp 12_4_0
```

where `--exp` stands for the experiment name executed before.

## Linear Regression


### Build

Create conda env:

```
conda env create -f autoconf_env_p1_p2.yml
conda activate autoconf_env_p1_p2
```

### Execute

To run one experiment:
```
python -u regression_with_sklearn.py --trials $dataset --regressor "LINEAR"
```

where:
- `--trials` stands for training instances type

To run all one-by-one:
```
nohup ./sklearn_exps.sh > sklearn_exps.log 2>&1 &
```

## Merging all results into common files

After all experiments have finished, run:

```
python concatenate_exps.py
```

and you're ready!

# Scalability tests on DBpedia dataset

## Using AutoML approach

Executing this will create the top-1 workflow suggested per training trials type for DBPedia.
```
nohup ./run_dbpedia_exps.sh > ./logs/dbpedia.log  2>&1 &
```

## Using LR approach

Create predictions for all instances:

```
python eteer_evaluate_ind_regressors.py --config ./configs/D1D10_DBPEDIA_ALL_LinearRegression.json
python eteer_evaluate_ind_regressors.py --config ./configs/D1D10_DBPEDIA_OPTUNA_LinearRegression.json
python eteer_evaluate_ind_regressors.py --config ./configs/D1D10_DBPEDIA_GRIDSEARCH_LinearRegression.json
```

where:
- `--config` stands for the experiment specifications (these configs are included). For example `D1D10_DBPEDIA_ALL_LinearRegression.json`, title stands for train in D1...D10 test in DBPEDIA, use all trials instances, and Linear Regression.


## Evaluating the prediction to get the real F1 (applies to both types of training)

For AutoML:
```
 ./eval_dbpedia_exps.sh
```

or for LR:

```
nohup python -u evaluate.py --confcsv ./results/D1D10_DBPEDIA_{$TYPE}_LinearRegression.csv  --datajson ./configs/data/dbpedia.json > ./logs/D1D10_DBPEDIA_LR.log 2>&1 &
```

same for `{$TYPE} = ALL, OPTUNA, GRIDSEARCH`

where:
-  `--confcsv`: is used in a similar way as before
-  ` --datajson`: contains the needed information of the dataset that will be evaluated

# Baseline

## ZeroER

1. Go to `cd ./baselines`
2. Create conda env
    1. `conda env create -f environment.yml`
    2. `conda activate ZeroER`
4. Run all exps `./run.sh ./logs`

## DITTO

Downloading NVIDIA container toolkit:
```
chmod +x nvidia_installation.sh
./nvidia_installation.sh
```

Creating the environment:
```
sudo docker build -t ditto ditto
```

Configuration:
```
CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D2  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize
```

Blocks for DITTO created in ready_for_ditto_input directory, using:
```
transform_all_for_ditto.sh
```
and more specifically:
```
python blocking.py --datajson '../../data/configs/D2.json'
```

where datajson is the configuration file for the dataset.


Moving files inside docker container:
```
docker cp ./configs.json acc70a93a256:/workspace/ditto     
docker cp ./ready_for_ditto_input/ acc70a93a256:/workspace/ditto/data/./ready_for_ditto_input/  
docker cp ./train_ditto.py acc70a93a256:/workspace/ditto
docker cp ./run_all_inside.sh 54d79d32d83d:/workspace/ditto
``` 

Entering docker:
```
sudo docker run -it --gpus all --entrypoint=/bin/bash ditto       
```

Inside docker:
```
cd /workspace/ditto
mkdir logs
chmod +x run_all_inside.sh
nohup ./run_all_inside.sh > nohup.out 2>&1 & 
```

Results will be in `./workspace/ditto/logs/`.

# Resources

| Spec    | Exp. P1 & P2                             | Exp. P2 - AutoML                                                   |
|---------|------------------------------------------|--------------------------------------------------------------------|
| OS      | Ubuntu 22.04 jammy                       | Ubuntu 22.04 jammy                                                 |
| Kernel  | x86_64 Linux 6.2.0-36-generic            | x86_64 Linux 6.5.0-18-generic                                      |
| CPU     | Intel Core i7-9700K @ 8x 4.9GHz [46.0°C] | Intel Xeon E5-4603 v2 @ 32x 2.2GHz [31.0°C]                        |
| GPU     | NVIDIA GeForce RTX 2080 Ti               | Matrox Electronics Systems Ltd. G200eR2                            |
| RAM     | 6622MiB / 64228MiB                       | 4381MiB / 128831MiB                                                |
