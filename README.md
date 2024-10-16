# AutoER: Auto Configuring Entity Resolution pipelines

This is the code and reproducibility code of the relevant submitted paper.

Contains:
- `data/`: datasets used for this paper (can be downloaded from here:).
- `figures/`: contains all figures created for paper
- `sheets/`: csv and spearsheets containing results
- `without_gt/`: all code and scripts to build P1s results
- `with_gt/`: all code and scripts to build P2s results (AutoML & LinearRegression)
- `baseline/': code used to replicate ZeroER
- `benchmarking/`: code used for evaluating ETEER pipeline in DBpedia
- `results.ipynd`: a view in results, figures and tables generator

Following instructions for build & execution.

# Problem 1: **With** Ground-Truth file

## Build

Create conda env

```
conda env create -f autoconf_env_automl.yml
conda activate autoconf_p1_p2
```

## Execution

Go to `/with_gt/scripts/` and run 

```
nohup ./run_exps.sh 2>&1 & 
```

In the end a concatenation is made to get the appropriate files needed. 
# Problem 2: **Without** Ground-Truth file

## AutoML Approach

### Build

Create conda env:

```
conda env create -f autoconf_env_automl.yml
conda activate autoconf_automl
```

To run one experiment:
```
python regression_with_automl.py --dataset gridsearch
```

To run all one-by-one:
```
nohup ./automl_exps.sh > ./automl_exps.log 2>&1 &
```

## Linear Regression

# Scalability tests on DBpedia dataset

# Baseline

## ZeroER

1. Go to `cd ./baselines`
2. Create conda env `conda env create -f environment.yml conda activate ZeroER`:
3. Run all exps `./run.sh ./logs`

# Resources

| Spec    | Exp. P1 & P2                             | Exp. P2 - AutoML                                                   |
|---------|------------------------------------------|--------------------------------------------------------------------|
| OS      | Ubuntu 22.04 jammy                       | Ubuntu 22.04 jammy                                                 |
| Kernel  | x86_64 Linux 6.2.0-36-generic            | x86_64 Linux 6.5.0-18-generic                                      |
| CPU     | Intel Core i7-9700K @ 8x 4.9GHz [46.0°C] | Intel Xeon E5-4603 v2 @ 32x 2.2GHz [31.0°C]                        |
| GPU     | NVIDIA GeForce RTX 2080 Ti               | Matrox Electronics Systems Ltd. G200eR2                            |
| RAM     | 6622MiB / 64228MiB                       | 4381MiB / 128831MiB                                                |
