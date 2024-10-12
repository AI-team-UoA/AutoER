import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')
import autosklearn.regression as autosklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

import os

from sklearn.inspection import plot_partial_dependence, permutation_importance

import argparse
import time
import pickle
import json

parser = argparse.ArgumentParser(description='Read configuration from a JSON file')
parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')

# Parse the arguments
args = parser.parse_args()

# Read the JSON file
with open(args.config, 'r') as f:
    config = json.load(f)

# Access the configurations
benchmark_name = config['benchmark_name']
test_datasets = config['test_datasets']
train_datasets = config['train_datasets']
trials_type = config['trials_type']
PER_RUNTIME_HOURS = config['per_runtime_h']
OVERALL_RUNTIME_HOURS = config['overall_runtime_h']
ENSEMBLE_SIZE = int(config['ensemble'])
WITH_ABLATION_ANALYSIS = config['ablation']
TOPK = int(config['topk'])

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------          PARAMS          ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

AUTOML_PER_RUNTIME = PER_RUNTIME_HOURS*60*60
AUTOML_OVERALL_RUNTIME = OVERALL_RUNTIME_HOURS*60*60
AUTOML_MEMORY = 6144*4
AUTOML_NJOBS = 1
RANDOM_STATE = 42

print("\n\n-----------------------------------\n")
print("TRAINING with: ", train_datasets)
print("TESTING with: ", test_datasets)
print("AUTOML_PER_RUNTIME: ", AUTOML_PER_RUNTIME)
print("AUTOML_OVERALL_RUNTIME: ", AUTOML_OVERALL_RUNTIME)
print("AUTOML_MEMORY: ", AUTOML_MEMORY)
print("AUTOML_NJOBS: ", AUTOML_NJOBS)
print("TOPK: ", TOPK)
print("ENSEMBLE_SIZE: ", ENSEMBLE_SIZE)
print("RANDOM_STATE: ", RANDOM_STATE)


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# --------------------  READING TRAINING DATASET     ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

print("\n\n-----------------------------------\n")
print("Reading/Formatting training dataset")
print("Trials Type: ", trials_type)
print("Train Datasets: ", train_datasets)
print("Reading trials from: ", '../data/trials.csv')

trials = pd.read_csv('../data/trials.csv', sep=',')
all_trials = trials.copy()
print("RAW Trials Shape: ", trials.shape)
print("Trials Type: ", trials_type)
print("Available trials: ", trials['sampler'].unique())

if trials_type == 'all':
    trials = all_trials
elif trials_type == 'optuna':
    trials = trials[trials['sampler']!='gridsearch']
else:
    trials = trials[trials['sampler']=='gridsearch']

trials = trials[trials['f1']!=0]

print("Trials Shape: ", trials.shape)

print(trials['dataset'].unique())
print(train_datasets)

trials = trials[trials['dataset'].isin(train_datasets)]
print("Trials Shape after dataset selection: ", trials.shape)
print("CHECK: Trials datasets: ", trials['dataset'].unique())

trials['f1'] = trials['f1'].round(4)
trials['threshold'] = trials['threshold'].round(4)

dataset_specs = pd.read_csv('../data/dataset_specs.csv', sep=',')
trials = pd.merge(trials, dataset_specs, on='dataset')

trials.drop_duplicates(inplace=True)

features = ['clustering', 'lm', 'k', 'threshold']
dataset_specs_features = dataset_specs.columns.tolist()
dataset_specs_features.remove('dataset')
if not WITH_ABLATION_ANALYSIS:
    features = features + dataset_specs_features

training_trials = trials[features + ['f1', 'dataset']]
print("Training Trials Shape: ", training_trials.shape)

training_trials = training_trials.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(training_trials.head(10))

print("Trials Shape: ", training_trials.shape)
print("Trials columns: ", training_trials.columns)
print("Trials dataset: ", training_trials['dataset'].unique())

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# --------------------    JOIN WITH DATA FEATURES    ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

print("\n\n-----------------------------------\n")
print("Joining with GRIDSEARCH data features")

gridsearch_trials = pd.read_csv('../data/trials.csv', sep=',').where(lambda x: x['sampler']=='gridsearch').dropna()
# gridsearch_trials = trials[trials['sampler']=='gridsearch']
gridsearch_trials = gridsearch_trials[['clustering', 'lm', 'k', 'threshold']]

gridsearch_trials.drop_duplicates(inplace=True)

print("Gridsearch trials are:",  gridsearch_trials.shape)

all_dataspecs = pd.read_csv(f'../data/dataset_specs.csv', sep=',')

test_set_as_ETEER = {}
print("Preparing test datasets for predictions")
for test_dataset_name in test_datasets:
    dataspecs = all_dataspecs[all_dataspecs['dataset']==test_dataset_name]

    print("Dataspecs: ")
    print(dataspecs)

    trials_for_join = gridsearch_trials.copy()
    
    cartesian_product = trials_for_join.assign(key=1).merge(dataspecs.assign(key=1), on='key').drop('key', axis=1)

    ready_for_predictions_df = cartesian_product[features]

    ready_for_predictions_df.drop_duplicates(inplace=True)
    print("Shape of ", test_dataset_name , " is: " ,ready_for_predictions_df.shape)

    test_set_as_ETEER[test_dataset_name] = ready_for_predictions_df


# sys.exit(0)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    EXPERIMENTS           ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 


print("\n\n-----------------------------------\n")
# print("TESTING with: ", test_datasets)
print("TRAINING with: ", train_datasets)

trainD = training_trials

print("\n\nNumber of entities")
y_train = trainD[['f1']]
X_train = trainD[features]
print("Train Size: ", len(X_train))

X_train_dummy = pd.get_dummies(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_dummy)

TRAIN_RUNTIME = time.time()

print("Started training at: ", time.ctime())
current_time = datetime.now()
approx_time = current_time + timedelta(seconds=AUTOML_OVERALL_RUNTIME)
print("and will approximately finish at: ", approx_time)

if ENSEMBLE_SIZE != 0:
    automl = autosklearn.AutoSklearnRegressor(
        time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
        per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
        memory_limit=AUTOML_MEMORY, 
        n_jobs=AUTOML_NJOBS,
        ensemble_size=ENSEMBLE_SIZE, 
        seed=RANDOM_STATE,
    )
else:
    automl = autosklearn.AutoSklearnRegressor(
        time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
        per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
        memory_limit=AUTOML_MEMORY, 
        n_jobs=AUTOML_NJOBS,
        seed=RANDOM_STATE,
    )

automl.fit(X_train_scaled, y_train, dataset_name='trials_optuna')
print("Finished training at: ", time.ctime())
print("Training time: ", time.time()-TRAIN_RUNTIME)
TRAIN_RUNTIME = time.time() - TRAIN_RUNTIME
print("\n\nBest Model Configuration: ")
print(automl.show_models())

ensemble = automl.get_models_with_weights()
for weight, model in ensemble:
    model_configuration = model.get_params()
    regressor_name = model_configuration['config']['regressor:__choice__']

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------           TESTING        ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 


RESULTS_FILE_NAME = f'./results/{benchmark_name}.csv'

with open(RESULTS_FILE_NAME, 'w') as f:
    f.write('dataset,clustering,lm,k,threshold,training_time,prediction_time,predicted_f1,real_f1\n')

    for test_dataset_name in test_datasets:

        PREDICTION_RUNTIME = time.time()

        testD = test_set_as_ETEER[test_dataset_name]

        X_test = testD[features]
        X_test_dummy = pd.get_dummies(X_test)
        X_test_scaled = scaler.transform(X_test_dummy)
        
        y_pred = automl.predict(X_test_scaled)

        PREDICTION_RUNTIME = time.time() - PREDICTION_RUNTIME
        print(f"Prediction time for {test_dataset_name}: {PREDICTION_RUNTIME}")
        testD['predicted_f1'] = y_pred
        results = testD.sort_values(by='predicted_f1', ascending=False).head(TOPK)

        if 'dbpedia' in test_dataset_name:
            real_f1 = -1
        else:
            best_config = all_trials[
                            (all_trials['sampler']=='gridsearch') & \
                            (all_trials['dataset']==test_dataset_name) & \
                            (all_trials['clustering']==results.iloc[0]['clustering']) & \
                            (all_trials['lm']==results.iloc[0]['lm']) & \
                            (all_trials['k']==results.iloc[0]['k']) & \
                            (all_trials['threshold']==results.iloc[0]['threshold'])]
            real_f1 = best_config['f1'].values[0]

        #  round 4 decimal places
        TRAIN_RUNTIME = round(TRAIN_RUNTIME, 4)
        PREDICTION_RUNTIME = round(PREDICTION_RUNTIME, 4)
        results['threshold'] = results['threshold'].round(4)
        results['predicted_f1'] = results['predicted_f1'].round(4)
        real_f1 = round(real_f1, 4)


        f.write(f"{test_dataset_name}, {results.iloc[0]['clustering']}, {results.iloc[0]['lm']}, {results.iloc[0]['k']}, {results.iloc[0]['threshold']}, {TRAIN_RUNTIME}, {PREDICTION_RUNTIME}, {results.iloc[0]['predicted_f1']}, {real_f1}\n")        
        print(f"{test_dataset_name}, {results.iloc[0]['clustering']}, {results.iloc[0]['lm']}, {results.iloc[0]['k']}, {results.iloc[0]['threshold']}, {TRAIN_RUNTIME}, {PREDICTION_RUNTIME}, {results.iloc[0]['predicted_f1']}, {real_f1}\n")        
        print(f"> Results for {test_dataset_name} are saved in {RESULTS_FILE_NAME}")

        ANALYTICAL_RESULTS_DIR = f'./results/analytical/{benchmark_name}/'

        if not os.path.exists(ANALYTICAL_RESULTS_DIR):
            os.makedirs(ANALYTICAL_RESULTS_DIR)
        
        ANALYTICAL_RESULTS_FILE_NAME = f'{ANALYTICAL_RESULTS_DIR}{test_dataset_name}.json'

        report_analytical_json = {
            'dataset': test_dataset_name,
            'clustering': results.iloc[0]['clustering'],
            'lm': results.iloc[0]['lm'],
            'k': results.iloc[0]['k'],
            'threshold': results.iloc[0]['threshold'],
            'predicted_f1': results.iloc[0]['predicted_f1'],
            'real_f1': real_f1,
            'training_time': TRAIN_RUNTIME,
            'prediction_time': PREDICTION_RUNTIME,
            'topk': TOPK,
            'ensemble_size': ENSEMBLE_SIZE,
            'per_runtime': PER_RUNTIME_HOURS,
            'overall_runtime': OVERALL_RUNTIME_HOURS,
            'random_state': RANDOM_STATE,
            'ablation': WITH_ABLATION_ANALYSIS,
            'test_datasets': test_datasets,
            'train_datasets': train_datasets,
            'benchmark_name': benchmark_name
        }

        with open(ANALYTICAL_RESULTS_FILE_NAME, 'w') as json_file:
            json.dump(report_analytical_json, json_file)
        
        print(f"Results for {test_dataset_name} are saved in {ANALYTICAL_RESULTS_FILE_NAME}")
