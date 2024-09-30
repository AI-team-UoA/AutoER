import sys
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

from sklearn.inspection import permutation_importance

import json
import os

import argparse
import time

from pprint import pprint

parser = argparse.ArgumentParser(description='Read configuration from a JSON file')
parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
parser.add_argument('--hidden_dataset', type=str, required=True, help='Hidden dataset to run the experiments')
parser.add_argument('--trials_type', type=str, required=True, help='Type of trials to run')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

hidden_dataset = args.hidden_dataset
trials_type = args.trials_type

PER_RUNTIME_HOURS = config['per_runtime_h']
OVERALL_RUNTIME_HOURS = config['overall_runtime_h']
ENSEMBLE_SIZE = int(config['ensemble'])
TOPK = int(config['topk'])

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------          PARAMS          ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

DATA_DIR = '../data/'
DIR = './automl/results/'

AUTOML_PER_RUNTIME = PER_RUNTIME_HOURS*60*60
AUTOML_OVERALL_RUNTIME = OVERALL_RUNTIME_HOURS*60*60
AUTOML_MEMORY = 6144*4
AUTOML_NJOBS = 1
RANDOM_STATE = 42

print("\nConfiguration: ")
print("Dataset: ", hidden_dataset)
print("Trials Type: ", trials_type)
print("PER_RUNTIME_HOURS: ", PER_RUNTIME_HOURS)
print("OVERALL_RUNTIME_HOURS: ", OVERALL_RUNTIME_HOURS)
print("AUTOML_MEMORY: ", AUTOML_MEMORY)
print("AUTOML_NJOBS: ", AUTOML_NJOBS)
print("TOPK: ", TOPK)
print("ENSEMBLE_SIZE: ", ENSEMBLE_SIZE)
print("RANDOM_STATE: ", RANDOM_STATE)
print()
print("Directory reading trials: ", DATA_DIR)
print("Directory working: ", DIR)

TIME_STARTED = time.ctime()

SUB_DIR = DIR+str(OVERALL_RUNTIME_HOURS)+'_'+str(PER_RUNTIME_HOURS)+'_'+str(ENSEMBLE_SIZE)+'/'+trials_type+'/'

if not os.path.exists(SUB_DIR):
    os.makedirs(SUB_DIR)
    print("Created: ", SUB_DIR)

RESULTS_SUB_DIR = SUB_DIR+'scores/'
if not os.path.exists(RESULTS_SUB_DIR):
    os.makedirs(RESULTS_SUB_DIR)
    print("Created: ", RESULTS_SUB_DIR)
else:
    print(RESULTS_SUB_DIR, " already exists")

IMPORTANCE_SUB_DIR = SUB_DIR+'importance/'
if not os.path.exists(IMPORTANCE_SUB_DIR):
    os.makedirs(IMPORTANCE_SUB_DIR)
    print("Created: ", IMPORTANCE_SUB_DIR)
else:
    print(IMPORTANCE_SUB_DIR, " already exists")

DETAILED_RESULTS_SUB_DIR = SUB_DIR+'detailed/'
if not os.path.exists(DETAILED_RESULTS_SUB_DIR):
    os.makedirs(DETAILED_RESULTS_SUB_DIR)
    print("Created: ", DETAILED_RESULTS_SUB_DIR)
else:
    print(DETAILED_RESULTS_SUB_DIR, " already exists")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    READING DATASET       ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

trials = pd.read_csv(DATA_DIR+'trials.csv', sep=',')
all_trials = trials.copy()

if trials_type == 'gridsearch':
    trials = trials[trials['sampler']=='gridsearch']
elif trials_type == 'all':
    pass
else:
    trials = trials[trials['sampler']!='gridsearch']

print("Preprocessing trials..")
trials = trials[trials['f1']!=0]
trials['f1'] = trials['f1'].round(4)
trials['threshold'] = trials['threshold'].round(4)

dataset_specs = pd.read_csv(DATA_DIR+'dataset_specs.csv', sep=',')
datasets = dataset_specs['dataset'].unique()
trials = pd.merge(trials, dataset_specs, on='dataset')

trials.drop_duplicates(inplace=True)

features = ['clustering', 'lm', 'k', 'threshold', 'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
            'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity', 
            'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
            'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity']
trials = trials[features + ['f1', 'dataset']]

# trials.to_csv('trials_optuna_clean.csv', sep=',', index=False)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    EXPERIMENTS           ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

filename = RESULTS_SUB_DIR+trials_type+'.csv' if hidden_dataset is None else RESULTS_SUB_DIR+hidden_dataset+'.csv'
f = open(filename, 'a')
f.write('TEST_SET, AUTOML_REGRESSOR, TEST_MSE, PREDICTED_F1, GLOBAL_BEST_F1, PERFORMANCE, PREDICTIONS_RUNTIME, OPTIMIZATION_TIME\n')
f.flush()
print("Writing to: ", filename)

for D in datasets:

    if hidden_dataset is not None:
        if D != hidden_dataset:
            continue
        else:
            D = hidden_dataset
            print("Processing: ", D)

    print("\n\n-----------------------------------\n")
    print("TEST SET: ", D)
    print("TRAINING with: ", [x for x in datasets if x!=D])

    testD = trials[trials['dataset']==D]
    trainD = trials[trials['dataset']!=D]
    trainDatasets = [x for x in datasets if x!=D]

    print("\n\nNumber of entities")
    y_train = trainD[['f1']]
    X_train = trainD[features]
    print("Train Size: ", len(X_train))

    X_test = testD[features]
    y_test = testD[['f1']]
    print("Test Size: ", len(X_test))

    X_train_dummy = pd.get_dummies(X_train)
    X_test_dummy = pd.get_dummies(X_test)

    print(X_train_dummy.columns)
    print("Size: ", len(X_train_dummy.columns))
    print(X_test_dummy.columns)
    print("Size: ", len(X_test_dummy.columns))
    print("Difference: ", set(X_train_dummy.columns) - set(X_test_dummy.columns))

    cols_to_remove = set(X_train_dummy.columns) - set(X_test_dummy.columns)

    for col in cols_to_remove:
        print("Removing:", col)    

    if len(cols_to_remove) > 0:    
        X_train_dummy = X_train_dummy.drop(columns=[col])
        print("New size: ", len(X_train_dummy.columns))

    if len(X_train_dummy.columns) != len(X_test_dummy.columns):
        print("Columns don't match")
        print("Train: ", X_train_dummy.columns)
        print("Test: ", X_test_dummy.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dummy)
    X_test_scaled = scaler.transform(X_test_dummy)

    END_TO_END_RUNTIME = time.time()

    if ENSEMBLE_SIZE == 0:
        automl = autosklearn.AutoSklearnRegressor(
            time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
            per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
            memory_limit=AUTOML_MEMORY, 
            n_jobs=AUTOML_NJOBS,
            seed=RANDOM_STATE
        )
    else:
        automl = autosklearn.AutoSklearnRegressor(
            time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
            per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
            memory_limit=AUTOML_MEMORY, 
            n_jobs=AUTOML_NJOBS,
            ensemble_size=1,
            seed=RANDOM_STATE
        )
    
    print("Started training at: ", time.ctime())
    current_time = datetime.now()
    approx_time = current_time + timedelta(seconds=AUTOML_OVERALL_RUNTIME)
    print("-> and will approximately finish at: ", approx_time)
    # Fit the model
    automl.fit(X_train_scaled, y_train, dataset_name='trials_optuna_'+str(D))

    END_TO_END_RUNTIME = time.time() - END_TO_END_RUNTIME    
    PREDICTION_RUNTIME = time.time()

    # Predict using the best model
    y_pred = automl.predict(X_test_scaled)
    
    PREDICTION_RUNTIME = time.time() - PREDICTION_RUNTIME

    # Display the details of the best model
    print("\n\nBest Model Configuration: ")
    pprint(automl.show_models(), indent=4)

    # Evaluate the predictions
    ensemble = automl.get_models_with_weights()
    regressor_name = None
    for weight, model in ensemble:
        if regressor_name == None:
            regressor_name = model.get_params()['config']['regressor:__choice__']
        model_configuration = model.get_params()
        regressor_name = " | ".join([regressor_name, model_configuration['config']['regressor:__choice__']])

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------     EVALUATION           ----------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

    print("\n\nPerformance on Test Set: ", D)
    TEST_MSE = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", TEST_MSE)

    result = X_test[['lm', 'clustering', 'k', 'threshold']]

    # add y_pred and y_test to res
    result['Predicted'] = y_pred
    result['True'] = y_test
    topKpredicted = result.sort_values(by='Predicted', ascending=False).head(TOPK)
    topKtrue = result.sort_values(by='True', ascending=False).head(TOPK)
    
    print("\n\nTop K (Sorted on Predicted): ")
    print(topKpredicted)

    print("\nTop K (Sorted on True)")
    print(topKtrue)

    LOCAL_BEST_TRUE = topKtrue['True'].max()
    # get the row with the max Predicted
    BEST_PREDICTED = topKpredicted['Predicted'].idxmax()
    BEST_PREDICTED = topKpredicted.loc[BEST_PREDICTED, 'True']

    print("\n\nBest Predicted: ", BEST_PREDICTED)
    print("Local Best True: ", LOCAL_BEST_TRUE)
    GLOBAL_MAX_TRUE = all_trials[all_trials['dataset']==D]['f1'].max()
    print("Global Max True: ", GLOBAL_MAX_TRUE)
    PERFORMANCE = BEST_PREDICTED / GLOBAL_MAX_TRUE
    diff = GLOBAL_MAX_TRUE - BEST_PREDICTED
    PERFORMANCE = round(PERFORMANCE, 4)
    print("Performance: ", PERFORMANCE)
    print("Difference between Predicted and Global Best: ", round(diff, 4))
    
    TEST_MSE = round(TEST_MSE, 4)
    PREDICTION_RUNTIME = round(PREDICTION_RUNTIME, 4)
    END_TO_END_RUNTIME = round(END_TO_END_RUNTIME, 4)

    f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(D,
                                                  regressor_name, 
                                                    TEST_MSE,
                                                    BEST_PREDICTED,
                                                    GLOBAL_MAX_TRUE,
                                                    PERFORMANCE,
                                                    PREDICTION_RUNTIME,
                                                    END_TO_END_RUNTIME))
    f.flush()

    print("Logged row: ")
    print("{}, {}, {}, {}, {}, {}, {}, {}\n".format(D,
                                                  regressor_name, 
                                                    TEST_MSE,
                                                    BEST_PREDICTED,
                                                    GLOBAL_MAX_TRUE,
                                                    PERFORMANCE,
                                                    PREDICTION_RUNTIME,
                                                    END_TO_END_RUNTIME))

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------   SAVIND DETAILED REPORT  ---------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

    run_params = {
        "AUTOML_PER_RUNTIME_H": PER_RUNTIME_HOURS,
        "AUTOML_OVERALL_RUNTIME_H": OVERALL_RUNTIME_HOURS,
        "AUTOML_MEMORY": AUTOML_MEMORY,
        "AUTOML_NJOBS": AUTOML_NJOBS,
        "TOPK": TOPK,
        "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "automl" : str(automl.get_params()),
        "automl_models" : str(automl.get_models_with_weights()),
        "results" : {
            "TEST_MSE": TEST_MSE,
            "BEST_PREDICTED": BEST_PREDICTED,
            "GLOBAL_MAX_TRUE": GLOBAL_MAX_TRUE,
            "PERFORMANCE": PERFORMANCE,
            "PREDICTION_RUNTIME": PREDICTION_RUNTIME,
            "END_TO_END_RUNTIME": END_TO_END_RUNTIME
        }
    } 

    json_filename = DETAILED_RESULTS_SUB_DIR+str(D)+'.json'
    with open(json_filename, 'w') as j:
        json.dump(run_params, j, indent=4)

    print("DETAILED_RESULTS SAVED TO: ", json_filename)

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------     FEATURE SELECTION    ----------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

    r = permutation_importance(automl, X_test_dummy, y_test, n_repeats=10, random_state=RANDOM_STATE)

    sort_idx = r.importances_mean.argsort()[::-1]

    dummy_features = X_test_dummy.columns
    
    print("\n\nFeature Importance: ")
    for i in sort_idx[::-1]:
        print(
            f"{dummy_features[i]:10s}: {r.importances_mean[i]:.3f} +/- "
            f"{r.importances_std[i]:.3f}"
        )
    print("\n\n")

    feature_importance_extended = pd.DataFrame()
    feature_importance_extended['Feature'] = dummy_features
    feature_importance_extended['Importance'] = r.importances_mean
    feature_importance_extended['Std'] = r.importances_std
    feature_importance_extended['Rank'] = np.arange(len(dummy_features))
    feature_importance_extended.to_csv(IMPORTANCE_SUB_DIR+str(D)+'.csv', index=False)

    plt.boxplot(
        r.importances[sort_idx].T, labels=[dummy_features[i] for i in sort_idx]
    )

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(IMPORTANCE_SUB_DIR+str(D)+'.png')


metadata = {
    "SETUP" : {
        "PER_RUNTIME_HOURS": PER_RUNTIME_HOURS,
        "OVERALL_RUNTIME_HOURS": OVERALL_RUNTIME_HOURS,
        "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
        "TOPK": TOPK,
        "RANDOM_STATE": RANDOM_STATE,
        "AUTOML_MEMORY": AUTOML_MEMORY,
        "AUTOML_NJOBS": AUTOML_NJOBS,
    },
    "RUN_INFO" : {
        "HIDDEN_DATASET": hidden_dataset,
        "TIME_STARTED": TIME_STARTED,
        "TIME_ENDED": time.ctime()
    },
    "STORAGE" : {
        "RESULTS": RESULTS_SUB_DIR,
        "IMPORTANCE": IMPORTANCE_SUB_DIR,
        "DETAILED_RESULTS": DETAILED_RESULTS_SUB_DIR
    }
}

# save the metadata
with open(SUB_DIR+'metadata.json', 'w') as m:
    json.dump(metadata, m, indent=4)

print("Saved metadata to: ", SUB_DIR+'metadata.json')

f.close()
