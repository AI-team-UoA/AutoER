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

from sklearn.inspection import plot_partial_dependence, permutation_importance

import argparse
import time
import pickle

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--testdata', type=str, required=True)
args = parser.parse_args()
test_dataset_name = args.testdata

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------          PARAMS          ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

# AUTOML_PER_RUNTIME = 5*60
# AUTOML_OVERALL_RUNTIME = 10*60
AUTOML_PER_RUNTIME = 30*60
AUTOML_OVERALL_RUNTIME = 3*60*60

AUTOML_MEMORY = 6144*4
AUTOML_NJOBS = 1
TOPK = 20
RANDOM_STATE = 42

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# --------------------  READING TRAINING DATASET     ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

trials = pd.read_csv('../data/trials.csv', sep=',')
all_trials = trials.copy()

trials = trials[trials['sampler']!='gridsearch']
trials = trials[trials['f1']!=0]

# Round column in 4 decimals
trials['f1'] = trials['f1'].round(4)
trials['threshold'] = trials['threshold'].round(4)

dataset_specs = pd.read_csv('../data/dataset_specs.csv', sep=',')
datasets = dataset_specs['dataset'].unique()
trials = pd.merge(trials, dataset_specs, on='dataset')

trials.drop_duplicates(inplace=True)

features = ['clustering', 'lm', 'k', 'threshold', 'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
            'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity', 
            'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
            'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity']
trials = trials[features + ['f1', 'dataset']]
print("Trials Shape: ", trials.shape)
# trials.to_csv('trials_optuna_clean.csv', sep=',', index=False)

# Shuffle the dataset

trials = trials.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
trials.head(1000)


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# --------------------      READING TEST DATASET     ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

test_dataset = pd.read_csv(f'../data/{test_dataset_name}_test_data.csv', sep=',')
print(f"{test_dataset_name} Data Shape: ", test_dataset.shape)


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    EXPERIMENTS           ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 


print("\n\n-----------------------------------\n")
print("TEST SET: ", test_dataset_name)
print("TRAINING with: ", [x for x in datasets])

testD = test_dataset
trainD = trials

print("\n\nNumber of entities")
y_train = trainD[['f1']]
X_train = trainD[features]
print("Train Size: ", len(X_train))

X_test = testD[features]
# y_test = testD[['f1']]
print("Test Size: ", len(X_test))

X_train_dummy = pd.get_dummies(X_train)
X_test_dummy = pd.get_dummies(X_test)

# if D == 'D3' and dataset == 'gridsearch':
#     X_train_dummy = X_train_dummy.drop(columns=['lm_sent_glove'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_dummy)
X_test_scaled = scaler.transform(X_test_dummy)

TRAIN_RUNTIME = time.time()

print("Started training at: ", time.ctime())

automl = autosklearn.AutoSklearnRegressor(
    time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
    per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
    memory_limit=AUTOML_MEMORY, 
    n_jobs=AUTOML_NJOBS,
    ensemble_size=1               # Use single best model
)


# Fit the model
automl.fit(X_train_scaled, y_train, dataset_name='trials_optuna_all')
print("Finished training at: ", time.ctime())
print("Training time: ", time.time()-TRAIN_RUNTIME)
# END_TO_END_RUNTIME = time.time() - END_TO_END_RUNTIME    
PREDICTION_RUNTIME = time.time()

# Predict using the best model
print("Started prediction at: ", time.ctime())
y_pred = automl.predict(X_test_scaled)
print("Finished prediction at: ", time.ctime())
print("Prediction time: ", time.time()-PREDICTION_RUNTIME)


# Display the details of the best model
print("\n\nBest Model Configuration: ")
print(automl.show_models())

# Save the best model
with open('dbpedia_best_model.pkl', 'wb') as f:
    pickle.dump(automl, f)

print("Best model saved as 'best_model.pkl'")

# Evaluate the predictions
ensemble = automl.get_models_with_weights()
for weight, model in ensemble:
    model_configuration = model.get_params()
    regressor_name = model_configuration['config']['regressor:__choice__']


result = X_test[['lm', 'clustering', 'k', 'threshold']]
result['predicted'] = y_pred
result['dataset'] = testD['Dataset']
result.to_csv(f'./predictions/{test_dataset_name}_predictions.csv', sep=',', index=False)
