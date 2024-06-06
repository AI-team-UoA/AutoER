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

# read args
# parser = argparse.ArgumentParser()
# parser.add_argument('--trials', type=str, required=True)
# args = parser.parse_args()
# dataset = args.trials

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------          PARAMS          ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

DIR = 'final/automl/'
FILE = ''
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

trials = pd.read_csv('./data/trials.csv', sep=',')
all_trials = trials.copy()

trials = trials[trials['sampler']!='gridsearch']
trials = trials[trials['f1']!=0]

# Round column in 4 decimals
trials['f1'] = trials['f1'].round(4)
trials['threshold'] = trials['threshold'].round(4)

dataset_specs = pd.read_csv('./data/dataset_specs.csv', sep=',')
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

census_data = pd.read_csv('./data/census_test_data.csv', sep=',')
print("Census Data Shape: ", census_data.shape)


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    EXPERIMENTS           ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 


print("\n\n-----------------------------------\n")
print("TEST SET: ", 'CENSUS')
print("TRAINING with: ", [x for x in datasets])

testD = census_data
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

# END_TO_END_RUNTIME = time.time()

automl = autosklearn.AutoSklearnRegressor(
    time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
    per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
    memory_limit=AUTOML_MEMORY, 
    n_jobs=AUTOML_NJOBS,
    ensemble_size=1               # Use single best model
)

# Fit the model
automl.fit(X_train_scaled, y_train, dataset_name='trials_optuna_all')

# END_TO_END_RUNTIME = time.time() - END_TO_END_RUNTIME    
# PREDICTION_RUNTIME = time.time()

# Predict using the best model
y_pred = automl.predict(X_test_scaled)
    
# PREDICTION_RUNTIME = time.time() - PREDICTION_RUNTIME

# Display the details of the best model
print("\n\nBest Model Configuration: ")
print(automl.show_models())

# Evaluate the predictions
ensemble = automl.get_models_with_weights()
for weight, model in ensemble:
    model_configuration = model.get_params()
    regressor_name = model_configuration['config']['regressor:__choice__']

# -------------------------------------------------------------------------- # 
# -------------------------------------------------------------------------- # 
# -------------------------     EVALUATION           ----------------------- # 
# -------------------------------------------------------------------------- # 
# -------------------------------------------------------------------------- # 

# print("\n\nPerformance on Test Set: ", D)
# TEST_MSE = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", TEST_MSE)

result = X_test[['lm', 'clustering', 'k', 'threshold']]

# add y_pred and y_test to res
result['Predicted'] = y_pred
result['dataset'] = testD['dataset']


result.to_csv('census_predictions.csv', sep=',', index=False)

# result['True'] = y_test
# topKpredicted = result.sort_values(by='Predicted', ascending=False).head(TOPK)
# topKtrue = result.sort_values(by='True', ascending=False).head(TOPK)

# print("\n\nTop K (Sorted on Predicted): ")
# print(topKpredicted)

# print("\nTop K (Sorted on True)")
# print(topKtrue)

# LOCAL_BEST_TRUE = topKtrue['True'].max()
# # get the row with the max Predicted
# BEST_PREDICTED = topKpredicted['Predicted'].idxmax()
# BEST_PREDICTED = topKpredicted.loc[BEST_PREDICTED, 'True']

# print("\n\nBest Predicted: ", BEST_PREDICTED)
# print("Local Best True: ", LOCAL_BEST_TRUE)
# GLOBAL_MAX_TRUE = all_trials[all_trials['dataset']==D]['f1'].max()
# print("Global Max True: ", GLOBAL_MAX_TRUE)
# PERFORMANCE = BEST_PREDICTED / GLOBAL_MAX_TRUE
# diff = GLOBAL_MAX_TRUE - BEST_PREDICTED
# PERFORMANCE = round(PERFORMANCE, 4)
# print("Performance: ", PERFORMANCE)
# print("Difference between Predicted and Global Best: ", round(diff, 4))

# TEST_MSE = round(TEST_MSE, 4)
# PREDICTION_RUNTIME = round(PREDICTION_RUNTIME, 4)
# END_TO_END_RUNTIME = round(END_TO_END_RUNTIME, 4)

# f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(D,
#                                                 regressor_name, 
#                                                 TEST_MSE,
#                                                 BEST_PREDICTED,
#                                                 GLOBAL_MAX_TRUE,
#                                                 PERFORMANCE,
#                                                 PREDICTION_RUNTIME,
#                                                 END_TO_END_RUNTIME))
# f.flush()

# f.close()