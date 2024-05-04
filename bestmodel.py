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

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()
dataset = args.dataset

# Params

DIR = 'predictions/'
FILE = 'automl'
AUTOML_PER_RUNTIME = 5*60
AUTOML_OVERALL_RUNTIME = 10*60
AUTOML_MEMORY = 6144
AUTOML_NJOBS = 1
TOPK = 20

def evaluate(y_true, y_pred):
    idx = np.argsort(y_pred)
    best_predicted = y_true.iloc[idx[-1]]
    max_y = np.max(y_true)
    div = (best_predicted / max_y)
    diff = max_y - best_predicted
    return div.values[0], diff.values[0]

trials = pd.read_csv('trials.csv', sep=',')

if dataset == 'gridsearch':
    trials = trials[trials['sampler']=='gridsearch']
elif dataset == 'all':
    pass
else:
    trials = trials[trials['sampler']!='gridsearch']

trials.drop_duplicates(inplace=True)
dataset_specs = pd.read_csv('dataset_specs.csv', sep=',')
datasets = dataset_specs['dataset'].unique()
trials = pd.merge(trials, dataset_specs, on='dataset')
trials = trials[trials['f1']!=0]

features = ['clustering', 'lm', 'k', 'threshold', 'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
            'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity', 
            'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
            'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity']
trials = trials[features + ['f1', 'dataset']]

trials.to_csv('trials_optuna_clean.csv', sep=',', index=False)

filename = DIR+FILE+'_'+dataset+'.csv'
f = open(filename, 'w')
f.write('TEST_SET, AUTOML_MODEL, R2, MAE, MSE, PERFROMANCE, DIFF_FROM_MAX\n')
f.flush()
print("Writing to: ", filename)

for D in datasets:
    print("TEST SET: ", D)
    print("TRAINING with: ", [x for x in datasets if x!=D])

    testD = trials[trials['dataset']==D]
    trainD = trials[trials['dataset']!=D]
    trainDatasets = [x for x in datasets if x!=D]

    y_train = trainD[['f1']]
    X_train = trainD[features]

    X_test = testD[features]
    y_test = testD[['f1']]

    X_train_dummy = pd.get_dummies(X_train)
    X_test_dummy = pd.get_dummies(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dummy)
    X_test_scaled = scaler.transform(X_test_dummy)

    automl = autosklearn.AutoSklearnRegressor(
        time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
        per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
        memory_limit=AUTOML_MEMORY, 
        n_jobs=AUTOML_NJOBS,
        ensemble_size=1               # Use single best model
    )

    # Fit the model
    automl.fit(X_train_scaled, y_train, dataset_name='trials_optuna_'+str(D))

    # Predict using the best model
    y_pred = automl.predict(X_test_scaled)

    # Evaluate the predictions
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Evaluate the predictions
    # print("Best Predicted distance from the True best configuration: ", evaluate(y_test, y_pred))
    ensemble = automl.get_models_with_weights()
    for weight, model in ensemble:
        model_configuration = model.get_params()
        regressor_name = model_configuration['config']['regressor:__choice__']

    result = X_test[['lm', 'clustering', 'k', 'threshold']]

    # add y_pred and y_test to res
    result['Predicted'] = y_pred
    result['True'] = y_test
    topKpredicted = result.sort_values(by='Predicted', ascending=False).head(TOPK)
    topKtrue = result.sort_values(by='True', ascending=False).head(TOPK)
    
    print("Top K Predicted: ")
    print(topKpredicted)

    print("Top K True: ")
    print(topKtrue)


    # Display the details of the best model
    print(automl.show_models())

    performance, diff = evaluate(y_test, y_pred)
    print("\n\nPerformance: ", performance)
    print("Difference between Predicted and Best: ", diff)
    f.write("{}, {}, {}, {}, {}, {}, {}\n".format(D, 
                                              regressor_name, 
                                              r2_score(y_test, y_pred), 
                                              mean_absolute_error(y_test, y_pred), 
                                              mean_squared_error(y_test, y_pred), 
                                              performance,
                                              diff))
    f.flush()

    # EXPLAIN FEATURE SELECTION
    r = permutation_importance(automl, X_test_dummy, y_test, n_repeats=10, random_state=0)

    sort_idx = r.importances_mean.argsort()[::-1]

    dummy_features = X_test_dummy.columns

    for i in sort_idx[::-1]:
        print(
            f"{dummy_features[i]:10s}: {r.importances_mean[i]:.3f} +/- "
            f"{r.importances_std[i]:.3f}"
        )
