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

from sklearn.inspection import permutation_importance

import argparse
import time

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, required=True)
parser.add_argument('--d', type=str, required=False)
args = parser.parse_args()
dataset = args.trials
D_input = args.d

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------          PARAMS          ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

DATA_DIR = '../data/'
DIR = './automl/'
FILE = ''
AUTOML_PER_RUNTIME = 2*60*60
AUTOML_OVERALL_RUNTIME = 24*60*60
AUTOML_MEMORY = 6144*4
AUTOML_NJOBS = 1
TOPK = 20

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    READING DATASET       ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

trials = pd.read_csv(DATA_DIR+'trials.csv', sep=',')
all_trials = trials.copy()

if dataset == 'gridsearch':
    trials = trials[trials['sampler']=='gridsearch']
elif dataset == 'all':
    pass
else:
    trials = trials[trials['sampler']!='gridsearch']

trials = trials[trials['f1']!=0]

# Round column in 4 decimals
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

filename = DIR+dataset+'.csv' if D_input is None else DIR+dataset+'_'+D_input+'.csv'
f = open(filename, 'a')
f.write('TEST_SET, AUTOML_REGRESSOR, TEST_MSE, PREDICTED_F1, GLOBAL_BEST_F1, PERFORMANCE, PREDICTIONS_RUNTIME, OPTIMIZATION_TIME\n')
f.flush()
print("Writing to: ", filename)

for D in datasets:

    if D_input is not None:
        if D != D_input:
            continue
        else:
            D = D_input
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

    # continue
    # sys.exit(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dummy)
    X_test_scaled = scaler.transform(X_test_dummy)

    END_TO_END_RUNTIME = time.time()

    automl = autosklearn.AutoSklearnRegressor(
        time_left_for_this_task=AUTOML_OVERALL_RUNTIME,  # Total time for the AutoML process
        per_run_time_limit=AUTOML_PER_RUNTIME,       # Time for each model
        memory_limit=AUTOML_MEMORY, 
        n_jobs=AUTOML_NJOBS,
        ensemble_size=1               # Use single best model
    )

    # Fit the model
    automl.fit(X_train_scaled, y_train, dataset_name='trials_optuna_'+str(D))

    END_TO_END_RUNTIME = time.time() - END_TO_END_RUNTIME    
    PREDICTION_RUNTIME = time.time()

    # Predict using the best model
    y_pred = automl.predict(X_test_scaled)
    
    PREDICTION_RUNTIME = time.time() - PREDICTION_RUNTIME

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
    # -------------------------     FEATURE SELECTION    ----------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

    r = permutation_importance(automl, X_test_dummy, y_test, n_repeats=10, random_state=0)

    sort_idx = r.importances_mean.argsort()[::-1]

    dummy_features = X_test_dummy.columns
    
    print("\n\nFeature Importance: ")
    for i in sort_idx[::-1]:
        print(
            f"{dummy_features[i]:10s}: {r.importances_mean[i]:.3f} +/- "
            f"{r.importances_std[i]:.3f}"
        )
    print("\n\n")

    # Save the feature importance to a file
    feature_importance = pd.DataFrame({'feature': dummy_features, 'importance': r.importances_mean})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    feature_importance.to_csv(DIR+dataset+'/'+'importance/'+str(D)+'.csv', index=False)

f.close()
