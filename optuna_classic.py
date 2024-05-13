import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.inspection import plot_partial_dependence, permutation_importance

import argparse

# import regressors like SVR, XGBoost and Random Forest

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import optuna

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, required=True)
parser.add_argument('--csv_name', type=str, required=True)
parser.add_argument('--regressor', type=str, required=True)

args = parser.parse_args()
dataset = args.trials
csv_name = args.csv_name

# Params

DIR = 'predictions_optuna/'
TOPK = 20
OPTUNA_NUM_OF_TRIALS = 1
REGRESSORS = {'SVR': SVR, 'XGB': XGBRegressor, 'RF': RandomForestRegressor}
REGRESSOR = REGRESSORS[args.regressor]
RESULTS_CSV_NAME = csv_name
RANDOM_STATE = 42
STUDY_NAME = 'classic_autoconf'
DB_NAME = 'sqlite:///{}.db'.format(STUDY_NAME)

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

trials = trials[trials['f1']!=0]

# Round column in 4 decimals
trials['f1'] = trials['f1'].round(4)
trials['threshold'] = trials['threshold'].round(4)

dataset_specs = pd.read_csv('dataset_specs.csv', sep=',')
datasets = dataset_specs['dataset'].unique()
trials = pd.merge(trials, dataset_specs, on='dataset')

trials.drop_duplicates(inplace=True)

features = ['clustering', 'lm', 'k', 'threshold', 'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
            'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity', 
            'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
            'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity']
trials = trials[features + ['f1', 'dataset']]

trials.to_csv(RESULTS_CSV_NAME, sep=',', index=False)

filename = DIR+RESULTS_CSV_NAME+'_'+dataset+'.csv'
f = open(filename, 'w')
f.write('TEST_SET, REGRESSOR, R2, MAE, MSE, AUTOCONF_METRIC, DIFF_FROM_MAX\n')
f.flush()
print("Writing to: ", filename)

for D in datasets:
    print("\n\n-----------------------------------\n")
    print("TEST SET: ", D)
    print("TRAINING with: ", [x for x in datasets if x!=D])

    testD = trials[trials['dataset']==D]
    trainD = trials[trials['dataset']!=D]
    trainDatasets = [x for x in datasets if x!=D]

    print("\n\nNumber of entities")
    
    X_train = trainD[features]
    y_train = trainD[['f1']]
    print("Train Size: ", len(X_train))

    X_test = testD[features]
    y_test = testD[['f1']]
    print("Test Size: ", len(X_test))

    X_train_dummy = pd.get_dummies(X_train, drop_first=True)
    X_test_dummy = pd.get_dummies(X_test, drop_first=True)

    if D == 'D3' and dataset == 'gridsearch':
        X_train_dummy = X_train_dummy.drop(columns=['lm_sent_glove'])

    X_train_dummy, X_test_dummy = X_train_dummy.align(X_test_dummy, join='outer', axis=1, fill_value=0)

    # Scaling
    scaler = StandardScaler()
    X_train_dummy[:] = scaler.fit_transform(X_train_dummy)
    X_test_dummy[:] = scaler.transform(X_test_dummy)

    # Splitting into validation and training sets using boolean indexing
    validation_set = pd.DataFrame()
    validation_mask = pd.Series(False, index=trainD.index)
    print("Validation Mask: ", validation_mask)

    for D_train in trainDatasets:
        indices = trainD[trainD['dataset'] == D_train].sample(frac=0.1, random_state=42).index
        validation_mask[indices] = True
    
    validation_set = trainD[validation_mask]
    train_set = trainD[~validation_mask]

    # Convert to NumPy arrays after subsetting
    X_train_final = X_train_dummy.loc[train_set.index]
    X_val = X_train_dummy.loc[validation_set.index]
    y_train_final = trainD.loc[train_set.index, 'f1']
    y_val = trainD.loc[validation_set.index, 'f1']

    # Using optuna to find the best hyperparameters

    def objective(trial):

        # Define the search space
        if REGRESSOR == SVR:
            param = {
                'C': trial.suggest_loguniform('C', 1e-2, 1e2),
                'epsilon': trial.suggest_loguniform('epsilon', 1e-2, 1e2),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'degree': trial.suggest_int('degree', 2, 5),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
        elif REGRESSOR == XGBRegressor:
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
                'gamma': trial.suggest_int('gamma', 0, 5),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 1e2),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 1e2)
            }
        elif REGRESSOR == RandomForestRegressor:
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }

        # Create the model
        model = REGRESSOR(**param)

        # Train the model
        model.fit(X_train_scaled, y_train.values.ravel())

        # Evaluate the model
        y_pred = model.predict(X_val_scaled)
        return mean_squared_error(y_val, y_pred)
    
    STUDY_NAME += '_'+D
    study = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                study_name=STUDY_NAME,
                                db_name=DB_NAME)
    study.optimize(objective, n_trials=OPTUNA_NUM_OF_TRIALS)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters: ", best_params)

    # Train the model with the best hyperparameters
    regressor = REGRESSOR(**best_params)
    regressor.fit(X_train_scaled, y_train.values.ravel())

    # Predict using the best model
    y_pred = regressor.predict(X_test_scaled)

    regressor_name = str(regressor).split('(')[0]

    # Evaluate the predictions
    print("\n\nPerformance on Test Set: ", D)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

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
    r = permutation_importance(regressor, X_test_dummy, y_test, n_repeats=10, random_state=0)

    sort_idx = r.importances_mean.argsort()[::-1]

    dummy_features = X_test_dummy.columns
    
    print("\n\nFeature Importance: ")
    for i in sort_idx[::-1]:
        print(
            f"{dummy_features[i]:10s}: {r.importances_mean[i]:.3f} +/- "
            f"{r.importances_std[i]:.3f}"
        )
    print("\n\n")