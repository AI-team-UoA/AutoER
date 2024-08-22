import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.inspection import permutation_importance

import argparse

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso, Ridge, LinearRegression

import time
import optuna

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, required=True)
parser.add_argument('--regressor', type=str, required=True)

args = parser.parse_args()
dataset = args.trials

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------          PARAMS          ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

DATA_DIR = '../data/'
DIR = './sklearn/'
TOPK = 20
OPTUNA_NUM_OF_TRIALS = 50
REGRESSORS = {
    'LASSO': Lasso, 
    'RIDGE': Ridge,
    'LINEAR': LinearRegression,
    # 'SVR': SVR, removed due to long training time
    'XGB': XGBRegressor, 
    'RF': RandomForestRegressor
}
REGRESSOR = REGRESSORS[args.regressor]
RESULTS_CSV_NAME = dataset + '_' + args.regressor
RANDOM_STATE = 42
DB_NAME = 'sqlite:///{}.db'.format(RESULTS_CSV_NAME)

if args.regressor == 'LINEAR':
    OPTUNA_NUM_OF_TRIALS = 1

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


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    EXPERIMENTS           ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

filename = DIR+RESULTS_CSV_NAME+'.csv'
f = open(filename, 'w')
f.write('TEST_SET, DATASET, REGRESSOR, VALIDATION_MSE, TEST_MSE, PREDICTED_F1, GLOBAL_BEST_F1, PERFORMANCE, OPTIMIZATION_TIME, BEST_REGRESSOR_FIT_TIME, BEST_REGRESSOR_PREDICTION_TIME\n')
f.flush()
print("Writing to: ", filename)


for D in datasets:

    STUDY_NAME = RESULTS_CSV_NAME+'_'+D
    
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
    # print("Validation Mask: ", validation_mask)
    print("Validation size: ", len(validation_mask))

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
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
        elif REGRESSOR == Lasso:
            param = {
                'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e1)
            }
        elif REGRESSOR == Ridge:
            param = {
                'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e1)
            }
        elif REGRESSOR == LinearRegression:
            # Linear Regression does not have hyperparameters to tune
            param = {}

        # Create the model
        model = REGRESSOR(**param)

        # Train the model
        print("Training with: ", X_train_final.shape, y_train_final.shape)

        model.fit(X_train_final, y_train_final)

        # Evaluate the model
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                study_name=STUDY_NAME,
                                load_if_exists=False,
                                storage=DB_NAME)
    OPTUNA_TRIALS_TIME = time.time()
    study.optimize(objective, n_trials=OPTUNA_NUM_OF_TRIALS)
    OPTUNA_TRIALS_TIME = time.time() - OPTUNA_TRIALS_TIME

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters: ", best_params)

    VALIDATION_MSE = study.best_value
    
    BEST_REGRESSOR_FIT_TIME = time.time()

    # Train the model with the best hyperparameters
    regressor = REGRESSOR(**best_params)
    regressor.fit(X_train_dummy, y_train)

    BEST_REGRESSOR_FIT_TIME = time.time() - BEST_REGRESSOR_FIT_TIME

    BEST_REGRESSOR_PREDICTION_TIME = time.time()

    # Predict using the best model
    y_pred = regressor.predict(X_test_dummy)

    BEST_REGRESSOR_PREDICTION_TIME = time.time() - BEST_REGRESSOR_PREDICTION_TIME

    regressor_name = str(regressor).split('(')[0]

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------     EVALUATION           ----------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

    print("\n\nPerformance on Test Set: ", D)
    TEST_MSE = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", TEST_MSE)
    result = X_test[['lm', 'clustering', 'k', 'threshold']]

    result['Predicted'] = y_pred
    result['True'] = y_test
    topKpredicted = result.sort_values(by='Predicted', ascending=False).head(TOPK)
    topKtrue = result.sort_values(by='True', ascending=False).head(TOPK)
    
    print("\n\nTop K (Sorted on Predicted): ")
    print(topKpredicted)

    print("\nTop K (Sorted on True)")
    print(topKtrue)

    LOCAL_BEST_TRUE = topKtrue['True'].max()
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
    print("Difference between Predicted and Best: ", round(diff, 4))

    TEST_MSE = round(TEST_MSE, 4)
    VALIDATION_MSE = round(VALIDATION_MSE, 4)
    BEST_REGRESSOR_FIT_TIME = round(BEST_REGRESSOR_FIT_TIME, 4)
    BEST_REGRESSOR_PREDICTION_TIME = round(BEST_REGRESSOR_PREDICTION_TIME, 4)
    OPTUNA_TRIALS_TIME = round(OPTUNA_TRIALS_TIME, 4)

    f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(D,
                                                              dataset,
                                                  regressor_name, 
                                                  VALIDATION_MSE,
                                                    TEST_MSE,
                                                    BEST_PREDICTED,
                                                    GLOBAL_MAX_TRUE,
                                                    PERFORMANCE,
                                                    OPTUNA_TRIALS_TIME,
                                                    BEST_REGRESSOR_FIT_TIME,
                                                    BEST_REGRESSOR_PREDICTION_TIME))
    f.flush()

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------     FEATURE SELECTION    ----------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

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

    # Save the feature importance
    feature_importance = pd.DataFrame()
    feature_importance['Feature'] = dummy_features
    feature_importance['Importance'] = r.importances_mean
    feature_importance['Std'] = r.importances_std
    feature_importance['Rank'] = np.arange(len(dummy_features))

    feature_importance.to_csv(DIR+RESULTS_CSV_NAME+'_'+D+'_feature_importance.csv', index=False)

f.close()
