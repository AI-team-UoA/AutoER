import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.inspection import permutation_importance

import argparse
import xgboost as xgb

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso, Ridge, LinearRegression

import time
import optuna
import random

# read args
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=str, required=True)
parser.add_argument('--regressor', type=str, required=True)
parser.add_argument('--with_data_features', type=int, required=True, default=1)

args = parser.parse_args()
dataset = args.trials
with_data_features = args.with_data_features

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------  REPRODUCIBILITY SEED    ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
from sklearn.utils import check_random_state
random_state = check_random_state(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
xgb_rng = np.random.RandomState(RANDOM_STATE)

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
RESULTS_CSV_NAME = dataset + '_' + args.regressor + ('_ablation' if not with_data_features else '')
DB_NAME = 'sqlite:///{}.db'.format(RESULTS_CSV_NAME)

if args.regressor == 'LINEAR':
    OPTUNA_NUM_OF_TRIALS = 1

if not with_data_features:
    DIR = DIR + 'ablation/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    print("Created: ", DIR)


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
# trials['f1'] = trials['f1'].round(4)
# trials['threshold'] = trials['threshold'].round(4)

dataset_specs = pd.read_csv(DATA_DIR+'dataset_specs.csv', sep=',')
dataset_specs_features = dataset_specs.columns.tolist()
dataset_specs_features.remove('dataset')

datasets = dataset_specs['dataset'].unique()
datasets = list(datasets)

if 'dbpedia' in datasets:
    datasets.remove('dbpedia')

if with_data_features:    
    trials = pd.merge(trials, dataset_specs, on='dataset')

trials.drop_duplicates(inplace=True)

features = ['clustering', 'lm', 'k', 'threshold'] 
features = features + dataset_specs_features  if with_data_features else features 

trials = trials[features + ['f1', 'dataset']]

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------    EXPERIMENTS           ----------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- # 

filename = DIR+RESULTS_CSV_NAME+'.csv'
aggregated_results_csv_file = open(filename, 'w')
aggregated_results_csv_file.write('TEST_SET, DATASET, REGRESSOR, VALIDATION_MSE, TEST_MSE, PREDICTED_F1, GLOBAL_BEST_F1, PERFORMANCE, OPTIMIZATION_TIME, BEST_REGRESSOR_FIT_TIME, BEST_REGRESSOR_PREDICTION_TIME, LM, K, CLUSTERING, THRESHOLD\n')
aggregated_results_csv_file.flush()

print("Writing to: ", filename)

for D in datasets:
    # if D!='D2':
    #     continue
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

        # if REGRESSOR == LinearRegression:
        #     weights = model.coef_
        #     intercepts = model.intercept_

        #     print("Weights: ", weights)
        #     print("Intercept: ", model.intercept_)

        #     # export
        #     if not os.path.exists(DIR+'weights'):
        #         os.makedirs(DIR+'weights/')
            
        #     with open(DIR+'weights/'+RESULTS_CSV_NAME+'_'+D+'.csv', 'w') as f:
        #         f.write("Feature,Weight\n")
        #         f.write("Intercept,{}\n".format(intercepts))
        #         for i, w in enumerate(weights):
        #             f.write("{},{}\n".format(X_train_dummy.columns[i], w))
                    

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

    if REGRESSOR == RandomForestRegressor:
        feature_names = X_train_dummy.columns
        feature_importances = regressor.feature_importances_
        print("Feature Importances: ", feature_importances)

        # print("Estimators: ", model.estimator_)
        # print("N Estimators: ", model.estimators_)
        # print("N Features: ", model.n_features_in_)

        # Create directory to save weights if it doesn't exist
        if not os.path.exists(DIR + 'weights'):
            os.makedirs(DIR + 'weights/')
        
        # Export feature importances
        with open(DIR + 'weights/' + RESULTS_CSV_NAME + '_' + D + '.csv', 'w') as importance_csv_file:
            importance_csv_file.write("Feature,Importance\n")
            for i, importance in enumerate(feature_importances):
                importance_csv_file.write("{},{}\n".format(X_train_dummy.columns[i], importance))

                # Select a single sample to trace its decision path
        # sample = X[0].reshape(1, -1)

        # # Access the trees in the random forest
        # for tree_idx, tree in enumerate(regressor.estimators_):
        #     print(f"Decision Path for Tree {tree_idx + 1}:")
            
        #     # Get the decision path for the sample
        #     decision_path = tree.decision_path(X_train_dummy)
        #     node_indicator = decision_path.toarray()
            
        #     # Get feature thresholds and feature indices used in splits
        #     feature = tree.tree_.feature
        #     threshold = tree.tree_.threshold

        # # Print the path with attribute names
        #     for node_id in np.where(node_indicator[0] == 1)[0]:
        #         if threshold[node_id] != -2:  # -2 indicates a leaf node
        #             feature_name = feature_names[feature[node_id]] if feature[node_id] != -2 else "Leaf Node"
        #             print(f"Node {node_id}: Split on '{feature_name}' "
        #                 f"at threshold {threshold[node_id]}")
        #         else:
        #             print(f"Node {node_id}: Leaf Node")
            
        #     print("\n")

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
    # save topKpredicted
    topKpredicted.to_csv(DIR+'temp/'+RESULTS_CSV_NAME+'_'+D+'_predicted.csv', index=False)

    print("\nTop K (Sorted on True)")
    print(topKtrue)

    LOCAL_BEST_TRUE = topKtrue['True'].max()
    BEST_PREDICTED = topKpredicted['Predicted'].idxmax()
    BEST_PREDICTED = topKpredicted.loc[BEST_PREDICTED, 'True']

    print("\nConfiguartion predicted as the best: ")
    print("LM: ", topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'lm'])
    BEST_LM = topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'lm']
    print("K: ", topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'k'])
    BEST_K = topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'k']
    print("Clustering: ", topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'clustering'])
    BEST_CLUSTERING = topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'clustering']
    print("Threshold: ", topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'threshold'])
    BEST_THRESHOLD = topKpredicted.loc[topKpredicted['Predicted'].idxmax(), 'threshold']

    print("\n\nBest Predicted: ", BEST_PREDICTED)
    print("Local Best True: ", LOCAL_BEST_TRUE)
    GLOBAL_MAX_TRUE = all_trials[all_trials['dataset']==D]['f1'].max()
    print("Global Max True: ", GLOBAL_MAX_TRUE)
    PERFORMANCE = BEST_PREDICTED / GLOBAL_MAX_TRUE
    diff = GLOBAL_MAX_TRUE - BEST_PREDICTED
    PERFORMANCE = round(PERFORMANCE, 4)
    print("Performance: ", PERFORMANCE)
    print("Difference between Predicted and Best: ", round(diff, 4))

    # TEST_MSE = round(TEST_MSE, 4)
    # VALIDATION_MSE = round(VALIDATION_MSE, 4)
    # BEST_REGRESSOR_FIT_TIME = round(BEST_REGRESSOR_FIT_TIME, 4)
    # BEST_REGRESSOR_PREDICTION_TIME = round(BEST_REGRESSOR_PREDICTION_TIME, 4)
    # OPTUNA_TRIALS_TIME = round(OPTUNA_TRIALS_TIME, 4)

    aggregated_results_csv_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(D,
                                                              dataset,
                                                  regressor_name, 
                                                  VALIDATION_MSE,
                                                    TEST_MSE,
                                                    BEST_PREDICTED,
                                                    GLOBAL_MAX_TRUE,
                                                    PERFORMANCE,
                                                    OPTUNA_TRIALS_TIME,
                                                    BEST_REGRESSOR_FIT_TIME,
                                                    BEST_REGRESSOR_PREDICTION_TIME,
                                                    BEST_LM,
                                                    BEST_K,
                                                    BEST_CLUSTERING,
                                                    BEST_THRESHOLD))
    aggregated_results_csv_file.flush()

    print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(D,
                                                              dataset,
                                                            regressor_name, 
                                                            VALIDATION_MSE,
                                                                TEST_MSE,
                                                                BEST_PREDICTED,
                                                                GLOBAL_MAX_TRUE,
                                                                PERFORMANCE,
                                                                OPTUNA_TRIALS_TIME,
                                                                BEST_REGRESSOR_FIT_TIME,
                                                                BEST_REGRESSOR_PREDICTION_TIME,
                                                                BEST_LM,
                                                                BEST_K,
                                                                BEST_CLUSTERING,
                                                                BEST_THRESHOLD))

    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------     FEATURE SELECTION    ----------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- # 

    # r = permutation_importance(regressor, X_test_dummy, y_test, n_repeats=10, random_state=0)

    # sort_idx = r.importances_mean.argsort()[::-1]

    # dummy_features = X_test_dummy.columns
    
    # print("\n\nFeature Importance: ")
    # for i in sort_idx[::-1]:
    #     print(
    #         f"{dummy_features[i]:10s}: {r.importances_mean[i]:.3f} +/- "
    #         f"{r.importances_std[i]:.3f}"
    #     )
    # print("\n\n")

    # # Save the feature importance
    # feature_importance = pd.DataFrame()
    # feature_importance['Feature'] = dummy_features
    # feature_importance['Importance'] = r.importances_mean
    # feature_importance['Std'] = r.importances_std
    # feature_importance['Rank'] = np.arange(len(dummy_features))

    # IMPORTANCE_DIR = DIR+'importance/'
    # if not os.path.exists(IMPORTANCE_DIR):
    #     os.makedirs(IMPORTANCE_DIR)

    # feature_importance.to_csv(DIR+'importance/'+RESULTS_CSV_NAME+'_'+D+".csv", index=False)

aggregated_results_csv_file.close()
