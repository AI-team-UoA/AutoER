import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from sklearn.model_selection import KFold

import time
import optuna

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
DIR = './dl/'
TOPK = 20
OPTUNA_NUM_OF_TRIALS = 50
REGRESSOR = 'NN'
RESULTS_CSV_NAME = dataset
RANDOM_STATE = 42
DB_NAME = 'sqlite:///{}.db'.format(RESULTS_CSV_NAME)
VALIDATION_SET_FRACTION = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    if D_input is not None:
        if D != D_input:
            continue
        else:
            D = D_input
            print("Processing: ", D)

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

    for D_train in trainDatasets:
        indices = trainD[trainD['dataset'] == D_train].sample(frac=VALIDATION_SET_FRACTION, random_state=RANDOM_STATE).index
        validation_mask[indices] = True
    
    print("Validation size: ", len(trainD[validation_mask]), "Training size: ", len(trainD[~validation_mask]))

    validation_set = trainD[validation_mask]
    train_set = trainD[~validation_mask]

    # Convert to NumPy arrays after subsetting
    X_train_final = X_train_dummy.loc[train_set.index]
    X_val = X_train_dummy.loc[validation_set.index]
    y_train_final = trainD.loc[train_set.index, 'f1']
    y_val = trainD.loc[validation_set.index, 'f1']

    X_train_tensor = torch.tensor(X_train_final.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_final.values, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test_dummy.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.sigmoid(out)
            return out * 100

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        num_epochs = trial.suggest_int('num_epochs', 2, 50)

        # print("Hidden Dim: ", hidden_dim)
        # print("Learning Rate: ", lr)
        # print("Number of Epochs: ", num_epochs)
        

        model = SimpleNN(input_dim=X_train_final.shape[1], hidden_dim=hidden_dim, output_dim=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10

        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            # print(f"Epoch {epoch}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                # print(f"Early stopping at epoch {epoch}")
                break

        return best_val_loss
    
    OPTUNA_TRIALS_TIME = time.time()
    study = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                study_name=STUDY_NAME,
                                storage=DB_NAME,
                                load_if_exists=True)
    study.optimize(objective, n_trials=OPTUNA_NUM_OF_TRIALS)
    OPTUNA_TRIALS_TIME = time.time() - OPTUNA_TRIALS_TIME

    best_params = study.best_params
    print("Best Hyperparameters: ", best_params)
    print("Best validation loss: ", study.best_value)

    VALIDATION_MSE = study.best_value
    
    best_validation_loss = study.best_value
    best_hidden_dim = best_params['hidden_dim']
    best_lr = best_params['lr']
    best_epochs = best_params['num_epochs']

    BEST_REGRESSOR_FIT_TIME = time.time()

    model = SimpleNN(input_dim=X_train_final.shape[1], hidden_dim=best_hidden_dim, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10

    for epoch in range(best_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        # print(f"Epoch {epoch}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            # print(f"Early stopping at epoch {epoch}")
            break

    BEST_REGRESSOR_FIT_TIME = time.time() - BEST_REGRESSOR_FIT_TIME

    BEST_REGRESSOR_PREDICTION_TIME = time.time()

    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch in test_loader:
            outputs = model(X_batch[0])
            y_pred_list.append(outputs.cpu().numpy())
    
    y_pred = np.concatenate(y_pred_list, axis=0)

    BEST_REGRESSOR_PREDICTION_TIME = time.time() - BEST_REGRESSOR_PREDICTION_TIME


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
                                                  REGRESSOR, 
                                                  VALIDATION_MSE,
                                                    TEST_MSE,
                                                    BEST_PREDICTED,
                                                    GLOBAL_MAX_TRUE,
                                                    PERFORMANCE,
                                                    OPTUNA_TRIALS_TIME,
                                                    BEST_REGRESSOR_FIT_TIME,
                                                    BEST_REGRESSOR_PREDICTION_TIME))
    f.flush()
f.close()