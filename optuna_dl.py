import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from optuna.integration import PyTorchLightningPruningCallback
import argparse
from sklearn.model_selection import KFold

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
REGRESSOR = args.regressor
RESULTS_CSV_NAME = csv_name+'_'+args.regressor+'_'+dataset
RANDOM_STATE = 42
DB_NAME = 'sqlite:///{}.db'.format(RESULTS_CSV_NAME)
regressor_name = args.regressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# trials.to_csv(RESULTS_CSV_NAME, sep=',', index=False)

filename = DIR+RESULTS_CSV_NAME+'.csv'
f = open(filename, 'w')
f.write('TEST_SET, REGRESSOR, R2, MAE, MSE, BEST_VALIDATION_LOSS, AUTOCONF_METRIC, DIFF_FROM_MAX\n')
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
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out * 100

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        num_epochs = trial.suggest_int('num_epochs', 2, 50)

        print("Hidden Dim: ", hidden_dim)
        print("Learning Rate: ", lr)
        print("Number of Epochs: ", num_epochs)
        

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
            print(f"Epoch {epoch}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return best_val_loss
    
    study = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                study_name=STUDY_NAME,
                                storage=DB_NAME,
                                load_if_exists=True)
    study.optimize(objective, n_trials=OPTUNA_NUM_OF_TRIALS)
    best_params = study.best_params
    print("Best Hyperparameters: ", best_params)
    print("Best validation loss: ", study.best_value)
    
    best_validation_loss = study.best_value
    best_hidden_dim = best_params['hidden_dim']
    best_lr = best_params['lr']
    best_epochs = best_params['num_epochs']

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
        print(f"Epoch {epoch}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch in test_loader:
            outputs = model(X_batch[0])
            y_pred_list.append(outputs.cpu().numpy())
    
    y_pred = np.concatenate(y_pred_list, axis=0)


    print("\n\nPerformance on Test Set: ", D)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))
    print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred))


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
    performance = round(performance[0], 4)
    diff = round(diff[0], 4)
    print("\n\nPerformance: ", performance)
    print("Difference between Predicted and Best: ", diff)
    f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(D, 
                                              regressor_name, 
                                              r2_score(y_test, y_pred), 
                                              mean_absolute_error(y_test, y_pred), 
                                              mean_squared_error(y_test, y_pred), 
                                              best_validation_loss,
                                              performance,
                                              diff))
    f.flush()
f.close()