import pandas as pd
import os

def process_automl_file(file_path, dataset_name):
    df = pd.read_csv(file_path, skipinitialspace=True)
    df['DATASET'] = dataset_name.split('.')[0]
    df['REGRESSOR'] = df['AUTOML_REGRESSOR'].str.replace('|', ' ').str.replace(r'0{3,}', '', regex=True).str.lower()

    df['VALIDATION_MSE'] = None
    df['BEST_REGRESSOR_FIT_TIME'] = None
    df.rename(columns={'PREDICTIONS_RUNTIME': 'BEST_REGRESSOR_PREDICTION_TIME'}, inplace=True)
    df = df[['TEST_SET', 'DATASET', 'REGRESSOR', 'VALIDATION_MSE', 'TEST_MSE', 'PREDICTED_F1', 'GLOBAL_BEST_F1', 'PERFORMANCE', 'OPTIMIZATION_TIME', 'BEST_REGRESSOR_FIT_TIME', 'BEST_REGRESSOR_PREDICTION_TIME', 'WITH_DATA_FEATURES', 'LM', 'K', 'CLUSTERING', 'THRESHOLD']]
    return df


directories = {
    'automl': 'automl',
    'dl': 'dl',
    'sklearn': 'sklearn'
}

automl_files = [f for f in os.listdir('automl') if (f.endswith('.csv') and 'importance' not in f and 'ablation' not in f)]
print(automl_files)
automl_dfs = [process_automl_file(os.path.join('automl', file), file) for file in automl_files]
# automl_result = pd.concat(automl_dfs, ignore_index=True)
# for df in automl_dfs:
#     df['WITH_DATA_FEATURES'] = 1


dl_files = [os.path.join('dl', f) for f in os.listdir('dl') if (f.endswith('.csv') and 'importance' not in f)]
sklearn_files = [os.path.join('sklearn', f) for f in os.listdir('sklearn') if (f.endswith('.csv') and 'importance' not in f)]
sklearn_files_ablation = [os.path.join('sklearn/ablation', f) for f in os.listdir('sklearn/ablation') if (f.endswith('.csv') and 'importance' not in f)]


# dl_dfs = [pd.read_csv(file, skipinitialspace=True) for file in dl_files]
sklearn_dfs = [pd.read_csv(file, skipinitialspace=True) for file in sklearn_files]
for df in sklearn_dfs:
    df['WITH_DATA_FEATURES'] = 1

print(sklearn_dfs)
sklearn_dfs_ablation = [pd.read_csv(file, skipinitialspace=True) for file in sklearn_files_ablation]
for df in sklearn_dfs_ablation:
    df['WITH_DATA_FEATURES'] = 0

print(automl_dfs)
all_dfs = automl_dfs + sklearn_dfs + sklearn_dfs_ablation
result = pd.concat(all_dfs, ignore_index=True)

# remove rows with REGRESSOR == 'Lasso' and 'XGBRegressor'
# result = result[~result['REGRESSOR'].isin(['Lasso', 'XGBRegressor'])]

result.to_csv('autoconf_final_results.csv', index=False)
print("The files from all directories have been merged successfully.")
