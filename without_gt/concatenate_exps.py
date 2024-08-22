import pandas as pd
import os

def process_file(file_path, dataset_name):
    df = pd.read_csv(file_path, skipinitialspace=True)
    df['DATASET'] = dataset_name.split('.')[0]
    df['REGRESSOR'] = 'automl_' + df['AUTOML_REGRESSOR']
    df['VALIDATION_MSE'] = None
    df['BEST_REGRESSOR_FIT_TIME'] = None
    df.rename(columns={'PREDICTIONS_RUNTIME': 'BEST_REGRESSOR_PREDICTION_TIME'}, inplace=True)
    df = df[['TEST_SET', 'DATASET', 'REGRESSOR', 'VALIDATION_MSE', 'TEST_MSE', 'PREDICTED_F1', 'GLOBAL_BEST_F1', 'PERFORMANCE', 'OPTIMIZATION_TIME', 'BEST_REGRESSOR_FIT_TIME', 'BEST_REGRESSOR_PREDICTION_TIME']]
    return df

directories = {
    'automl': 'automl',
    'dl': 'dl',
    'sklearn': 'sklearn'
}

automl_files = [f for f in os.listdir('automl') if f.endswith('.csv')]
automl_dfs = [process_file(os.path.join('automl', file), file) for file in automl_files]
automl_result = pd.concat(automl_dfs, ignore_index=True)

dl_files = [os.path.join('dl', f) for f in os.listdir('dl') if f.endswith('.csv')]
sklearn_files = [os.path.join('sklearn', f) for f in os.listdir('sklearn') if f.endswith('.csv')]

dl_dfs = [pd.read_csv(file, skipinitialspace=True) for file in dl_files]
sklearn_dfs = [pd.read_csv(file, skipinitialspace=True) for file in sklearn_files]
print(automl_dfs)
all_dfs = automl_dfs + dl_dfs + sklearn_dfs
result = pd.concat(all_dfs, ignore_index=True)

result.to_csv('autoconf_final_results.csv', index=False)
print("The files from all directories have been merged successfully.")
