import pandas as pd
import os

def process_file(file_path, dataset_name):
    df = pd.read_csv(file_path, skipinitialspace=True)
    df['DATASET'] = dataset_name.split('.')[0]
    df['REGRESSOR'] = 'AutoML Ensemble {' + df['AUTOML_REGRESSOR'].str.replace('|', ' ').str.lower() + '}'

    # df['PROCESSED_REGRESSOR'] = df['AUTOML_REGRESSOR'].apply(lambda x: x.replace('_', ' ').title().replace(' ', ''))
    # unique_regressors = df['PROCESSED_REGRESSOR'].unique().tolist()
    # df['PROCESSED_REGRESSOR'] = df['PROCESSED_REGRESSOR'].apply(lambda x: x.replace(' ', '_').lower())

    # # Chhang with shorts
    # if 'RandomForest' in unique_regressors:
    #     unique_regressors[unique_regressors.index('RandomForest')] = 'RF'
    # if 'GradientBoosting' in unique_regressors:
    #     unique_regressors[unique_regressors.index('GradientBoosting')] = 'GB'
    # if 'ExtraTrees' in unique_regressors:
    #     unique_regressors[unique_regressors.index('ExtraTrees')] = 'ET'



    # formatted_string = f"AutoML Ensemble {{{', '.join(unique_regressors)}}}"
    # df['REGRESSOR'] = formatted_string
    # df.drop(columns=['PROCESSED_REGRESSOR'], inplace=True)

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

automl_files = [f for f in os.listdir('automl') if (f.endswith('.csv') and 'importance' not in f)]
print(automl_files)
automl_dfs = [process_file(os.path.join('automl', file), file) for file in automl_files]
automl_result = pd.concat(automl_dfs, ignore_index=True)

dl_files = [os.path.join('dl', f) for f in os.listdir('dl') if (f.endswith('.csv') and 'importance' not in f)]
sklearn_files = [os.path.join('sklearn', f) for f in os.listdir('sklearn') if (f.endswith('.csv') and 'importance' not in f)]

dl_dfs = [pd.read_csv(file, skipinitialspace=True) for file in dl_files]
sklearn_dfs = [pd.read_csv(file, skipinitialspace=True) for file in sklearn_files]
print(automl_dfs)
all_dfs = automl_dfs + dl_dfs + sklearn_dfs
result = pd.concat(all_dfs, ignore_index=True)

# remove rows with REGRESSOR == 'Lasso' and 'XGBRegressor'
# result = result[~result['REGRESSOR'].isin(['Lasso', 'XGBRegressor'])]

result.to_csv('autoconf_final_results.csv', index=False)
print("The files from all directories have been merged successfully.")
