import pandas as pd
import glob

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Concatenate CSV files')
parser.add_argument('--exp', type=str, help='Path to the directory containing CSV files')
args = parser.parse_args()

exp = args.exp


RESULTS_DIR = './results/'
EXP_DIR = RESULTS_DIR + exp
print(f"Concatenating CSV files for {exp}...")

scores_file_patterns = 'D*.csv'

def concatenate_csv_files(exp_type):
    path_exploring = EXP_DIR + "/" + exp_type + "/scores/" + scores_file_patterns
    csv_files = glob.glob(path_exploring)
    
    if not csv_files:
        print(f"No files found for path: {path_exploring}")
        return
    print(f"Concatenating {len(csv_files)} CSV files for {exp_type}...")
        
    concatenated_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    concatenated_df = concatenated_df.sort_values('TEST_SET')
    concatenated_df.to_csv(exp_type+'.csv', index=False)
    print(f"Concatenated CSV saved to {exp_type+'.csv'}")

# Concatenate files for each group

concatenate_csv_files('all')
concatenate_csv_files('optuna')
concatenate_csv_files('gridsearch')


# ------------------------------------------------------------
# ------------------------------------------------------------

def concatenate_importance_files(exp_type):
    path_exploring = EXP_DIR + "/" + exp_type + "/importance/" + scores_file_patterns

    csv_files = glob.glob(path_exploring)

    if not csv_files:
        print(f"No files found in directory: {path_exploring}")
        return pd.DataFrame()
    
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        df['DATASET'] = exp_type
        df['TEST_SET'] = file.split('/')[-1].split('.')[0].split('_')[-1]
        df.columns = df.columns.str.upper()
        df_list.append(df)
    
    concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df['REGRESSOR'] = 'AutoML'
    return concatenated_df


optuna_importance = concatenate_importance_files('optuna')
all_importance = concatenate_importance_files('all')
gridsearch_importance = concatenate_importance_files('gridsearch')

combined_importance = pd.concat([optuna_importance, all_importance, gridsearch_importance], ignore_index=True)

combined_importance.to_csv('./automl_feature_importance.csv', index=False)
print("Combined feature importance CSV saved to 'automl_feature_importance.csv'")
