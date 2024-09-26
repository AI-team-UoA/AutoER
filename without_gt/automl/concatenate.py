import pandas as pd
import glob

directory_path = './'

file_patterns = {
    'all': 'all_D*.csv',
    'optuna': 'optuna_D*.csv',
    'gridsearch': 'gridsearch_D*.csv'
}

def concatenate_csv_files(file_pattern, output_filename):
    csv_files = glob.glob(directory_path + file_pattern)
    
    if not csv_files:
        print(f"No files found for pattern: {file_pattern}")
        return
        
    concatenated_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    concatenated_df = concatenated_df.sort_values('TEST_SET')
    concatenated_df.to_csv(output_filename, index=False)
    print(f"Concatenated CSV saved to {output_filename}")

concatenate_csv_files(file_patterns['all'], 'all.csv')
concatenate_csv_files(file_patterns['optuna'], 'optuna.csv')
concatenate_csv_files(file_patterns['gridsearch'], 'gridsearch.csv')


# ------------------------------------------------------------
# ------------------------------------------------------------


directory_paths = {
    'optuna': './optuna/importance/',  # Replace with your directory path
    'all': './all/importance/',        # Replace with your directory path
    'gridsearch': './gridsearch/importance/'  # Replace with your directory path
}


# Function to concatenate all feature importance CSV files from a directory and include file name info
def concatenate_importance_files(directory_path):
    csv_files = glob.glob(directory_path + '*.csv')
    
    # Check if any files match the pattern
    if not csv_files:
        print(f"No files found in directory: {directory_path}")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found
    
    # Initialize an empty list to store DataFrames
    df_list = []

    # Read each CSV file and add its contents to the list
    for file in csv_files:
        df = pd.read_csv(file)
        df['DATASET'] = directory_path.split('/')[-3].upper()  # Using directory name as the source label
        df['TEST_SET'] = file.split('/')[-1].split('.')[0].split('_')[-1]  # Extracting the test set name from the file name
        
        # Convert column names to uppercase
        df.columns = df.columns.str.upper()
        
        # Add a rank column based on the 'IMPORTANCE' column
        # df['RANK'] = df['IMPORTANCE'].rank(ascending=False, method='dense').astype(int)
        
        df_list.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df['REGRESSOR'] = 'AutoML'
    return concatenated_df


# Concatenate files for each group
optuna_importance = concatenate_importance_files(directory_paths['optuna'])
all_importance = concatenate_importance_files(directory_paths['all'])
gridsearch_importance = concatenate_importance_files(directory_paths['gridsearch'])

# Combine all dataframes into one
combined_importance = pd.concat([optuna_importance, all_importance, gridsearch_importance], ignore_index=True)

# Save the combined feature importance data to a new CSV file
combined_importance.to_csv('./importance/automl_feature_importance.csv', index=False)
print("Combined feature importance CSV saved to 'automl_feature_importance.csv'")
