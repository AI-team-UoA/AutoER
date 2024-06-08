import pandas as pd
import os

def combine_csv_files(directories):
    # List to hold dataframes
    df_list = []

    # Loop through each directory
    for directory in directories:
        # List all csv files in the directory
        print("Path: ", os.getcwd())

        print(f"Reading files in {directory}")
        files = [file for file in os.listdir(os.path.join('results', directory)) if file.endswith('.csv')]
        
        # Read each file and append to the list
        for file in files:
            print(f"Reading {file}")
            file_path = os.path.join(os.path.join('results', directory), file)
            df = pd.read_csv(file_path)
            df_list.append(df)

    # Concatenate all dataframes into one
    print("Combining all files into one dataframe")
    combined_df = pd.concat(df_list, ignore_index=True)
    print(combined_df.head())

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv('dataset_trials.csv', index=False)
    print("All files have been combined into combined_records.csv")

# List of directories to process
directories = ['qmc', 'tpe', 'random', 'gridsearch']

# Call the function
combine_csv_files(directories)