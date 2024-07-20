import pandas as pd
import os

def combine_csv_files(directories):
    df_list = []

    for directory in directories:
        print("Path: ", os.getcwd())

        print(f"Reading files in {directory}")
        files = [file for file in os.listdir(os.path.join('../results', directory)) if file.endswith('.csv')]
        
        for file in files:
            print(f"Reading {file}")
            file_path = os.path.join(os.path.join('../results', directory), file)
            df = pd.read_csv(file_path)
            df_list.append(df)

    print("Combining all files into one dataframe")
    combined_df = pd.concat(df_list, ignore_index=True)
    print(combined_df.head())

    combined_df.to_csv('trials.csv', index=False)
    print("All files have been combined into trials.csv")

directories = ['qmc', 'tpe', 'random', 'gps', 'gridsearch']

combine_csv_files(directories)
