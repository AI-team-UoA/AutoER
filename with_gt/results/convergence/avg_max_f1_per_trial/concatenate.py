import os
import pandas as pd

# Directory with CSV files
directory = "./"  # Adjust the directory path if needed

# Initialize an empty list to store dataframes
dataframes = []

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Split the filename to extract sampler and dataset
        sampler, dataset = filename.split('_')[0], filename.split('_')[1].replace(".csv", "")
        
        # Read the CSV file into a DataFrame
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        
        # Add 'sampler' and 'dataset' columns to the DataFrame
        df['sampler'] = sampler
        df['dataset'] = dataset
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames into one
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Rearrange the columns to match the desired order
concatenated_df = concatenated_df[['sampler', 'dataset', 'max_trials', 'f1', 'f1_ratio']]

# Save the concatenated DataFrame to a new CSV file
output_file = os.path.join(directory, "concatenated_output.csv")
concatenated_df.to_csv(output_file, index=False)

