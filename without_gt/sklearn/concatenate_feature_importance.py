import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the CSV files
directory = "weights"  # Change this to your actual path
output_file = "sklearn_feature_importance.csv"

# List all CSV files matching the pattern
csv_files = glob.glob(os.path.join(directory, "*_RF_*.csv"))

# Initialize an empty DataFrame
merged_df = pd.DataFrame()

for file in csv_files:
    # Extract dataset and test set from filename
    filename = os.path.basename(file)
    dataset, test_set = filename.split("_RF_")
    test_set = test_set.replace(".csv", "")
    
    # Read CSV file
    df = pd.read_csv(file)
    
    # Add new columns
    df["DATASET"] = dataset
    df["REGRESSOR"] = "RIDGE"
    df["TEST_SET"] = test_set
    
    # Rename columns to uppercase
    df.columns = ["FEATURE", "IMPORTANCE", "DATASET", "REGRESSOR", "TEST_SET"]
    
    # Append to merged DataFrame
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV saved to {output_file}")