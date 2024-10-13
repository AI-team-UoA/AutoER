import time
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering, UniqueMappingClustering, KiralyMSMApproximateClustering, BestMatchClustering
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Create test data for benchmarking')
parser.add_argument('--topk', type=int, default=1, help='Number of top rows to consider')
parser.add_argument('--datajson', type=str)
parser.add_argument('--confcsv', type=str)
args = parser.parse_args()

datajson = args.datajson
confcsv = args.confcsv
TOPK = args.topk


print("Data JSON: ", datajson)
print("Configuration CSV: ", confcsv)
print("Top K: ", TOPK)

import json

with open(datajson) as f:
    test_dataset = json.load(f)

print("JSON Data: ")
print(test_dataset)

dataset = test_dataset['name']
print("Dataset: ", dataset)
format = test_dataset['format']
print("Format: ", format)

print("Time started: ", time.ctime())

RESULTS_DIR = './results'
EVALUATED_DIR = 'real_f1s'

# ------------------------------- CLUSTERING ------------------------------- #
CLUSTERING_MAPPING = {
    "ConnectedComponentsClustering": ConnectedComponentsClustering,
    "UniqueMappingClustering": UniqueMappingClustering,
    "BestMatchClustering": BestMatchClustering,
    "KiralyMSMApproximateClustering": KiralyMSMApproximateClustering,
    "ConnectedComponentsClustering": ConnectedComponentsClustering
}

verbose=True

# ------------------------------- DATA ------------------------------- #

predictions = pd.read_csv(confcsv)

print("Predictions: ")
print(predictions.head())


results_df = pd.DataFrame(columns=['dataset', 'lm', 'k', 'clustering', 'threshold', 'f1', 'precision', 'recall', 'pyjedai_runtime'])

print("\n\nRunning for: ", dataset)
start_time = time.time()
print("Time started: ", time.ctime())
dataset_predictions = predictions[predictions['dataset'] == dataset]
true_f1s = []

data = Data(
    dataset_1=pd.read_csv(f"../data/{test_dataset['dir']}/{test_dataset['d1']}.{test_dataset['format']}", 
                                  sep=test_dataset['separator'], 
                                  engine=test_dataset['engine']).astype(str),
    id_column_name_1=test_dataset['d1_id'],
    dataset_name_1=test_dataset['d1'],
    dataset_2=pd.read_csv(f"../data/{test_dataset['dir']}/{test_dataset['d2']}.{test_dataset['format']}",
                          sep=test_dataset['separator'], 
                          engine=test_dataset['engine']).astype(str),
    id_column_name_2=test_dataset['d2_id'],
    dataset_name_2=test_dataset['d2'],
    ground_truth=pd.read_csv(f"../data/{test_dataset['dir']}/{test_dataset['gt']}.{test_dataset['format']}", 
                                 sep=test_dataset['separator']).astype(str)
)

top_rows = dataset_predictions.nlargest(TOPK, 'predicted_f1')

# Loop over each row in the top K rows
for index, row in top_rows.iterrows():
    print("Max predicted regressor score: ", row['predicted_f1'])
    
    lm = row['lm'].replace(' ', '')
    k = int(row['k'])
    clustering_method = row['clustering'].replace(' ', '')
    threshold = float(row['threshold'])
    print("Running for: ", dataset, lm, k, clustering_method, threshold)
    
    print('Proposed configuration: ')
    print('LM: ', lm)
    print('K: ', k)
    print('Clustering: ', clustering_method)
    print('Threshold: ', threshold)

    print('\nBuilding blocks...')
    print("Time started: ", time.ctime())
    emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
    blocks, g = emb.build_blocks(data,
                                top_k=k,
                                load_embeddings_if_exist=True,
                                save_embeddings=False,
                                tqdm_disable=False,
                                verbose=True,
                                with_entity_matching=True)
    # print("Device used: ", emb.device)    
    # emb.evaluate(blocks, verbose=verbose)
    print('Time ended: ', time.ctime())
    print('Finished building blocks')

    print('\nClustering...')        
    print("Time started: ", time.ctime())
    ccc = CLUSTERING_MAPPING[clustering_method]()
    clusters = ccc.process(g, data, similarity_threshold=threshold)
    results = ccc.evaluate(clusters, with_classification_report=False, verbose=verbose)
    print('Finished clustering at: ', time.ctime())

    t2 = time.time()
    runtime = t2 - start_time
    f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']
    
    f1 = round(f1, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    runtime = round(runtime, 4)
    
    print("\nF1: ", f1)
    print("Time: ", runtime)
    
    new_data = pd.DataFrame({
        'dataset': [dataset],
        'lm': [lm],
        'k': [k],
        'clustering': [clustering_method],
        'threshold': [threshold],
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'pyjedai_runtime': [runtime]
    })

    print('Execution finished:')
    print(new_data)
    
    results_df = pd.concat([results_df, new_data], ignore_index=True)

print(results_df.head())
print(top_rows.head())

# Ensure columns are stripped of whitespaces and of the correct type
for col in ['dataset', 'lm', 'clustering']:
    results_df[col] = results_df[col].astype(str).str.strip()
    top_rows[col] = top_rows[col].astype(str).str.strip()

# Ensure that 'k' and 'threshold' have the same type (both float or both int)
results_df['k'] = results_df['k'].astype(float)
top_rows['k'] = top_rows['k'].astype(float)

results_df['threshold'] = results_df['threshold'].astype(float)
top_rows['threshold'] = top_rows['threshold'].astype(float)

# Merge the dataframes on the specified columns
merged_df = pd.merge(results_df, top_rows, 
                     on=['dataset', 'lm', 'k', 'clustering', 'threshold'], how='inner')

# Check if the merge produced any results
if merged_df.empty:
    print("No matching rows found in the merge. Check the data for inconsistencies.")
else:
    # Assign the 'f1' from results_df to 'real_f1' and remove the old 'f1' column
    merged_df['real_f1'] = merged_df['f1']
    merged_df.drop(columns=['f1'], inplace=True)

    # Rename columns if necessary, or adjust as per requirements
    merged_df.rename(columns={
        'precision': 'real_precision',
        'recall': 'real_recall'
    }, inplace=True)

    # Print the merged dataframe to check the result
    print("Results: ")
    print(merged_df)

    # Save the final dataframe to a CSV file
    confcsv_name = confcsv.split('/')[-1].split('.')[0].lower()
    merged_df.to_csv(f"./{RESULTS_DIR}/{EVALUATED_DIR}/{confcsv_name}.csv", index=False)

    # Print the time the script ended
    print("Time ended: ", time.ctime())