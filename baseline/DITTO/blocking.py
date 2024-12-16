import time
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering, UniqueMappingClustering, KiralyMSMApproximateClustering, BestMatchClustering
from tqdm import tqdm
import os
import argparse
parser = argparse.ArgumentParser(description='Create test data for benchmarking')
parser.add_argument('--datajson', type=str)
args = parser.parse_args()

datajson = args.datajson
# datajson = '../../data/configs/D2.json'

print("Data JSON: ", datajson)

import json

with open(datajson) as f:
    DATASET_SPECS = json.load(f)

print("JSON Data: ")
print(DATASET_SPECS)

dataset = DATASET_SPECS['name']
print("Dataset: ", dataset)
format = DATASET_SPECS['format']
print("Format: ", format)

print("Time started: ", time.ctime())

RESULTS_DIR = './blocks'
DATA_DIR = '../../data'

data = Data(
    dataset_1=pd.read_csv(f"{DATA_DIR}/{DATASET_SPECS['dir']}/{DATASET_SPECS['d1']}.{DATASET_SPECS['format']}", 
                                  sep=DATASET_SPECS['separator'], 
                                  engine=DATASET_SPECS['engine']).astype(str),
    id_column_name_1=DATASET_SPECS['d1_id'],
    dataset_name_1=DATASET_SPECS["name"] + "_" + DATASET_SPECS['d1'],
    dataset_2=pd.read_csv(f"{DATA_DIR}/{DATASET_SPECS['dir']}/{DATASET_SPECS['d2']}.{DATASET_SPECS['format']}",
                          sep=DATASET_SPECS['separator'], 
                          engine=DATASET_SPECS['engine']).astype(str),
    id_column_name_2=DATASET_SPECS['d2_id'],
    dataset_name_2=DATASET_SPECS["name"] + "_" + DATASET_SPECS['d2'],
    ground_truth=pd.read_csv(f"{DATA_DIR}/{DATASET_SPECS['dir']}/{DATASET_SPECS['gt']}.{DATASET_SPECS['format']}", 
                                 sep=DATASET_SPECS['separator']).astype(str)
)

start_time = time.time()

lm = 'st5'
k = 10
threshold = 0.5
clustering = UniqueMappingClustering
print("Running for: ", dataset, lm, k, clustering, threshold)

print('Proposed configuration: ')
print('LM: ', lm)
print('K: ', k)
print('Clustering: ', clustering)
print('Threshold: ', threshold)

print('\nBuilding blocks...')
print("Time started: ", time.ctime())
emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
blocks, g = emb.build_blocks(data,
                            top_k=k,
                            load_embeddings_if_exist=True,
                            save_embeddings=True,
                            tqdm_disable=False,
                            verbose=True,
                            with_entity_matching=True)
# print("Device used: ", emb.device)    
verbose = True
# emb.evaluate(blocks, verbose=verbose)
print('Time ended: ', time.ctime())
print('Finished building blocks')
# print(blocks)



t2 = time.time()
runtime = t2 - start_time

ALL_CANDIDATE_PAIRS = set()
for entity_from_d1, set_of_d2_candidates in blocks.items():
    for entity_from_d2 in list(set_of_d2_candidates):
        ALL_CANDIDATE_PAIRS.add((str(entity_from_d1), str(entity_from_d2-data.dataset_limit)))
# len(ALL_CANDIDATE_PAIRS)

# create a random split in train test validation set in 3-1-1

import random

random.seed(42)

ALL_CANDIDATE_PAIRS = list(ALL_CANDIDATE_PAIRS)
random.shuffle(ALL_CANDIDATE_PAIRS)
train_size = int(0.6 * len(ALL_CANDIDATE_PAIRS))
train_set = ALL_CANDIDATE_PAIRS[:train_size]
test_size = int(0.2 * len(ALL_CANDIDATE_PAIRS))
test_set = ALL_CANDIDATE_PAIRS[train_size:train_size+test_size]
validation_set = ALL_CANDIDATE_PAIRS[train_size+test_size:]

print("Train set: ", len(train_set))
print("Test set: ", len(test_set))
print("Validation set: ", len(validation_set))

# Helper function to format a record into Ditto's COL/VAL style
def format_entry(entry, columns):
    formatted = []
    for col in columns:
        formatted.append(f"COL {str.lower(col)} VAL {str.lower(entry[col])}")
    return " ".join(formatted)

# Load dataset columns
columns_1 = list(data.dataset_1.columns)
columns_2 = list(data.dataset_2.columns)

# Load ground truth
ground_truth_pairs = set(
    tuple(x) for x in data.ground_truth.values
)

# Helper function to process candidate pairs into Ditto format
def process_pairs(candidate_pairs, dataset_1, dataset_2, ground_truth_pairs):
    ditto_data = []
    tp = 0
    for d1_id, d2_id in candidate_pairs:

        record_1 = dataset_1[dataset_1[DATASET_SPECS['d1_id']] == d1_id].iloc[0]
        record_2 = dataset_2[dataset_2[DATASET_SPECS['d2_id']] == d2_id].iloc[0]
        
        # Format entries
        entry_1 = format_entry(record_1, columns_1)
        entry_2 = format_entry(record_2, columns_2)
        
        # Determine label
        label = 1 if (d1_id, d2_id) in ground_truth_pairs else 0

        if label == 1:
            tp += 1
        
        # Combine into Ditto format
        ditto_data.append(f"{entry_1} \t {entry_2} \t {label}")
    print("TP: ", tp)
    return ditto_data

# Process train, test, and validation sets
train_ditto = process_pairs(train_set, data.dataset_1, data.dataset_2, ground_truth_pairs)
test_ditto = process_pairs(test_set, data.dataset_1, data.dataset_2, ground_truth_pairs)
validation_ditto = process_pairs(validation_set, data.dataset_1, data.dataset_2, ground_truth_pairs)
import os
# Create a folder with the dataset name
output_folder = 'ready_for_ditto_input/'+dataset
os.makedirs(output_folder, exist_ok=True)

# Write the TXT files into the folder
with open(os.path.join(output_folder, "train.txt"), "w") as train_file:
    train_file.write("\n".join(train_ditto))

with open(os.path.join(output_folder, "test.txt"), "w") as test_file:
    test_file.write("\n".join(test_ditto))

with open(os.path.join(output_folder, "valid.txt"), "w") as validation_file:
    validation_file.write("\n".join(validation_ditto))

print(f"Files written to folder: {output_folder}")
