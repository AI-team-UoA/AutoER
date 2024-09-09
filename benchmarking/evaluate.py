import time
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering, UniqueMappingClustering, KiralyMSMApproximateClustering, BestMatchClustering
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Create test data for benchmarking')
parser.add_argument('--topk', type=int, default=1, help='Number of top rows to consider')
parser.add_argument('--data', type=str, default='dbpedia')
args = parser.parse_args()

dataset_name = args.data
TOPK = args.topk

# print time started
print("Time started: ", time.ctime())


if dataset_name == 'census':
    CLUSTERING_MAPPING = {
        "ConnectedComponentsClustering": ConnectedComponentsClustering
    }
elif dataset_name == 'dbpedia':
    CLUSTERING_MAPPING = {
        "UniqueMappingClustering": UniqueMappingClustering,
        "BestMatchClustering": BestMatchClustering,
        "KiralyMSMApproximateClustering": KiralyMSMApproximateClustering,
        "ConnectedComponentsClustering": ConnectedComponentsClustering
    }
else:
    raise ValueError("Dataset not found")

verbose=True

# ------------------------------- DATA ------------------------------- #

predictions = pd.read_csv(f"./predictions/{dataset_name}_predictions.csv")

if dataset_name == 'census':
    predictions = predictions[predictions['clustering'] == 'ConnectedComponentsClustering']
    datasets = ['50K','100K','200K','300K','1M','2M']
elif dataset_name == 'dbpedia':
    datasets = ['dbpedia']

# census_datasets = ['10K','50K']
# predictions = predictions.head(10)

results_df = pd.DataFrame(columns=['dataset', 'lm', 'k', 'clustering', 'threshold', 'f1', 'precision', 'recall', 'time'])

for dataset in datasets:
    print("\n\nRunning for: ", dataset)
    start_time = time.time()
    print("Time started: ", time.ctime())
    dataset_predictions = predictions[predictions['dataset'] == dataset]
    true_f1s = []

    if dataset_name == 'dbpedia':
        data = Data(dataset_1=pd.read_csv("../data/" + dataset + "/newDBPedia1.csv", sep='|', engine='python').astype(str),
                id_column_name_1='Id',
                dataset_name_1=dataset+'1',
                dataset_2=pd.read_csv("../data/" + dataset + "/newDBPedia2.csv", sep='|', engine='python').astype(str),
                id_column_name_2='Id',
                dataset_name_2=dataset+'2',
                ground_truth=pd.read_csv("../data/" + dataset + "/newDBPediaMatchesgt.csv", sep='|').astype(str)
            )
    elif dataset_name == 'census':
        data = Data(dataset_1=pd.read_csv("../data/census/" + dataset + "/full.csv", sep='|', engine='python').astype(str),
                    id_column_name_1='Id',
                    attributes_1=['Aggregate Value'],
                    dataset_name_1=dataset,
                    ground_truth=pd.read_csv("../data/census/" + dataset + "/duplicates.csv", sep='|', engine='python').astype(str)
                )

    top_rows = dataset_predictions.nlargest(TOPK, 'predicted')

    # Loop over each row in the top K rows
    for index, row in top_rows.iterrows():
        print("Max predicted regressor score: ", row['predicted'])
        
        lm = row['lm']
        k = int(row['k'])
        clustering_method = row['clustering']
        threshold = float(row['threshold'])
        print("Running for: ", dataset_name, lm, k, clustering_method, threshold)
        
        print('Proposed configuration: ')
        print('LM: ', lm)
        print('K: ', k)
        print('Clustering: ', clustering_method)
        print('Threshold: ', threshold)

        print('Executing...')

        emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
        blocks, g = emb.build_blocks(data,
                                    top_k=k,
                                    load_embeddings_if_exist=True,
                                    save_embeddings=True,
                                    tqdm_disable=False,
                                    with_entity_matching=True)
        # emb.evaluate(blocks, verbose=verbose)

        print('Finished building blocks')

        print('Clustering...')        
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
            'dataset': [dataset_name],
            'lm': [lm],
            'k': [k],
            'clustering': [clustering_method],
            'threshold': [threshold],
            'f1': [f1],
            'precision': [precision],
            'recall': [recall],
            'time': [runtime]
        })
        
        results_df = pd.concat([results_df, new_data], ignore_index=True)

results_df.to_csv(f"./results/{dataset_name}_results.csv", index=False)