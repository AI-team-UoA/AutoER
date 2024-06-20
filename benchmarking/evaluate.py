import time
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering, UniqueMappingClustering, KiralyMSMApproximateClustering, BestMatchClustering
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Create test data for benchmarking')
parser.add_argument('--data', type=str, default='census')
args = parser.parse_args()

dataset_name = args.data

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

verbose=False

# ------------------------------- DATA ------------------------------- #

predictions = pd.read_csv(f"./predictions/{dataset_name}_predictions.csv")

if dataset_name == 'census':
    predictions = predictions[predictions['clustering'] == 'ConnectedComponentsClustering']
    datasets = ['50K','100K','200K','300K','1M','2M']
elif dataset_name == 'dbpedia':
    datasets = ['dbpedia']

# census_datasets = ['10K','50K']
# predictions = predictions.head(10)


for dataset in datasets:
    print("\n\nRunning for: ", dataset)
    start_time = time.time()
    print("Time started: ", time.ctime())
    dataset_predictions = predictions[predictions['dataset'] == dataset]
    true_f1s = []

    if dataset_name == 'dbpedia':
        data = Data(dataset_1=pd.read_csv("../data/dbpedia/" + dataset + "/full.csv", sep='|', engine='python').astype(str),
                id_column_name_1='Id',
                attributes_1=['Aggregate Value'],
                dataset_name_1=dataset,
                ground_truth=pd.read_csv("../data/dbpedia/" + dataset + "/duplicates.csv", sep='|', engine='python').astype(str)
            )
    elif dataset_name == 'census':
        data = Data(dataset_1=pd.read_csv("../data/census/" + dataset + "/full.csv", sep='|', engine='python').astype(str),
                    id_column_name_1='Id',
                    attributes_1=['Aggregate Value'],
                    dataset_name_1=dataset,
                    ground_truth=pd.read_csv("../data/census/" + dataset + "/duplicates.csv", sep='|', engine='python').astype(str)
                )

    for row in tqdm(dataset_predictions.iterrows(), total=dataset_predictions.shape[0]):
        row = row[1]
        lm = row['lm']
        k = row['k']
        clustering_method = row['clustering']

        threshold = row['threshold']

        # print("Running for: ", dataset, lm, k, clustering_method, threshold)

        emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
        blocks, g = emb.build_blocks(data,
                                    top_k=k,
                                    load_embeddings_if_exist=True,
                                    save_embeddings=True,
                                    tqdm_disable=True,
                                    with_entity_matching=True)
        emb.evaluate(blocks, verbose=verbose)

        ccc = CLUSTERING_MAPPING[clustering_method]()
        clusters = ccc.process(g, data, similarity_threshold=threshold)
        results = ccc.evaluate(clusters, with_classification_report=False, verbose=verbose)

        t2 = time.time()
        f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']

        f1 = round(f1, 4)
        precision = round(precision, 4)
        recall = round(recall, 4)
        
        true_f1s.append(f1)
        # print("Time finished: ", time.time() - start_time)

        # print(f"Dataset: {dataset}, LM: {lm}, K: {k}, Clustering: {clustering_method}, Threshold: {threshold},\n F1: {f1},\n Precision: {precision},\n Recall: {recall}")

    dataset_predictions.loc[:, 'true'] = true_f1s
    dataset_predictions.to_csv('./predictions/'+dataset+"_results.csv", index=False)
