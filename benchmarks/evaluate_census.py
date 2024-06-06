import time
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering
from tqdm import tqdm

CLUSTERING_MAPPING = {
    "ConnectedComponentsClustering": ConnectedComponentsClustering
}
verbose=False

# ------------------------------- DATA ------------------------------- #

census_predictions = pd.read_csv("./predictions/census_predictions.csv")

# Drop all with clustering method not ConnectedComponentsClustering

census_predictions = census_predictions[census_predictions['clustering'] == 'ConnectedComponentsClustering']
census_datasets = ['10K','50K','100K','200K','300K','1M','2M']


# census_datasets = ['10K','50K']
# census_predictions = census_predictions.head(10)


for census_dataset in census_datasets:
    print("\n\nRunning for: ", census_dataset)
    census_dataset_predictions = census_predictions[census_predictions['dataset'] == census_dataset]
    true_f1s = []

    data = Data(dataset_1=pd.read_csv("../data/census/" + census_dataset + "/full.csv", sep='|', engine='python').astype(str),
                id_column_name_1='Id',
                attributes_1=['Aggregate Value'],
                dataset_name_1=census_dataset,
                ground_truth=pd.read_csv("../data/census/" + census_dataset + "/duplicates.csv", sep='|', engine='python').astype(str)
            )

    for row in tqdm(census_dataset_predictions.iterrows(), total=census_dataset_predictions.shape[0]):
        row = row[1]
        lm = row['lm']
        k = row['k']
        clustering_method = row['clustering']

        if clustering_method != 'ConnectedComponentsClustering':
            continue

        threshold = row['threshold']

        print("Running for: ", census_dataset, lm, k, clustering_method, threshold)

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

        print(f"Dataset: {census_dataset}, LM: {lm}, K: {k}, Clustering: {clustering_method}, Threshold: {threshold},\n F1: {f1},\n Precision: {precision},\n Recall: {recall}")

    census_dataset_predictions.loc[:, 'true'] = true_f1s
    census_dataset_predictions.to_csv('./predictions/'+census_dataset+"_results.csv", index=False)
