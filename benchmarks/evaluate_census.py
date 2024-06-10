import time
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import ConnectedComponentsClustering
from tqdm import tqdm
import torch

CLUSTERING_MAPPING = {
    "ConnectedComponentsClustering": ConnectedComponentsClustering
}
verbose=False

# ------------------------------- DATA ------------------------------- #

census_predictions = pd.read_csv("./predictions/census_predictions.csv")
census_predictions = census_predictions[census_predictions['clustering'] == 'ConnectedComponentsClustering']
census_datasets = ['10K','50K','100K','200K','300K','1M','2M']


# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA device not found. Using CPU.")

results_df = pd.DataFrame(columns=['dataset', 'lm', 'k', 'clustering', 'threshold', 'f1', 'precision', 'recall', 'time'])

for census_dataset in census_datasets:
    print("\n\nRunning for: ", census_dataset)
    start_time = time.time()
    print("Time started: ", time.ctime())
    census_dataset_predictions = census_predictions[census_predictions['dataset'] == census_dataset]
    data = Data(dataset_1=pd.read_csv("../data/census/" + census_dataset + "/full.csv", sep='|', engine='python').astype(str),
                id_column_name_1='Id',
                attributes_1=['Aggregate Value'],
                dataset_name_1=census_dataset,
                ground_truth=pd.read_csv("../data/census/" + census_dataset + "/duplicates.csv", sep='|', engine='python').astype(str)
            )

    max_predicted_index = census_dataset_predictions['predicted'].idxmax()
    row = census_dataset_predictions.loc[max_predicted_index]

    print("Max predicted regressor score: ", row['predicted'])

    lm = row['lm']
    k = int(row['k'])
    clustering_method = row['clustering']

    if clustering_method != 'ConnectedComponentsClustering':
        continue

    threshold = float(row['threshold'])

    print("Running for: ", census_dataset, lm, k, clustering_method, threshold)

    emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
    blocks, g = emb.build_blocks(data,
                                top_k=k,
                                load_embeddings_if_exist=True,
                                save_embeddings=True,
                                tqdm_disable=False,
                                with_entity_matching=True)
    emb.evaluate(blocks, verbose=verbose)

    ccc = CLUSTERING_MAPPING[clustering_method]()
    clusters = ccc.process(g, data, similarity_threshold=threshold)
    results = ccc.evaluate(clusters, with_classification_report=False, verbose=verbose)

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
        'dataset': [census_dataset],
        'lm': [lm],
        'k': [k],
        'clustering': [clustering_method],
        'threshold': [threshold],
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'time': [runtime]
    })

    # Use concat to add the new row to results_df
    results_df = pd.concat([results_df, new_data], ignore_index=True)

results_df.to_csv("results/census_results.csv", index=False)