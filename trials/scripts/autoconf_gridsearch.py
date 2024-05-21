import time
import optuna
import os
import sys
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import UniqueMappingClustering
from pyjedai.matching import EntityMatching
from pyjedai.clustering import CorrelationClustering, CenterClustering, MergeCenterClustering, \
                                UniqueMappingClustering, ConnectedComponentsClustering, ExactClustering, \
                                BestMatchClustering, KiralyMSMApproximateClustering
from tqdm import tqdm
import numpy as np

from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

import argparse

# ------------------------------- DATA ------------------------------- #
D1CSV = [
    "rest1.csv", "abt.csv", "amazon.csv", "dblp.csv",  "imdb.csv",  "imdb.csv",  "tmdb.csv",  "walmart.csv",   "dblp.csv",    "imdb.csv"
]
D2CSV = [
    "rest2.csv", "buy.csv", "gp.csv",     "acm.csv",   "tmdb.csv",  "tvdb.csv",  "tvdb.csv",  "amazon.csv",  "scholar.csv", "dbpedia.csv"
]
GTCSV = [
    "gt.csv",   "gt.csv",   "gt.csv",     "gt.csv",   "gt.csv", "gt.csv", "gt.csv", "gt.csv", "gt.csv", "gt.csv"
]
D = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9','D10']

separator = [
    '|', '|', '#', '%', '|', '|', '|', '|', '>', '|'
]
engine = [
    'python', 'python','python','python','python','python','python','python','python', None
]
# -------------------------------  DATA END  ------------------------------- #

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--did', type=int, default=-1)
parser.add_argument('--dbname', type=str, default=-1)
parser.add_argument('--clustering', type=str, default=None)
parser.add_argument('--verbose', type=bool, default=False)

did = parser.parse_args().did
db_name= parser.parse_args().dbname
clustering = parser.parse_args().clustering
verbose = parser.parse_args().verbose

# ------------------------------- EXPERIMENTS CONFIGURATION ------------------------------- #

CLUSTERING_MAPPING = {
    "UniqueMappingClustering": UniqueMappingClustering,
    "BestMatchClustering": BestMatchClustering,
    "KiralyMSMApproximateClustering": KiralyMSMApproximateClustering,
    "ConnectedComponentsClustering": ConnectedComponentsClustering
}

SEARCH_SPACE = {
    "threshold": np.arange(0.05, 0.95, 0.05),
    'k': range(1, 100, 1),
    'lm': ["smpnet", "st5", "sdistilroberta", "sminilm", "sent_glove"],
    "clustering" : ["UniqueMappingClustering", "BestMatchClustering", \
        "KiralyMSMApproximateClustering", "ConnectedComponentsClustering"]
}

DB_NAME = "autoconf_gridsearch" if db_name == -1 else db_name
STORAGE_NAME = "sqlite:///{}.db".format(DB_NAME)
SEED = 42
CSV_FILE_COLUMNS = 'trial,dataset,clustering,lm,k,threshold,sampler,seed,precision,recall,f1,runtime\n'
DESTINATION_FOLDER = 'results/gridsearch/'
PYJEDAI_TQDM_DISABLE = True
NUM_OF_TRIALS = len(SEARCH_SPACE["threshold"]) * len(SEARCH_SPACE["k"]) * len(SEARCH_SPACE['lm'])
CSV_FILE_NAMES = [d+'.csv' for d in D]
OUTPUT_EXCEL_FILE_NAME = 'gridsearch.xlsx'
SAMPLER = 'gridsearch'
# ------------------------------- EXPERIMENTS CONFIGURATION END ------------------------------- #


for i in range(0,len(D)):
    
    if did != -1:
        i = did-1
    
    print("\n\nDataset: ", D[i])

    d = D[i]
    d1 = D1CSV[i]
    d2 = D2CSV[i]
    gt = GTCSV[i]
    s = separator[i]
    e = engine[i]

    with open(DESTINATION_FOLDER + d + '.csv', 'w') as f:
        f.write(CSV_FILE_COLUMNS)
        data = Data(
            dataset_1=pd.read_csv("../data/ccer/" + d + "/" + d1 , 
                                sep=s,
                                engine=e,
                                na_filter=False).astype(str),
            id_column_name_1='id',
            dataset_name_1=d+"_"+d1.split(".")[0],
            dataset_2=pd.read_csv("../data/ccer/" + d + "/" + d2 , 
                                sep=s, 
                                engine=e,
                                na_filter=False).astype(str),
            id_column_name_2='id',
            dataset_name_2=d+"_"+d2.split(".")[0],
            ground_truth=pd.read_csv("../data/ccer/" + d + "/gt.csv", sep=s, engine=e))

        if verbose:
            data.print_specs()

        study_name = title  = d
        
        def objective(trial):
            try:
                t1 = time.time()

                lm = trial.suggest_categorical('lm', SEARCH_SPACE['lm'])
                emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
                k = trial.suggest_categorical("k", SEARCH_SPACE['k'])
                blocks, g = emb.build_blocks(data,
                                            top_k=k,
                                            load_embeddings_if_exist=True,
                                            save_embeddings=True,
                                            tqdm_disable=PYJEDAI_TQDM_DISABLE,
                                            with_entity_matching=True)
                emb.evaluate(blocks, verbose=verbose)
                
                clustering_method = trial.suggest_categorical('clustering', SEARCH_SPACE['clustering'])
                ccc = CLUSTERING_MAPPING[clustering_method]()
                threshold = trial.suggest_categorical("threshold", SEARCH_SPACE["threshold"])
                clusters = ccc.process(g, data, similarity_threshold=threshold)
                results = ccc.evaluate(clusters, with_classification_report=True, verbose=verbose)

                t2 = time.time()
                f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']

                f1 = round(f1, 4)
                precision = round(precision, 4)
                recall = round(recall, 4)

                f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(trial.number, d, clustering_method, lm, k, threshold, SAMPLER, SEED, precision, recall, f1, t2-t1))
                f.flush()

                return f1
    
            except ValueError as e:
                trial.set_user_attr("failed", True)
                return optuna.TrialPruned()
        
        study = optuna.create_study(
            directions=["maximize"],
            sampler=optuna.samplers.BruteForceSampler(),
            study_name=study_name,
            storage=STORAGE_NAME,
            load_if_exists=False
        )

        study.optimize(
            objective, 
            n_trials=NUM_OF_TRIALS, 
            show_progress_bar=True,
            callbacks=[MaxTrialsCallback(NUM_OF_TRIALS, states=(TrialState.COMPLETE,))]
        )

        f.close()
        
        if did != -1:
            break

with pd.ExcelWriter(OUTPUT_EXCEL_FILE_NAME, engine='openpyxl') as writer:
    for csv_file in CSV_FILE_NAMES:
        df = pd.read_csv(DESTINATION_FOLDER + csv_file)
        sheet_name = csv_file.rsplit('.', 1)[0]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

