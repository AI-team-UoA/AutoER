"""
This script is used to run optuna trials for pyjedai auto configuration.

Parameters:
    --dbname: database name
    --ntrials: number of trials
    --clustering: clustering method
    --sampler: optuna sampler to use
    --d: dataset id (1-10)
    --verbose: verbose mode

Example:
    python autoconf_sampling.py --dbname dbs/autoconf_sampling --ntrials 100 --clustering UniqueMappingClustering --sampler TPESampler --d 1 --verbose True
"""
import time
import optuna
import os
import sys
import pandas as pd

import numpy as np
from scipy.integrate import simps

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-notebook')

from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
from pyjedai.clustering import UniqueMappingClustering, ConnectedComponentsClustering, KiralyMSMApproximateClustering
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from optuna.samplers import RandomSampler, QMCSampler, TPESampler, GPSampler


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

parser = argparse.ArgumentParser(description='Run optuna trials for pyjedai auto configuration')
parser.add_argument('--dbname', type=str, default="None")
parser.add_argument('--ntrials', type=int, default=100)
parser.add_argument('--clustering', type=str, default=None)
parser.add_argument('--sampler', type=str, default="TPESampler", help='Optuna sampler to use')
parser.add_argument('--d', type=int, default=None)
parser.add_argument('--verbose', type=bool, default=False)

args = parser.parse_args()
db_name = args.dbname
num_of_trials = args.ntrials
clustering = args.clustering
sampler = args.sampler
prompt_d = args.d
verbose = args.verbose

# ------------------------------- EXPERIMENTS CONFIGURATION ------------------------------- #

COLOR_MAPPING = {
    "tpe" : "blue",
    "random": "red",
    "qmc": "green",
    "gps": "purple"
}

COLOR_MAPPING_FILL = {
    "tpe" : "lightblue",
    "random": "pink",
    "qmc": "lightgreen",
    "gps": "violet"
}

CLUSTERING_MAPPING = {
    "UniqueMappingClustering": UniqueMappingClustering,
    # "BestMatchClustering": BestMatchClustering,
    "KiralyMSMApproximateClustering": KiralyMSMApproximateClustering,
    "ConnectedComponentsClustering": ConnectedComponentsClustering,
    # "CorrelationClustering": CorrelationClustering,
    # "CutClustering": CutClustering,
    # "RowColumnClustering": RowColumnClustering,
    # "RicochetRClustering": RicochetRClustering,
    # "CenterClustering": CenterClustering
}

SAMPLERS_MAPPING = {
    "tpe" : TPESampler,
    "random": RandomSampler,
    "qmc": QMCSampler,
    "gps": GPSampler
}

SAMPLERS_OPTUNA_NAMES_MAPPING = {
    "tpe" : "TPESampler",
    "random": "RandomSampler",
    "qmc": "QMCSampler",
    "gps": "GPSampler"
}

SEARCH_SPACE = {
    "threshold": [0.05, 0.95],
    'k': [1, 100],
    'lm': ["smpnet", "st5", "sdistilroberta", "sminilm", "sent_glove", 'fasttext', 'word2vec'],
    "clustering" : list(CLUSTERING_MAPPING.keys())
}

SEEDS = [16, 64, 256, 1024, 4096]
TRIALS_SERIES = range(5, num_of_trials+5, 5)

# SEEDS = [16]
# TRIALS_SERIES = range(5, 20+5, 5)

DESTINATION_FOLDER = 'results/'
DATA_DIR = '../data/'

PYJEDAI_TQDM_DISABLE = True
MAX_F1 = 100

CSV_FILE_NAMES = [d+'.csv' for d in D]
OUTPUT_EXCEL_FILE_NAME = 'sampling.xlsx'

TRIALS_FILE_COLUMNS = 'trial,dataset,clustering,lm,k,threshold,sampler,seed,precision,recall,f1,runtime\n'

# ------------------------------- EXPERIMENTS CONFIGURATION END ------------------------------- #

AUC_FILE_NAME = DESTINATION_FOLDER+"auc.csv"
DATASET_NAME = DESTINATION_FOLDER+"trials_dataset.csv"

if os.path.exists(AUC_FILE_NAME):
    AUC_RESULTS_FILE = open(AUC_FILE_NAME, "a")
else:
    AUC_RESULTS_FILE = open(AUC_FILE_NAME, "w")
    AUC_RESULTS_FILE.write(f"sampler,dataset,area_trapz,auc,runtime\n")


if os.path.exists(DATASET_NAME):
    DATASET_FILE = open(DATASET_NAME, "a")
else:
    DATASET_FILE = open(DATASET_NAME, "w")
    DATASET_FILE.write(TRIALS_FILE_COLUMNS)

# FOR ALL SAMPLERS
for sampler in SAMPLERS_MAPPING.keys():
    DB_NAME = "autoconf_" + sampler
    STORAGE_NAME = "sqlite:///{}.db".format(DB_NAME)
    
    if verbose:
        print("\n\n\n\n")
        print("DB name: ", db_name)

    avg_f1_per_trial = {t: 0 for t in TRIALS_SERIES}

    # FOR ALL SEEDS
    for seed in SEEDS:
        
        best_trials = []
        # FOR ALL TRIALS
        for num_of_trials in TRIALS_SERIES:
            
            # FOR ALL DATASETS
            for i in range(0,len(D)):
                
                if prompt_d:
                    i=prompt_d-1
                
                if verbose:
                    print("\n\nDataset: ", D[i])

                d = D[i]
                d1 = D1CSV[i]
                d2 = D2CSV[i]
                gt = GTCSV[i]
                s = separator[i]
                e = engine[i]

                if os.path.exists(DESTINATION_FOLDER + "/" + sampler + "/" + d + '.csv'):
                    TRIALS_FILE = open(DESTINATION_FOLDER + "/" + sampler + "/" + d + '.csv', 'a')
                else:
                    TRIALS_FILE = open(DESTINATION_FOLDER + "/" + sampler + "/" + d + '.csv', 'w')
                    TRIALS_FILE.write(TRIALS_FILE_COLUMNS)
                
                data = Data(
                    dataset_1=pd.read_csv(DATA_DIR + d + "/" + d1 , 
                                        sep=s,
                                        engine=e,
                                        na_filter=False).astype(str),
                    id_column_name_1='id',
                    dataset_name_1=d+"_"+d1.split(".")[0],
                    dataset_2=pd.read_csv(DATA_DIR + d + "/" + d2 , 
                                        sep=s, 
                                        engine=e, 
                                        na_filter=False).astype(str),
                    id_column_name_2='id',
                    dataset_name_2=d+"_"+d2.split(".")[0],
                    ground_truth=pd.read_csv(DATA_DIR + d + "/gt.csv", sep=s, engine=e))
                
                if verbose:
                    data.print_specs()

                STUDY_NAME = d + "_" + str(num_of_trials) + "_" + str(seed)

                def objective(trial):
                    try:
                        t1 = time.time()

                        lm = trial.suggest_categorical('lm', SEARCH_SPACE['lm'])
                        emb = EmbeddingsNNBlockBuilding(vectorizer=lm, similarity_search='faiss')
                        k = trial.suggest_int('k', SEARCH_SPACE['k'][0], SEARCH_SPACE['k'][1])
                        blocks, g = emb.build_blocks(data,
                                                    top_k=k,
                                                    load_embeddings_if_exist=True,
                                                    save_embeddings=True,
                                                    tqdm_disable=PYJEDAI_TQDM_DISABLE,
                                                    with_entity_matching=True,
                                                    verbose=verbose)
                        emb.evaluate(blocks, verbose=verbose)
                        
                        clustering_method = trial.suggest_categorical('clustering', SEARCH_SPACE['clustering'])
                        ccc = CLUSTERING_MAPPING[clustering_method]()
                        threshold = trial.suggest_float("threshold", SEARCH_SPACE['threshold'][0], SEARCH_SPACE['threshold'][1])
                        clusters = ccc.process(g, data, similarity_threshold=threshold)
                        results = ccc.evaluate(clusters, with_classification_report=True, verbose=verbose)

                        t2 = time.time()
                        f1, precision, recall = results['F1 %'], results['Precision %'], results['Recall %']
                        f1 = round(f1, 4)
                        precision = round(precision, 4)
                        recall = round(recall, 4)
                        execution_time = round(t2-t1, 4)
                        TRIALS_FILE.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(trial.number, d, clustering_method, lm, \
                                                                                            k, threshold, sampler, seed, precision, recall, f1, execution_time))
                        TRIALS_FILE.flush()
                        
                        return f1

                    except ValueError as e:
                        print(e)
                        trial.set_user_attr("failed", True)
                        return optuna.TrialPruned()
                
                overall_runtime = time.time()
                if sampler == "random" or sampler == "gps":
                    optuna_sampler = SAMPLERS_MAPPING[sampler](seed=seed)
                else:
                    optuna_sampler = SAMPLERS_MAPPING[sampler](seed=seed, warn_independent_sampling=False if sampler == "qmc" else True)

                study = optuna.create_study(
                    directions=["maximize"],
                    sampler=optuna_sampler,
                    study_name=STUDY_NAME,
                    storage=STORAGE_NAME,
                    load_if_exists=False
                )

                study.optimize(
                    objective, 
                    n_trials=num_of_trials, 
                    show_progress_bar=False,
                    callbacks=[MaxTrialsCallback(num_of_trials, states=(TrialState.COMPLETE,))]
                )
                overall_runtime = time.time() - overall_runtime
                best_f1 = study.best_trial.value
                avg_f1_per_trial[num_of_trials] += best_f1
                
                TRIALS_FILE.close()
                if prompt_d:
                    break
            
    x, y = list(avg_f1_per_trial.keys()), list(avg_f1_per_trial.values())    
    y = [i/len(SEEDS) for i in y]
    
    # plt.plot(x, y, 
    #          label=SAMPLERS_OPTUNA_NAMES_MAPPING[sampler], 
    #          color=COLOR_MAPPING[sampler])

    area_trapz = np.trapz(y, x)

    max_x = max(x)
    max_y = MAX_F1
    square_area = max_x * max_y
    auc = area_trapz / square_area
    overall_runtime = round(overall_runtime, 4)
    area_trapz = round(area_trapz, 4)
    auc = round(auc, 4)
    AUC_RESULTS_FILE.write(f"{sampler},{d},{area_trapz},{auc},{overall_runtime}\n")
    AUC_RESULTS_FILE.flush()

# plt.legend()
# plt.ylim(0, 100)
# plt.grid()
# plt.xticks(TRIALS_SERIES)
# plt.xlabel('Number of trials')
# plt.ylabel('Average F1 score')
# plt.title('Convergence for '+  'D'+ str(prompt_d) + ' dataset')
# plt.savefig(DESTINATION_FOLDER+'plots/' + 'D'+ str(prompt_d) + '.png')

AUC_RESULTS_FILE.close()