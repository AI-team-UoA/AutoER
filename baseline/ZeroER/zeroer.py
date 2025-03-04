from data_loading_helper.data_loader import load_data
from data_loading_helper.feature_extraction import *
from utils import run_zeroer
from blocking_functions import *
from os.path import join
import os
import re

import json
from time import time
import pandas as pd
pd.options.mode.chained_assignment = None
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset",type=str)
parser.add_argument("dataset_alias",type=str)
parser.add_argument("log_dir",type=str)
parser.add_argument("--run_transitivity",type=bool,default=False,nargs="?",const=True, help="whether to enforce transitivity constraint")
parser.add_argument("--LR_dup_free",type=bool,default=False,nargs="?",const=True, help="are the left table and right table duplicate-free?")
parser.add_argument("--LR_identical",type=bool,default=False,nargs="?",const=True, help="are the left table and right table identical?")
parser.add_argument("--sep",type=str,default=",",nargs="?",const=True, help="separator for files")

data_path = "datasets"

if __name__ == '__main__':
    args = parser.parse_args()
    LR_dup_free = args.LR_dup_free
    run_trans = args.run_transitivity
    LR_identical = args.LR_identical
    dataset_name = args.dataset
    dataset_alias = args.dataset_alias
    sep = args.sep
    print("Separator", sep)
    dataset_path = join(data_path,dataset_name)
    #blocking_func = blocking_functions_mapping[dataset_name]
    blocking_func = block_agg
    log_dir = args.log_dir
    log_file = log_dir+'ZeroER.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    features = time()
    
    try:
        candset_features_df = pd.read_csv(join(dataset_path,"candset_features_df.csv"), index_col=0)
        print(candset_features_df.head(5))
        candset_features_df.reset_index(drop=True,inplace=True)
        if run_trans==True:
            id_df = candset_features_df[["ltable_id","rtable_id"]]
            id_df.reset_index(drop=True,inplace=True)
            if LR_dup_free==False and LR_identical==False:
                candset_features_df_l = pd.read_csv(join(dataset_path,"candset_features_df_l.csv"), index_col=0)
                candset_features_df_l.reset_index(drop=True,inplace=True)
                candset_features_df_r = pd.read_csv(join(dataset_path,"candset_features_df_r.csv"), index_col=0)
                candset_features_df_r.reset_index(drop=True,inplace=True)
                id_df_l = candset_features_df_l[["ltable_id","rtable_id"]]
                id_df_l.reset_index(drop=True,inplace=True)
                id_df_r = candset_features_df_r[["ltable_id","rtable_id"]]
                id_df_r.reset_index(drop=True,inplace=True)
        print(
            "Features already generated, reading from file: " + dataset_path + "/candset_features_df.csv")

    except FileNotFoundError:
        print("Generating features and storing in: " + dataset_path + "/candset_features_df.csv")


        f = open(join(dataset_path, 'metadata.txt'), "r")
        print("path", dataset_path)
        LEFT_FILE = join(dataset_path, f.readline().strip())

        print("left file", LEFT_FILE)
        if LR_identical:
            RIGHT_FILE = LEFT_FILE
        else:
            RIGHT_FILE = join(dataset_path, f.readline().strip())
        print("right file", RIGHT_FILE)
        
        DUPLICATE_TUPLES = join(dataset_path, f.readline().strip())
        print("duplicate tuples", DUPLICATE_TUPLES)
        f.close()
        if run_trans==True and LR_dup_free==False and LR_identical==False:
            ltable_df, rtable_df, duplicates_df, candset_df,candset_df_l,candset_df_r = load_data(LEFT_FILE, RIGHT_FILE, DUPLICATE_TUPLES,
                                                                                              blocking_func,
                                                                                              include_self_join=True, sep=sep)
        else:
            ltable_df, rtable_df, duplicates_df, candset_df = load_data(LEFT_FILE, RIGHT_FILE, DUPLICATE_TUPLES,
                                                                                              blocking_func,
                                                                                              include_self_join=False, sep=sep)
            # print('bla')                                                                                  
            # print(ltable_df.head(5))                                                                               
            # print(rtable_df.head(5))
            # print(duplicates_df.head(5))
            # print(candset_df.head(5))           
            print(ltable_df.shape, rtable_df.shape, duplicates_df.shape, candset_df.shape)
            print("ltable", ltable_df.head(5))
            print("rtable", rtable_df.head(5))
            print("duplicates", duplicates_df.head(5))

            # print columns
            print("ltable columns", ltable_df.columns)
            print("rtable columns", rtable_df.columns)
            print("duplicates columns", duplicates_df.columns)                                
                                                                                              
            if LR_identical:
                print("removing self matches")
                candset_df = candset_df.loc[candset_df.ltable_id!=candset_df.rtable_id,:]
                candset_df.reset_index(inplace=True,drop=True)
                candset_df['_id'] = candset_df.index
        if duplicates_df is None:
            duplicates_df = pd.DataFrame(columns=["ltable_id", "rtable_id"])
        candset_features_df = gather_features_and_labels(ltable_df, rtable_df, duplicates_df, candset_df)
        candset_features_df.to_csv(join(dataset_path,"candset_features_df.csv"))
        id_df = candset_df[["ltable_id", "rtable_id"]]

        if run_trans == True and LR_dup_free == False and LR_identical==False:
            duplicates_df_r = pd.DataFrame()
            duplicates_df_r['l_id'] = rtable_df["id"]
            duplicates_df_r['r_id'] = rtable_df["id"]
            candset_features_df_r = gather_features_and_labels(rtable_df, rtable_df, duplicates_df_r, candset_df_r)
            candset_features_df_r.to_csv(join(dataset_path,"candset_features_df_r.csv"))


            duplicates_df_l = pd.DataFrame()
            duplicates_df_l['l_id'] = ltable_df["id"]
            duplicates_df_l['r_id'] = ltable_df["id"]
            candset_features_df_l = gather_features_and_labels(ltable_df, ltable_df, duplicates_df_l, candset_df_l)
            candset_features_df_l.to_csv(join(dataset_path,"candset_features_df_l.csv"))

            id_df_l = candset_df_l[["ltable_id","rtable_id"]]
            id_df_r = candset_df_r[["ltable_id","rtable_id"]]
            id_df_l.to_csv(join(dataset_path,"id_tuple_df_l.csv"))
            id_df_r.to_csv(join(dataset_path,"id_tuple_df_r.csv"))

    similarity_features_df = gather_similarity_features(candset_features_df)
    similarity_features_lr = (None,None)
    id_dfs = (None, None, None)
    if run_trans == True:
        id_dfs = (id_df, None, None)
        if LR_dup_free == False and LR_identical==False:
            similarity_features_df_l = gather_similarity_features(candset_features_df_l)
            similarity_features_df_r = gather_similarity_features(candset_features_df_r)
            features = set(similarity_features_df.columns)
            features = features.intersection(set(similarity_features_df_l.columns))
            features = features.intersection(set(similarity_features_df_r.columns))
            features = sorted(list(features))
            similarity_features_df = similarity_features_df[features]
            similarity_features_df_l = similarity_features_df_l[features]
            similarity_features_df_r = similarity_features_df_r[features]
            similarity_features_lr = (similarity_features_df_l,similarity_features_df_r)
            id_dfs = (id_df, id_df_l, id_df_r)

    true_labels = candset_features_df.gold.values
    if np.sum(true_labels)==0:
        true_labels = None
        
        
    features = time() - features
    
    zeroer = time()
    y_pred, p, r, f1 = run_zeroer(similarity_features_df, similarity_features_lr,id_dfs,
                        true_labels ,LR_dup_free,LR_identical,run_trans)
    zeroer = time() - zeroer                        
                        
    pred_df = candset_features_df[["ltable_id","rtable_id"]]
    pred_df['pred'] = y_pred
    pred_df.to_csv(join(dataset_path,"pred.csv"))
    
    log = {'features': features, 'zeroer': zeroer, 'precision': p, 'recall': r, 'f1': f1, 'dataset': dataset_alias}
    with open(log_file, 'a') as out:
       out.write(json.dumps(log)+'\n')

