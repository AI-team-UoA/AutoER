import pandas as pd

trials = pd.read_csv('../data/trials.csv', sep=',')
trials = trials[trials['sampler']=='gridsearch']
trials = trials[['clustering', 'lm', 'k', 'threshold']]

trials.drop_duplicates(inplace=True)

print("Gridsearch trials are:",  trials.shape)

import argparse
parser = argparse.ArgumentParser(description='Create test data for benchmarking')
parser.add_argument('--data', type=str, default='dbpedia')
args = parser.parse_args()

dataset = args.data

dataspecs = pd.read_csv(f'../data/{dataset}_features.csv', sep=',')

cartesian_product = trials.assign(key=1).merge(dataspecs.assign(key=1), on='key').drop('key', axis=1)


final_df = cartesian_product[[
    'Dataset', 'clustering', 'lm', 'k', 'threshold',
    'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
    'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity',
    'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
    'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity'
]]

print(cartesian_product.shape)
cartesian_product.drop_duplicates(inplace=True)
print(cartesian_product.shape)

final_df.to_csv(f'../data/{dataset}_test_data.csv', sep=',', index=False)
