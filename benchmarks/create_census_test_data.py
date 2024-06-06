import pandas as pd

trials = pd.read_csv('../data/trials.csv', sep=',')
trials = trials[trials['sampler']=='gridsearch']
trials = trials[['clustering', 'lm', 'k', 'threshold']]

trials.drop_duplicates(inplace=True)

print("Gridsearch trials are:",  trials.shape)

census_dataspecs = pd.read_csv('../data/SyntheticDatasetsFeatures.csv', sep=',')

cartesian_product = trials.assign(key=1).merge(census_dataspecs.assign(key=1), on='key').drop('key', axis=1)


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

final_df.to_csv('../data/census_test_data.csv', sep=',', index=False)
