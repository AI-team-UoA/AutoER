import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

dataset = 'D10'

df = pd.read_csv('/home/gpapadak/autoconf/alltrials.csv', sep=';')
data = df[df['f1']!=0]

testingD1 = data[data['dataset']==dataset]
trainingD1 = data[data['dataset']!=dataset]

train_y = trainingD1[['f1']]
train_X = trainingD1[['clustering', 'lm', 'k', 'threshold', 'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
                'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity', 
                'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
                'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity']]
testing_X = testingD1[['clustering', 'lm', 'k', 'threshold', 'InputEntityProfiles', 'NumberOfAttributes', 'NumberOfDistinctValues', 
                'NumberOfNameValuePairs', 'AverageNVPairsPerEntity', 'AverageDistinctValuesPerEntity', 
                'AverageNVpairsPerAttribute', 'AverageDistinctValuesPerAttribute', 'NumberOfMissingNVpairs', 
                'AverageValueLength', 'AverageValueTokens', 'MaxValuesPerEntity']]
y_true = testingD1[['f1']]

scaler = StandardScaler()
train_X = scaler.fit_transform(pd.get_dummies(train_X))
testing_X = scaler.transform(pd.get_dummies(testing_X))

clf = LinearRegression().fit(train_X, train_y)
y_pred = clf.predict(testing_X)

print('LR', r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred))

for alpha in [0.0001, 0.001,0.01, 0.1, 1, 10]:
    clf = Lasso(alpha=alpha).fit(train_X, train_y)
    y_pred = clf.predict(testing_X)
    print('Lasso', alpha, r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred))
	
for alpha in [0.0001, 0.001,0.01, 0.1, 1, 10]:
    clf = Ridge(alpha=alpha).fit(train_X, train_y)
    y_pred = clf.predict(testing_X)
    print('Ridge', alpha, r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred))
	
for k in ['linear', 'rbf', 'poly', 'sigmoid']:
    for c_value in [1, 10, 100, 1000, 10000]:
        for e_value in [0.1, 0.2, 0.3, 0.4]:
            for gamma_value in [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9]:
                clf = SVR(C=c_value, epsilon=e_value, gamma=gamma_value, kernel=k).fit(train_X, train_y)
                y_pred = clf.predict(testing_X)
                print('SVR', c_value, e_value, gamma_value, k, r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred))
