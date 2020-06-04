import sys
import numpy as np
import argparse
from sklearn.feature_selection import chi2
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import random

parser = argparse.ArgumentParser(description='chi2 and random sampling compare')
parser.add_argument('file_name', type=str, help='dataframe')
parser.add_argument('file_name_test', type=str)
args = parser.parse_args()

file_name, file_name_test = args.file_name, args.file_name_test

df = np.load(file_name)
y = df[:,-1]
X = df[:,:-1]
features = [i for i in range(len(df[1]))]
z = zip(chi2(X, y)[0], features)
df_test = np.load(file_name_test)
scores = []
z = sorted(z, reverse=True)

for n_features in range(1000, 10001, 1000):
    new_z = z[:n_features]
    best_features = [new_z[i][1] for i in range(n_features)]
    X_train = df[:,best_features]
    X_test = df_test[:,best_features] 
    y_train = y
    y_test = df_test[:,-1]
    
    clf = CatBoostClassifier()
    clf.fit(X_train, y_train)
    scores.append(roc_auc_score(y_test, clf.predict(X_test))) 

np.save('chi2_catboost_scores.npy', scores)
