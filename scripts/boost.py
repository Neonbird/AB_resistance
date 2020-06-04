import sys
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel, chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot  as plt
from itertools import chain

file_train, file_test = sys.argv[1], sys.argv[2]

train_df = np.load(file_train)
test_df = np.load(file_test)
print(train_df.shape)

y_train = train_df[:, -1].astype('int64')
X_train = train_df[:, :-1]
y_test = test_df[:, -1].astype('int64')

features = [i for i in range(len(train_df[1]))]

# chi-square feature selection
z = zip(chi2(X_train, y_train)[0], features)
z = sorted(z, reverse=True)[:50000]
best_features = [z[i][1] for i in range(50000)] + [-1]
train_chi_sel = train_df[:, best_features]
test_chi_sel = test_df[:, best_features]
X_train_chi_sel = train_chi_sel[:, :-1]
X_test_chi_sel = test_chi_sel[:, :-1]

# LASSO REGRESSION
lasso = LogisticRegression(C=1, penalty='l1', solver='liblinear', tol=0.1, max_iter=5000)
# print("start lasso...")
lasso = lasso.fit(X_train_chi_sel, y_train)
# print("lasso fitted...")

y_pred = lasso.predict(X_test_chi_sel)
r2 = metrics.r2_score(y_test, y_pred)
print("\n### LASSO LOG-REGRESSION (tol = {}; chi-features = {}) ###".format(0.1, 50000))
print("Test Lasso r2-score is {}".format(r2))

y_pred = lasso.predict(X_train_chi_sel)
r2 = metrics.r2_score(y_train, y_pred)
print("Train Lasso r2-score is {}".format(r2))

# lasso feature selection
sel = SelectFromModel(lasso)
sel.fit(X_train_chi_sel, y_train)

X_train_selected = sel.transform(X_train_chi_sel)
X_test_selected = sel.transform(test_chi_sel[:, :-1])
print("datasets trasformed to {} features by lasso...".format(X_train_selected.shape[1]))

clf_dev = GradientBoostingClassifier()
clf_dev.fit(X_train_selected, y_train)
print('MSE score for XGBoost loss = deviance: ', metrics.mean_squared_error(y_test, clf_dev.predict(X_test_selected)))

clf_exp = GradientBoostingClassifier(loss='exponential')
clf_exp.fit(X_train_selected, y_train)
print('MSE score for XGBoost loss = exponential: ', metrics.mean_squared_error(y_test, clf_exp.predict(X_test_selected)))
