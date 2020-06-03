import sys
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import matplotlib.pyplot  as plt
import csv

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
print("\n### LASSO LOG-REGRESSION ###")
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

importance = np.abs(lasso.coef_)
features_sub_X_train = []
features_sub_X_test = []
r2_sub_train = []
r2_sub_test = []

X_train_selected_trans = X_train_chi_sel.transpose()
X_test_selected_trans = X_test_chi_sel.transpose()

for i in range(5000):
    ind = importance[0].argsort()[-i]
    features_sub_X_train.append(X_train_selected_trans[ind].tolist())
    features_sub_X_test.append(X_test_selected_trans[ind].tolist())

    if i % 50 == 0:
        features_sub_X_train_np = np.array(features_sub_X_train)
        # features_sub_X_train_np = np.squeeze(features_sub_X_train_np, axis=0)
        features_sub_X_test_np = np.array(features_sub_X_test)
        # features_sub_X_test_np = np.squeeze(features_sub_X_test_np, axi
        lasso_test = LogisticRegression(C=1, penalty='l1', solver='liblinear', tol=0.1, max_iter=5000)
        lasso_test.fit(features_sub_X_train_np.transpose(), y_train)

        y_pred_on_test = lasso_test.predict(features_sub_X_test_np.transpose())
        r2 = metrics.r2_score(y_test, y_pred_on_test)
        r2_sub_test.append(r2)

        y_pred_on_train = lasso_test.predict(features_sub_X_train_np.transpose())
        r2 = metrics.r2_score(y_train, y_pred_on_train)
        r2_sub_train.append(r2)

with open("r2_stats.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(r2_sub_test)
    wr.writerow(r2_sub_train)

plt.figure()
plt.plot(range(1, 5001, 50), r2_sub_test, c='r')
plt.plot(range(1, 5001, 50), r2_sub_train, c='b')
plt.savefig('r2_stats.png')

print("DONE")







# # Random Forest
# clf = RandomForestRegressor(random_state=0)
# clf.fit(X_train_selected, y_train)
# # print("random forest fitted...")
#
# y_pred = clf.predict(X_test_selected)
# r2 = metrics.r2_score(y_test, y_pred)
# rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
#
# print("\nRANDOM FOREST")
# print('selected features by lasso: {}'.format(X_train_selected.shape[1]))
# print("Test Random forest r2-score is {}".format(r2))
# print("Test Random forest RMSE is {}".format(rmse))
#
# y_pred = clf.predict(X_train_selected)
# r2 = metrics.r2_score(y_train, y_pred)
# rmse = metrics.mean_squared_error(y_train, y_pred, squared=False)
#
# print("Train Random forest r2-score is {}".format(r2))
# print("Train Random forest RMSE is {}".format(rmse))

print("DONE")
