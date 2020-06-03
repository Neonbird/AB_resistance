import sys
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

file_train, file_test = sys.argv[1], sys.argv[2]

train_df = np.load(file_train)
test_df = np.load(file_test)

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


# LASOO REGRESSION
lasso = LogisticRegression(C=1, penalty='l1', solver='liblinear', tol=0.1, max_iter=5000)
lasso.fit(X_train_chi_sel, y_train)

y_pred = lasso.predict(X_test_chi_sel)
r2 = metrics.r2_score(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
print("### LOG LASSO REGRESSION ###")
print("Test Lasso r2-score is {}".format(r2))
print("Test Lasso RMSE is {}".format(rmse))

y_pred = lasso.predict(X_train_chi_sel)
r2 = metrics.r2_score(y_train, y_pred)
rmse = metrics.mean_squared_error(y_train, y_pred, squared=False)
print("Train Lasso r2-score is {}".format(r2))
print("Train Lasso RMSE is {}".format(rmse))

# lasso feature selection
sel = SelectFromModel(lasso)
sel.fit(X_train_chi_sel, y_train)
selected_feat = X_train_chi_sel[:, sel.get_support()]

X_train_selected = sel.transform(X_train_chi_sel)
X_test_selected = sel.transform(test_chi_sel[:, :-1])
print("datasets trasformed to {} features...".format(X_train_selected.shape[1]))

# Random Forest
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train_selected, y_train)

y_pred = clf.predict(X_test_selected)
r2 = metrics.r2_score(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print("\nRANDOM FOREST")
print('selected features by lasso: {}'.format(X_train_selected.shape[1]))
print("Test Random forest r2-score is {}".format(r2))
print("Test Random forest RMSE is {}".format(rmse))

y_pred = clf.predict(X_train_selected)
r2 = metrics.r2_score(y_train, y_pred)
rmse = metrics.mean_squared_error(y_train, y_pred, squared=False)

print("Train Random forest r2-score is {}".format(r2))
print("Train Random forest RMSE is {}".format(rmse))

print("DONE")
