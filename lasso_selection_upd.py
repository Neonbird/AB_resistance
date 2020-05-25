import sys
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel, chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np

file_train, file_test = sys.argv[1], sys.argv[2]

train_df = np.load(file_train)
test_df = np.load(file_test)
n_features = 100000
print("train df shape is {}; test df shape is {}".format(train_df.shape, test_df.shape))

y_train = train_df[:, -1].astype('int64')
X_train = train_df[:, :-1]
y_test = test_df[:, -1].astype('int64')

features = [i for i in range(len(train_df[1]))]
z = zip(chi2(X_train, y_train)[0], features)
z = sorted(z, reverse=True)[:n_features]
best_features = [z[i][1] for i in range(n_features)] + [-1]
train_chi_sel = train_df[:,best_features]
test_chi_sel = test_df[:,best_features]
print("train chi df shape is {}; test chi df shape is {}".format(train_chi_sel.shape, test_chi_sel.shape))


sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
X_train_chi_sel = train_chi_sel[:, :-1]
sel.fit(X_train_chi_sel, y_train)
selected_feat = X_train_chi_sel[:, sel.get_support()]
print("selected_feat shape is {}".format(selected_feat.shape))
print('total features: {}'.format((X_train_chi_sel.shape[1])))
print('selected features: {}'.format(selected_feat.shape[1]))

X_train_selected = sel.transform(X_train_chi_sel)
print("X_train_selected is {}".format(X_train_selected.shape))
X_test_selected = sel.transform(test_chi_sel[:, :-1])
print("X_test_selected is {}".format(X_test_selected.shape))

# np.save(f'lasso_chi_train_{file_train}', np.append(X_train_selected, [train_df[:, -1]], axis=1))
# np.save(f'lasso_chi_test_{file_test}', np.append(X_test_selected, [test_df[:, -1]], axis=1))

clf = RandomForestRegressor(random_state=0)
clf.fit(X_train_selected, y_train)
y_pred = clf.predict(X_test_selected)

r2 = metrics.r2_score(y_test, y_pred)
explained_variance_score = metrics.explained_variance_score(y_test, y_pred)

print("r2-score is {}".format(r2))
print("explained_variance_score is {}".format(explained_variance_score))

param_grid = {
    'n_estimators': [100, 200, 500],
    'criterion': ["mse", "mae"],
    'min_samples_split': [2, 5, 10, 20, 50],
    'max_features': ['auto', 'sqrt', 'log2'],
}
CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train_selected, y_train)
y_pred_cv = CV_rfc.predict(X_test_selected)

r2 = metrics.r2_score(y_test, y_pred_cv)
explained_variance_score = metrics.explained_variance_score(y_test, y_pred_cv)

print("r2-score is {}".format(r2))
print("explained_variance_score is {}".format(explained_variance_score))

print("DONE")
