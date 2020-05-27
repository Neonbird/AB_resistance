import sys
from sklearn.linear_model import Lasso, LogisticRegression
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

for n_features in (50000, 150000):
    # chi-square feature selection
    z = zip(chi2(X_train, y_train)[0], features)
    z = sorted(z, reverse=True)[:n_features]
    best_features = [z[i][1] for i in range(n_features)] + [-1]
    train_chi_sel = train_df[:, best_features]
    test_chi_sel = test_df[:, best_features]
    X_train_chi_sel = train_chi_sel[:, :-1]
    for max_iter in [10000]:
        # lasso feature selection
        for tol in (0.00001, 0.0001, 0.01, 0.1):
            sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear', tol=tol, max_iter=max_iter))
            sel.fit(X_train_chi_sel, y_train)
            selected_feat = X_train_chi_sel[:, sel.get_support()]
            print("\n### RESULTS FOR {} CHI-FEATURES AND {} TOL ###".format(n_features, tol))
            print('selected features: {}'.format(selected_feat.shape[1]))

            X_train_selected = sel.transform(X_train_chi_sel)
            X_test_selected = sel.transform(test_chi_sel[:, :-1])

            clf = RandomForestRegressor(random_state=0)
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)

            r2 = metrics.r2_score(y_test, y_pred)
            explained_variance_score = metrics.explained_variance_score(y_test, y_pred)
            rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

            print("r2-score is {}".format(r2))
            print("RMSE is {}".format(rmse))
            print("explained_variance_score is {}".format(explained_variance_score))

print("DONE")
