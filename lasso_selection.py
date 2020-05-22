import sys
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
import numpy as np

file_train, file_test = sys.argv[1], sys.argv[2]

train_df = np.load(file_train)
y_train = train_df[:, -1].astype('int64')
X_train = train_df[:, :-1]

print(y_train.dtype)

sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel.estimator_.coef_ == 0)))

X_train_selected = sel.transform(X_train)
np.save(f'lasso_train_{file_train}', X_train_selected)

test_df = np.load(file_test)
y_test= test_df[:, -1].astype('int64')
X_test = test_df[:, :-1]

X_test_selected = sel.transform(X_test)
np.save(f'lasso_test_{file_test}', X_test_selected)
