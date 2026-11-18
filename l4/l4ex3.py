from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np

from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

d = loadmat("shuttle.mat")
X, y = d['X'], d['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=34152)

train_scaler = StandardScaler()
test_scaler = StandardScaler()
X_train = train_scaler.fit_transform(X_train)

X_test = test_scaler.fit_transform(X_test)

small_svm = OCSVM(nu = 0.25)

large_svm = DeepSVDD(n_features = 9, c = 0.8, epochs = 10, batch_size = 16)

small_svm.fit(X_train, y_train)
large_svm.fit(X_train, y_train)

y_pred_small = small_svm.predict(X_test)
y_pred_large = large_svm.predict(X_test)

ba_small = balanced_accuracy_score(y_test, y_pred_small)
ba_large = balanced_accuracy_score(y_test, y_pred_large) # deepSVDD is shooting flies with a howitzer, perhaps we need a more involved dataset?

roc_small = roc_auc_score(y_test, y_pred_small)
roc_large = roc_auc_score(y_test, y_pred_large)
print(f"BA's: {ba_small}, {ba_large}")
print(f"ROC's: {roc_small}, {roc_large}")