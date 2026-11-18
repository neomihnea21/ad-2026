from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
d = loadmat("cardio.mat")
X, y = d['X'], d['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, random_state=122)

param_grid = {
    'ocsvm__kernel': ['linear', 'rbf', 'poly'],
    'ocsvm__nu': np.linspace(0.01, 0.5, 5),          
    'ocsvm__gamma': np.logspace(-3, -2, 5) 
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])

y_train = 1 - 2 * y_train
y_test = 1 - 2 * y_test
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=3,
    n_jobs=-1 # could I have done the edit in reverse?
)

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

print("Best Parameters found:", grid_search.best_params_)
print("Test Score:", balanced_accuracy_score(y_test, y_pred)) # WTF: test outperforms train?

