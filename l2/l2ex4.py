import scipy 
import numpy as np
from sklearn.model_selection import train_test_split
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization
from sklearn.metrics import balanced_accuracy_score
d = scipy.io.loadmat("cardio.mat")
X, y = d['X'], d['y']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# in order for KNN to work we do normalization
def normalize(X):
    row_means = np.mean(X, axis=1, keepdims=True)
    row_stds = np.std(X, axis=1, keepdims=True)
    
    X = (X - row_means) / np.where(row_stds == 0, 1.0, row_stds)
    return X

X_train = normalize(X_train)
X_test = normalize(X_test)

# this can be KNN, or we can toy with the numbers.
models = [LOF(n_neighbors = 3*i + 20) for i in range(10)] 

train_scores = np.zeros((X_train.shape[0], len(models)))

for i, model in enumerate(models):
    model.fit(X_train)
    train_scores[:, i] = model.decision_scores_

train_scores = normalize(train_scores)

test_scores_matrix = np.zeros((X_test.shape[0], len(models)))

for i, clf in enumerate(models):
    test_scores_matrix[:, i] = clf.decision_function(X_test)

#as requested, standardize the test scores
test_scores_norm = standardizer(test_scores_matrix)
mean_moe_test_scores = average(test_scores_norm)
max_moe_test_scores = maximization(test_scores_norm)

good_samples_rate = 90
mean_threshold = np.quantile(mean_moe_test_scores, good_samples_rate / 100) # 100 (1 - contamination_rate)
max_threshold = np.quantile(max_moe_test_scores, good_samples_rate / 100)

final_test_labels_1 = (mean_moe_test_scores > mean_threshold).astype(int)
final_test_labels_2 = (max_moe_test_scores > max_threshold).astype(int)

ba_1 = balanced_accuracy_score(final_test_labels_1, y_test)
ba_2 = balanced_accuracy_score(final_test_labels_2, y_test)
print(f"Mean BA: {ba_1}, max BA: {ba_2}")