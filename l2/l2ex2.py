import sklearn, pyod 
import numpy as np
from pyod.models import knn
import matplotlib.pyplot as plt
#l2-ex2

def get_normal_and_outlier(X, y, subplot, title):
    mask_true = (y == 0)
    mask_false = (y == 1)
    X1, X2 = np.array(X[mask_true]), np.array(X[mask_false])
    subplot.scatter(X1[:, 0], X1[:, 1], color = 'green')
    subplot.scatter(X2[:, 0], X2[:, 1], color = 'red')
    subplot.set_title(title)

def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(y_true)
    recalls = []
    
    for c in classes:
        true_c = (y_true == c)
        recall_c = np.sum((y_pred == c) & true_c) / np.sum(true_c)
        recalls.append(recall_c)
        
    return float(np.mean(recalls))

X_train, X_test, y_train, y_test = pyod.utils.data.generate_data_clusters(n_train = 400, n_test = 200)

model = knn.KNN(n_neighbors = 11)

model = model.fit(X_train)
y_hat = model.predict(X_test)

y_train_hat = model.predict(X_train)
ba = balanced_accuracy(y_test, y_hat)
print(ba)

# now, let's deal with the plotting

fig, axs = plt.subplots(2, 2)

get_normal_and_outlier(X_train, y_train, axs[0, 0], "Training Ground Truth")
get_normal_and_outlier(X_test, y_test, axs[1, 0], "Testing Ground Truth")
get_normal_and_outlier(X_test, y_hat, axs[1, 1], "Testing Predictions")
get_normal_and_outlier(X_train, y_train_hat, axs[0, 1], "Training Predictions")

plt.savefig("l2e2.pdf")
