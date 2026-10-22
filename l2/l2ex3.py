import pyod, sklearn, numpy as np
import sklearn.datasets 
import matplotlib.pyplot as plt
from pyod.models import knn, lof
close, _ = sklearn.datasets.make_blobs(
    n_samples = 200, 
    centers = [[-10, -10]],
    cluster_std = 2, 
)
far, _ = sklearn.datasets.make_blobs(
    n_samples = 100, 
    centers= [[10, 10]],
    cluster_std=6,
)
print(np.shape(close))
model = knn.KNN(
    contamination = 0.07, 
    n_neighbors = 11
)

lof_model = lof.LOF(
    n_neighbors = 7,
    contamination=0.07
)

far_knn_labels = lof_model.fit_predict(far)
close_knn_labels = lof_model.fit_predict(close)

far_normal_mask = (far_knn_labels == 0)
far_outlier_mask = (far_knn_labels != 0)

close_normal_mask = (close_knn_labels == 0)
close_outlier_mask = (close_knn_labels != 0)

plt.scatter(
    far[far_normal_mask, 0], 
    far[far_normal_mask, 1], 
    color='green', 
    label='normal'
)

plt.scatter(
    far[far_outlier_mask, 0], 
    far[far_outlier_mask, 1], 
    color='red', 
    label='outlier'
)

plt.scatter(
    close[close_normal_mask, 0], 
    close[close_normal_mask, 1], 
    color='green',
)

plt.scatter(
    close[close_outlier_mask, 0], 
    close[close_outlier_mask, 1], 
    color='red', 
)
plt.legend()
plt.savefig("l2e3-LOF.pdf")