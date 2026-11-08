import pyod, sklearn
from pyod.models import iforest, loda, dif
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
X1, y1 = sklearn.datasets.make_blobs(n_samples=500, n_features=3, centers=[[0, 10, 0]], cluster_std=1)
X2, y2 = sklearn.datasets.make_blobs(n_samples=500, n_features=3, centers=[[10, 0, 10]], cluster_std=1)

dataset = np.vstack((X1, X2))

#it's the same old theme since last exercise, we just make 3 vars instead of 2
uniforms_x = np.random.uniform(-10, 20, 1000)
uniforms_y = np.random.uniform(-10, 20, 1000)
uniforms_z = np.random.uniform(-10, 20, 1000)

test_data = np.column_stack((uniforms_x, uniforms_y, uniforms_z))

list_models = [
    iforest.IForest(contamination=0.02),
    loda.LODA(contamination=0.02, n_bins=15),
    dif.DIF(hidden_neurons=[32, 16], contamination=0.02),
]
names = ["if", "loda", "dif"]

fig, axs = plt.subplots(3, figsize=(8, 15), subplot_kw={"projection": "3d"})
for i, (model, name) in enumerate(zip(list_models, names)):
    model.fit(dataset)
    print(f"Test data shape for {name}: {test_data.shape}")
    scores = model.predict(test_data)
    scatter = axs[i].scatter(
        test_data[:, 0],
        test_data[:, 1],
        test_data[:, 2],
        c=scores,
        cmap="inferno",
        s=25,
        edgecolor="none",
    )
    fig.colorbar(scatter, ax=axs[i], shrink=0.6)
    axs[i].set_title(f"Model: {name}")

plt.tight_layout()
plt.savefig("comparison_v2_3d.pdf")
plt.close()