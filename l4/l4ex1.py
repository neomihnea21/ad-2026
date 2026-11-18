from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
model = OCSVM(kernel='rbf', contamination=0.15)


rng = np.random.default_rng(851651516)
n_inliers = 425
X_inliers = rng.multivariate_normal(
    mean=np.zeros(3),
    cov=2 * np.eye(3),
    size=n_inliers
)
y_inliers = np.zeros(n_inliers, dtype=int)
n_outliers = 75
X_outliers = rng.uniform(low=-4.0, high=4.0, size=(n_outliers, 3))
y_outliers = np.ones(n_outliers, dtype=int)

X = np.vstack([X_inliers, X_outliers])
y = np.concatenate([y_inliers, y_outliers])

X_train, X_test, y_train, y_test = train_test_split( # or other anomalies, to taste
    X, y, test_size=0.2, random_state=851651516, stratify=y
)


model.fit(X_train)

y_hat = model.predict(X_test)
y_train_hat = model.predict(X_train)
ba = balanced_accuracy_score(y_test, y_hat)
roc_auc = roc_auc_score(y_test, y_hat)

print(f"BA: {ba}")
print(f"ROC-AUC: {roc_auc}")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

def plot_3d(ax, X, y, title, cmap):
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], 
        c=y, 
        edgecolor='k', 
        alpha=0.7, 
        cmap = cmap,
    )
    ax.set_title(title)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_zlabel('feature 2')

plots = [
    (X_train, y_train,     'True Training'),
    (X_train, y_train_hat, 'Predicted Training'),
    (X_test,  y_test,      'True Testing'),
    (X_test,  y_hat,       'Predicted Testing'),
]

fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(14, 10))

for ax, (X, y, title) in zip(axes.ravel(), plots):
    plot_3d(ax, X, y, title, 'inferno')

plt.tight_layout()
plt.show()

fig.savefig("l4ex1.pdf")
