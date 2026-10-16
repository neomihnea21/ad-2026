import numpy as np
import matplotlib.pyplot as plt 
import pyod 
from pyod.models import knn
from sklearn.metrics import confusion_matrix, roc_curve, auc
# ex1

# we can set the standard deviation to taste, dependimg how many outliers are needed
X_train, X_test, y_train, y_test = pyod.utils.data.generate_data(n_train=400, n_test = 100, n_features =2, contamination=0.1) 

mask_0 = (y_train == 0)
mask_1 = (y_train == 1)

plt.figure(figsize=(12, 6))
plt.scatter(X_train[mask_0, 0], X_train[mask_0, 1], c='orchid', label='normal')
plt.scatter(X_train[mask_1, 0], X_train[mask_1, 1], c='lime', label='anomaly')
plt.legend()
plt.savefig("ex1.pdf")

#ex2 
model = knn.KNN(contamination=0.1)
model.fit(X_train)

decisions = model.decision_function(X_test)
test_preds = model.predict(X_test)
matrix = confusion_matrix(y_test, test_preds)

print(test_preds)
print(matrix)
ba = (matrix[0, 0]/(matrix[0, 0] + matrix[0, 1])  + matrix[1, 1] / (matrix[1, 0]+ matrix[1, 1])) / 2
print(f"BA: {ba}")

fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.close()
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="magenta", lw=2, linestyle="--", label="Random Chance")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig("roc.pdf")