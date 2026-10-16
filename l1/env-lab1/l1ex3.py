import numpy as np
import pyod

n_train = 1000
# Change the offset to taste, lower ones lead to easier classification
X_train, _, y_train, _ = pyod.utils.data.generate_data(n_train = n_train, n_test= 0, n_features = 1, offset = 15) 

high_bar = np.quantile(X_train, 0.95)
low_bar = np.quantile(X_train, 0.05)

is_anomaly = np.zeros(1000)
for i in range(len(X_train)):
   if X_train[i] < low_bar or X_train[i] > high_bar:
     is_anomaly[i] = 1

true_positives = 0 
true_negatives = 0 
false_positives = 0
false_negatives = 0
for i in range(len(X_train)):
   if y_train[i] == 1 and is_anomaly[i] == 1: 
      true_positives +=1 
   elif y_train[i] == 0 and is_anomaly[i] == 0: 
      true_negatives += 1
   elif y_train[i] == 1 and is_anomaly[i] == 0:
      false_negatives += 1
   elif y_train[i] == 0 and is_anomaly[i] == 1:
      false_positives += 1

print(f"BA: {(true_positives / (true_positives + false_negatives) + (true_negatives / (true_negatives + false_positives)) )/ 2}")