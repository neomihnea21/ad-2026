#d) ex4 
import numpy as np 
import pyod
import matplotlib.pyplot as plt
#
def get_psd_matrix(n):
   X = np.random.uniform(4, 5, (n, n)) # or some other values
   return X.T @ X

n_dim = 11
# points_train is already from a standard Gaussian, so we don't need anything more (it already includes outliers)
# (the outliers aren't from a Gaussian, but they shouldn't be, either)
points_train, _, labels, _ = pyod.utils.data.generate_data(n_train = 1000, n_features=n_dim)

mean_vector = np.ones(n_dim)

covariance_matrix = get_psd_matrix(n_dim)

L = np.linalg.cholesky(covariance_matrix)
L_reverse = np.linalg.inv(L)


points = np.array([L @ x + mean_vector for x in points_train])


z_scores = []
for x in points:
   score = (x-mean_vector).T @ L_reverse @ (x-mean_vector)
   z_scores.append(score)


high_bar = np.quantile(z_scores, 0.95)
low_bar = np.quantile(z_scores, 0.05)

is_anomaly = np.zeros(1000)
for i in range(len(z_scores)):
   if z_scores[i] < low_bar or z_scores[i] > high_bar:
     is_anomaly[i] = 1

true_positives = 0 
true_negatives = 0 
false_positives = 0
false_negatives = 0
for i in range(len(z_scores)):
   if labels[i] == 1 and is_anomaly[i] == 1: 
      true_positives +=1 
   elif labels[i] == 0 and is_anomaly[i] == 0: 
      true_negatives += 1
   elif labels[i] == 1 and is_anomaly[i] == 0:
      false_negatives += 1
   elif labels[i] == 0 and is_anomaly[i] == 1:
      false_positives += 1

print(f"BA: {(true_positives / (true_positives + false_negatives) + (true_negatives / (true_negatives + false_positives)) )/ 2}")