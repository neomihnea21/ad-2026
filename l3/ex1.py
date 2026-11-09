
import numpy as np 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# in most linear algebra texts, the norm is squared
# but we care about the vector in absolute terms, not in units of v
def get_projection(u, v):
   return np.dot(u, v) / (np.linalg.norm(v))

def single_histogram_score(x, basis_vector, distro, n_bins, lo=-4, hi=4):
   proj = get_projection(x, basis_vector)
   bin_index = int(np.clip(((proj - lo) / (hi - lo)) * n_bins, 0, n_bins - 1))
   return distro[bin_index]


X_train, _ = make_blobs(500, centers= [[0, 0]])

basis = []
for _ in range(5):
   arg = np.random.uniform(0, 2*np.pi)
   basis.append([np.cos(arg), np.sin(arg)])

bins = 13
histograms = []
distros = []
for basis_vector in basis:
   proj = np.array([get_projection(x, basis_vector) for x in X_train])
   histogram, _ = np.histogram(proj, bins = bins, range = (-5, 5))
   distro = histogram / np.sum(histogram) 
   distros.append(distro)
   histograms.append(histogram)

test_data = np.random.uniform(-5, 5, (300, 2))

final_scores = []
for i, distro in enumerate(distros):
   scores = np.array([single_histogram_score(x, basis[i], distro, bins) for x in test_data])
   final_scores.append(scores)

final_scores = np.array(final_scores)
final_mean_scores = np.mean(final_scores, axis=0)

scatter = plt.scatter(test_data[:, 0], test_data[:, 1], c=final_mean_scores, cmap='inferno')
plt.title(f'{bins} bins - LODA scores for histogram {i}')
plt.colorbar(scatter)
plt.savefig(f'ex1/Anomaly_Score_{bins}_bins.pdf')
plt.clf()