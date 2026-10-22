import numpy as np 

a = 5
b = 1


points_v1 = []
points_v2 = []
points_v3 = []
points_v4 = []

all_points = []
n_samples = 1000

sigmas = [0.2, 3]

for i in range(n_samples):
    t = np.random.uniform(0, 6, 1)
    base_point = np.array([t, a*t+b])
    
    small_noise = np.random.normal(0, sigmas[0], (2, 1))
    base_point += small_noise
    
    all_points.append(base_point)

all_points = np.array(all_points)
all_points = all_points.squeeze(2)

X = np.ones((n_samples, 2))
X[:, 1] = all_points[:, 1]

H = X @ np.linalg.inv(X.T @ X) @ X.T 
print(np.shape(H))
leverage_scores = np.array([H[i, i] for i in range(np.shape(H)[0])])

print(f"Max leverage score: {max(leverage_scores)}, attained for {all_points[np.argmax(leverage_scores)]}")

