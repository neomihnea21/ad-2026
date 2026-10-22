import numpy as np
import matplotlib.pyplot as plt

a = 3
b = 2 
c = 1 # set these to taste

def get_leverages(points):
    n_points = np.shape(points)[0]
    X = np.ones((n_points, 3))
    X[:, 1] = points[:, 0]  
    X[:, 2] = points[:, 1]  

    H = X @ np.linalg.inv(X.T @ X) @ X.T 
    leverage_scores = np.array([H[i, i] for i in range(np.shape(H)[0])])

    return leverage_scores

regular = []
off_x = []
off_y = []
off_both = []

sigma = 0.1
large_sigma = 2

for i in range(100):
    x = np.random.normal(1, 0.1, 1)
    y = np.random.normal(1, 0.1, 1)
    z = a*x + b*y + c + np.random.normal(0, sigma, 1)
    regular.append([x, y, z])

for i in range(5):
    x = np.random.uniform(0, 2, 1)
    y = np.random.uniform(0, 2, 1)
    z = a*x + b*y + c + np.random.uniform(0, sigma, 1)
    off_x.append([x, y, z])

for i in range(5):
    x = np.random.normal(1, 0.1, 1)
    y = np.random.normal(1, 0.1, 1)
    z = a*x + b*y + c + np.random.normal(0, large_sigma, 1)
    off_y.append([x, y, z])

for i in range(5):
    x = np.random.uniform(0, 2, 1)
    y = np.random.uniform(0, 2, 1)
    z = a*x + b*y + c + np.random.normal(0, large_sigma, 1)
    off_both.append([x, y, z])

regular = np.array(regular).squeeze(2)
off_x = np.array(off_x).squeeze(2)
off_y = np.array(off_y).squeeze(2)
off_both = np.array(off_both).squeeze(2)

all_points = np.concatenate([regular, off_x, off_y, off_both], axis=0)

leverages = get_leverages(all_points)
scaling_factor = 100
colors = ['green', 'skyblue', 'orange', 'darkred']
group_names = ['close-both', 'off-x', 'off-y', 'off-both'] 
group_sizes = [100, 5, 5, 5]

group_prefix_sums = [0] + list(np.cumsum(group_sizes))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

for g_idx, color in enumerate(colors):
    start = group_prefix_sums[g_idx]
    end = group_prefix_sums[g_idx + 1]
    
    group_points = all_points[start:end]
    group_leverages = leverages[start:end]
    
    ax.scatter(
        group_points[:, 0], 
        group_points[:, 1], 
        group_points[:, 2], 
        s=np.pi * (group_leverages * scaling_factor) ** 2, 
        c=color, 
        label=group_names[g_idx]
    )

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(title="Categories")
plt.savefig("leverages-3d.pdf")