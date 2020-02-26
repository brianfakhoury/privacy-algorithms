#!/usr/bin/env python
import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt


def generate_data(n, sigma=0.01):
    # This function generates random data in three clusters
    # They always have the same centers, but gaussian variation in each dimension
    true_centers = np.array([[0.0, 0.5], [0.2, -0.2], [-0.2, -0.2]])
    true_clusters = np.random.randint(3, size=n)
    x = np.zeros([n, 2])
    for i in range(n):
        x[i] = true_centers[true_clusters[i]]
    x = x + np.random.normal(loc=0.0, scale=sigma, size=[n, 2])

    # Project to the L1 ball of radius 1.
    for i in range(n):
        l1norm = np.sum(np.abs(x[i]))
        if l1norm > 1:
            x[i] = x[i] / l1norm
    return x, true_centers, true_clusters


def lloydsalg(x, k, T, initial_centers=None):
    # This algorithm runs Lloyd's algorithm (also called k-means)
    # If no initial centers provided, generate random ones
    if initial_centers == None:
        initial_centers = np.random.normal(loc=0.0, scale=0.3, size=[k, 2])

    # Create lists to document algorithm's history
    centers_list = [initial_centers]
    closest_list = []

    current_centers = initial_centers.copy()

    n = len(x)

    for t in range(T):
        # assign each point to its nearest cluster
        closest = np.zeros(n, dtype=np.int32)  # The type forces indices to be integers
        for i in range(n):
            distances = np.zeros(k)
            for j in range(k):
                distances[j] = np.linalg.norm(x[i, :] - current_centers[j, :])
            closest[i] = np.argmin(distances)

        closest_list.append(closest)

        # Now compute the number of points in each group, and ...
        sums = np.zeros([k, 2])
        counts = np.zeros(k, dtype=np.int32)  # The type forces counts to be integers
        for i in range(n):
            this_index = closest[i]
            counts[this_index] += 1  # counter number of points
            sums[this_index, :] += x[i, :]  # add the points for this group
        # ... compute new cluster centers
        current_centers = np.zeros([k, 2])
        for j in range(k):
            if counts[j] == 0:
                print("Whoa! Group of size zero at (t,j)= ", t, j)
                # Re-initialize this center at random.
                current_centers[j, :] = np.random.normal(loc=0.0, scale=0.1, size=2)
            else:
                # compute noisy sum
                aj_hat = sums[j, :] + np.random.normal(0, 0.05, 2)
                # compute noisu count
                nj_hat = float(counts[j]) + np.random.normal(0, 0.05)
                # DP mechanism for assigning new center, low count reinitializes the centroid
                if nj_hat <= 5:
                    current_centers[j, :] = np.random.normal(loc=0.0, scale=0.1, size=2)
                else:
                    current_centers[j, :] = aj_hat / nj_hat

        centers_list.append(current_centers)

    closest_list.append(np.zeros(n, dtype=np.int8))
    return centers_list, closest_list


# Let's generate some data!
x, true_centers, true_clusters = generate_data(400, sigma=0.2)
# Plot the data. Colors represent the true cluster of origin
# The clusters overlap significantly for sigma bigger than about 0.1
plt.scatter(x[:, 0], x[:, 1], c=true_clusters, alpha=0.5)
plt.scatter(true_centers[:, 0], true_centers[:, 1], marker='+', c='orange', s=200)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
# Now run non-noisy k-means

centers_list, closest_list = lloydsalg(x, k=3, T=10)
# Plot the centers and clusters at each stage.

for t in range(len(centers_list) - 1):
    centers = centers_list[t]
    closest = closest_list[t]
    plt.scatter(x[:, 0], x[:, 1], c=closest, alpha=0.5)
    plt.scatter(true_centers[:, 0], true_centers[:, 1], marker='+', c='red', s=300)
    plt.scatter(centers[:, 0], centers[:, 1], marker='+', c='blue', s=300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
# The last plot shows the final cluster centers with colors representing original clusters.
final_centers = centers_list[-1]
plt.scatter(x[:, 0], x[:, 1], c=true_clusters, alpha=0.5)
plt.scatter(true_centers[:, 0], true_centers[:, 1], marker='+', c='red', s=300)
plt.scatter(final_centers[:, 0], final_centers[:, 1], marker='+', c='blue', s=300)
plt.gca().set_aspect('equal', adjustable='box')
plt.show


# #
