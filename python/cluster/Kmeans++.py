import numpy as np


def kmeans_plusplus_1d(data, k):
    centroids = [np.random.choice(data)]

    while len(centroids) < k:
        distances = np.array(
            [min([abs(x - c) for c in centroids]) for x in data])
        probabilities = distances / np.sum(distances)
        next_centroid_index = np.random.choice(len(data), p=probabilities)
        centroids.append(data[next_centroid_index])

    return np.array(centroids)


# Read dataset from file
file_path = "datasets/largeDataset.txt"
data_1d = np.loadtxt(file_path)

# Number of clusters
k = 3

# Perform K-means++ initialization
initial_centroids_1d = kmeans_plusplus_1d(data_1d, k)
print("Initial centroids for 1D dataset:")
print(initial_centroids_1d)
