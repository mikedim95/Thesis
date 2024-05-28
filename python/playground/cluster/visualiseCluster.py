import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = np.loadtxt("s1.txt")

# Visualize the dataset (optional)
plt.scatter(data[:, 0], data[:, 1], s=5)
plt.title("Dataset Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Apply KMeans clustering
# You can adjust the number of clusters as needed
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, s=5, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
            s=100, color='red', label='Centroids')
plt.title("Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
