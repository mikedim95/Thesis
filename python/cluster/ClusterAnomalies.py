import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function to load dataset


def load_dataset(url):
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        data = np.loadtxt(response.iter_lines())
        return data
    else:
        print("Failed to load the dataset.")
        return None

# Function to identify anomalies


def identify_anomalies(data, labels, centroids, threshold_percentile=80):
    distances = np.zeros((len(data),))
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        distances[labels == i] = np.linalg.norm(
            cluster_points - centroid, axis=1)

    threshold = np.percentile(distances, threshold_percentile)
    anomalies_indices = np.where(distances > threshold)[0]
    return data[anomalies_indices]


# Load the dataset
url = "https://cs.joensuu.fi/sipu/datasets/s1.txt"
data = load_dataset(url)

if data is not None:
    # Apply KMeans clustering
    # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=15)
    kmeans.fit(data)

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Identify anomalies
    anomalies = identify_anomalies(data, labels, centroids)

    # Visualize the clusters and anomalies
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=5, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=100, color='red', label='Centroids')
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                color='black', marker='o', s=50, label='Anomalies')
    plt.title("Clustering Result with Anomalies")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
