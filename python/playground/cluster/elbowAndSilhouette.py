import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_best_k(data):
    # Reshape data to make it 2-dimensional if needed
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Range of k values to try
    k_values = range(2, 11)

    # Initialize lists to store results
    inertia_values = []
    silhouette_scores = []

    # Iterate over different k values
    for k in k_values:
        # Fit KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        # Compute inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)

        # Compute silhouette score
        if k > 1:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        else:
            silhouette_scores.append(None)

    # Plot elbow method
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    # Plot silhouette method
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    # Determine the best k value using the elbow method
    # Add 2 to offset 0-based indexing
    best_k_elbow = np.argmin(np.diff(inertia_values)) + 2
    print("Best k value (Elbow Method):", best_k_elbow)

    # Determine the best k value using the silhouette method
    best_k_silhouette = k_values[np.argmax(silhouette_scores)]
    print("Best k value (Silhouette Method):", best_k_silhouette)
