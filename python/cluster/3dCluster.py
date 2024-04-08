import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

# Load the CSV dataset
df = pd.read_csv('insurance.csv')

# Choose four columns for 3D visualization (including the color column)
# Replace 'column1', 'column2', 'column3', and 'color_column' with the names of your chosen columns
data = df[['age', 'bmi', 'charges', 'smoker']].values

# Apply KMeans clustering (excluding the color column)
kmeans = KMeans(n_clusters=3)
kmeans.fit(data[:, :-1])  # Exclude the last column (color column)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Apply Local Outlier Factor (LOF) for anomaly detection
# Adjust parameters as needed
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# Exclude the last column (color column)
anomaly_scores = lof.fit_predict(data[:, :-1])

# Separate anomalies from the data
# Exclude the last column (color column)
anomaly_data = data[anomaly_scores == -1, :-1]

# Visualize the clusters and anomalies in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all data points with the same color and style
scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                     c='green', s=50, label='All Points', picker=5)


# Plot centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
           marker='x', c='red', s=200, label='Centroids')

# Plot anomalies separately with a different color or style to stand out
ax.scatter(anomaly_data[:, 0], anomaly_data[:, 1],
           anomaly_data[:, 2], c='black', s=100, label='Anomalies')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cluster Visualization with Anomaly Detection')

plt.legend()

# Function to display the values of the clicked data point


def on_pick(event):
    ind = event.ind[0]
    x, y, z = data[ind, :3]  # Extract x, y, z coordinates
    values = data[ind, :]  # Get all values for the clicked point
    print("Clicked Data Point:")
    print("Coordinates (X, Y, Z):", x, y, z)
    print("All Values:", values)


# Connect the pick event to the figure
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
