import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate some sample data
# Example normal distribution data
data = np.random.normal(loc=0, scale=1, size=1000)

# Fit Kernel Density Estimation (KDE) model
kde = KernelDensity(bandwidth=0.5)  # You can adjust the bandwidth parameter
kde.fit(data.reshape(-1, 1))  # Reshape data to match sklearn's input format

# Generate points for plotting the KDE curve
x = np.linspace(min(data), max(data), 1000)
log_dens = kde.score_samples(x.reshape(-1, 1))

# Calculate anomaly threshold
# Example: 5th percentile as anomaly threshold
threshold = np.percentile(log_dens, 5)

# Identify anomalies
anomalies_indices = np.where(log_dens < threshold)[0]
anomalies = x[anomalies_indices]

# Visualize the KDE curve and anomalies
plt.plot(x, np.exp(log_dens), label='KDE')
plt.scatter(anomalies, np.exp(kde.score_samples(
    anomalies.reshape(-1, 1))), color='red', label='Anomalies')
plt.title('Kernel Density Estimation')
plt.xlabel('Data')
plt.ylabel('Density')
plt.legend()
plt.show()
