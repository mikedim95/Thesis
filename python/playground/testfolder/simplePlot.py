import numpy as np
import matplotlib.pyplot as plt

# Load the time series dataset from a file
time_series_data = np.loadtxt("datasets/largeDataset.txt")

# Plot the time series data against the index
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, color='blue', label='Time Series Data')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Time Series Plot')
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.show()
