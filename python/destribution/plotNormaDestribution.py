import numpy as np
import matplotlib.pyplot as plt

# Generate random normal data
mean = 0
std_dev = 1
num_samples = 1000
random_data = np.random.normal(mean, std_dev, num_samples)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(random_data, bins=90, density=True, color='skyblue', alpha=0.7)
plt.title('Histogram of Random Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Plot KDE plot
plt.figure(figsize=(8, 6))
sns.kdeplot(random_data, color='skyblue', shade=True)
plt.title('Kernel Density Estimation of Random Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()
