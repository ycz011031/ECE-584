import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Path to your data file
data_file = 'data.txt'  # Change this if your file has a different name

# Read delay data only (ignore index)
delays = []

with open(data_file, 'r') as f:
    for line in f:
        # Skip empty lines
        if line.strip() == "":
            continue
        parts = line.split()
        if len(parts) == 2:
            _, delay = parts  # ignore the first part
            delays.append(float(delay))

# Convert to numpy array for easier math
delays = np.array(delays)

# Mean and standard deviation of the delays
mean_delay = np.mean(delays)
std_delay = np.std(delays)

# Create histogram of the data
plt.figure(figsize=(10, 6))
#plt.hist(delays, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# Plot the Gaussian (normal) distribution on top of the histogram
x = np.linspace(min(delays), max(delays), 1000)
pdf = norm.pdf(x, mean_delay, std_delay)
plt.plot(x, pdf, 'r-', label=f'Gaussian Distribution\nMean: {mean_delay:.2e}, Std: {std_delay:.2e}')

# Add labels and grid
plt.title('Gaussian Distribution Fitted to Data')
plt.xlabel('Delay (seconds)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()