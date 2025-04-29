import matplotlib.pyplot as plt
import numpy as np
# Path to your data file
data_file = 'data.txt'  # change this to your filename if needed

# Read and extract data
def plot_raw():
    x = []
    y = []

    with open(data_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip() == "":
                continue
            parts = line.split()
            if len(parts) == 2:
                x_val, y_val = parts
                x.append(int(x_val))
                y.append(float(y_val))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title('Data Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Read delay data only (ignore index)
def plot_pdf():
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

    # Plot PDF
    plt.figure(figsize=(10, 6))
    plt.hist(delays, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Monte Carlo Delay PDF')
    plt.xlabel('Delay (seconds)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    plot_raw()
    plot_pdf()