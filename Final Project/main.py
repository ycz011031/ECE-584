# statistical_timing_verification.py
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import norm
import random

class TimingGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_path(self, src, dst, gate_delays):
        """
        gate_delays: list of either
            - (mean, stddev) tuple
            - list of delay samples (floats)
            - function that returns a sampled delay value
        """
        self.graph.add_edge(src, dst, gate_delays=gate_delays)

    def sample_delay(self, gate):
        if isinstance(gate, tuple) and len(gate) == 2:
            mu, sigma = gate
            return max(np.random.normal(mu, sigma), 0)
        elif isinstance(gate, list):
            return random.choice(gate)
        elif callable(gate):
            return max(gate(), 0)
        else:
            raise ValueError("Unsupported gate delay format")

    def simulate(self, clock_period, num_trials=10000):
        violations = defaultdict(int)
        delay_histograms = defaultdict(list)

        for _ in range(num_trials):
            sampled_delays = {}

            for u, v in self.graph.edges:
                gate_delays = self.graph[u][v]['gate_delays']
                total_delay = sum(self.sample_delay(gate) for gate in gate_delays)
                sampled_delays[(u, v)] = total_delay
                

            for node in nx.topological_sort(self.graph):
                arrival_time = 0
                for pred in self.graph.predecessors(node):
                    edge_delay = sampled_delays[(pred, node)]
                    arrival_time = max(arrival_time, self.graph.nodes[pred].get('arrival_time', 0) + edge_delay)
                self.graph.nodes[node]['arrival_time'] = arrival_time
                

            for u, v in self.graph.edges:
                launch = self.graph.nodes[u]['arrival_time']
                arrival = self.graph.nodes[v]['arrival_time']
                delay = arrival - launch
                delay_histograms[(u, v)].append(delay)
                if delay > clock_period:
                    violations[(u, v)] += 1
                

        total_violations = {path: count / num_trials for path, count in violations.items()}
        return total_violations, delay_histograms

    def find_max_frequency(self, max_failure_rate=0.01, num_trials=10000, tol=1e-4):
        """
        Returns the maximum frequency (1 / min_clock_period) such that
        the probability of timing violations across all paths is <= max_failure_rate.
        """
        # Initial binary search bounds (in ns)
        low = 0.1
        high = 100.0
        best_period = high

        while high - low > tol:
            mid = (low + high) / 2
            violations, _ = self.simulate(clock_period=mid, num_trials=num_trials)
            failure_rate = sum(violations.values()) / num_trials
            if failure_rate <= max_failure_rate:
                best_period = mid
                high = mid
            else:
                low = mid
            print(best_period, tol, low)

        return 1.0 / best_period  # frequency in GHz

    def compute_edge_slacks(self, clock_period):
        """
        Compute slacks for each edge based on the latest delay estimation.
        Slack = Clock Period - Edge Delay
        """
        slacks = {}
        for u, v in self.graph.edges:
            gate_delays = self.graph[u][v]['gate_delays']
            expected_delay = sum(
                np.mean(g) if isinstance(g, (list, tuple)) else g() for g in gate_delays
            )
            slacks[(u, v)] = clock_period - expected_delay
        return slacks

    def visualize_delay_histograms(self, delay_histograms):
        for (u, v), delays in delay_histograms.items():
            plt.hist(delays, bins=50, alpha=0.7)
            plt.title(f"Delay Distribution: {u} -> {v}")
            plt.xlabel("Delay (ns)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

class MonteCarloTimingGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_path(self, start_node, end_node, delays):
        """
        Add a path from start_node to end_node with a list of possible delays (samples).
        delays: list of floats
        """
        self.graph.add_edge(start_node, end_node, delays=delays)

    def load_from_file(self, filename):
        """
        Load graph from a text file.
        Each line should be: start_node end_node delay1 delay2 delay3 ...
        """
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                start_node, end_node = parts[0], parts[1]
                delays = list(map(float, parts[2:]))
                self.add_path(start_node, end_node, delays)

    def find_critical_path(self):
        """
        Find a critical path (longest expected delay using mean delay values).
        Returns: list of nodes (path)
        """
        # Use average delay for each edge
        for u, v, data in self.graph.edges(data=True):
            data['mean_delay'] = np.mean(data['delays'])
        
        # Find longest path weighted by mean delay
        path = nx.dag_longest_path(self.graph, weight='mean_delay')
        return path

    def calculate_total_delay(self, path, num_trials=10000):
        """
        For each trial, randomly pick one delay per edge in the path and sum.
        Returns: total_delays array (shape = num_trials)
        """
        total_delays = np.zeros(num_trials)

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            delays = self.graph.edges[u, v]['delays']
            samples = np.random.choice(delays, size=num_trials)
            total_delays += samples
        
        return total_delays

    def find_maximum_frequency(self, num_trials=10000):
        """
        Run full Monte Carlo to find the critical path delay distribution.
        Returns (mean, std, max_frequency)
        """
        path = self.find_critical_path()
        total_delays = self.calculate_total_delay(path, num_trials=num_trials)
        
        mean = np.mean(total_delays)
        std = np.std(total_delays)

        max_delay = np.percentile(total_delays, 99.7)  # 3-sigma worst case
        max_freq = 1.0 / max_delay if max_delay > 0 else 0.0  # GHz if delay in ns

        return mean, std, max_freq

class GaussianTimingGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_path(self, src, dst, gate_delays):
        """
        gate_delays: list of (mean, stddev) tuples only
        """
        for g in gate_delays:
            if not (isinstance(g, tuple) and len(g) == 2):
                raise ValueError("GaussianTimingGraph only supports (mean, stddev) gate delays.")
        self.graph.add_edge(src, dst, gate_delays=gate_delays)

    def compute_total_delays(self):
        """
        Computes total mean and stddev for each edge.
        """
        delay_stats = {}
        for u, v in self.graph.edges:
            gate_delays = self.graph[u][v]['gate_delays']
            means = [mu for mu, _ in gate_delays]
            stds = [sigma for _, sigma in gate_delays]
            total_mean = sum(means)
            total_std = np.sqrt(sum(sigma ** 2 for sigma in stds))
            delay_stats[(u, v)] = (total_mean, total_std)
        return delay_stats

    def visualize_pdf(self, delay_stats):
        for (u, v), (mean, std) in delay_stats.items():
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            pdf = norm.pdf(x, mean, std)

            plt.figure(figsize=(8, 5))
            plt.plot(x, pdf, label=f"N({mean:.2f}, {std:.2f}²)", color='blue')
            plt.title(f"Gaussian PDF: {u} -> {v}")
            plt.xlabel("Delay (ns)")
            plt.ylabel("Probability Density")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def find_max_frequency(self, max_failure_rate=0.01):
        """
        Finds the maximum clock frequency such that
        the probability of timing violations is ≤ max_failure_rate.
        Since the delay is Gaussian, we can solve it analytically.
        """

        # Step 1: Compute total mean and stddev for all paths
        delay_stats = self.compute_total_delays()

        # Step 2: Find the "critical" path (the one with worst-case behavior)
        critical_values = []
        for (u, v), (mean, std) in delay_stats.items():
            # Find required clock period for this edge
            # For a given failure rate, we want:
            # P(delay > clock_period) = max_failure_rate
            # => clock_period = mean + Z * std
            z = norm.ppf(1 - max_failure_rate)
            required_period = mean + z * std
            critical_values.append(required_period)

        # Step 3: The maximum clock period must satisfy *all* edges → take max
        critical_period = max(critical_values)

        # Step 4: Frequency = 1 / period
        return 1.0 / critical_period  # in GHz

def read_samples(filename):
    """
    Read a file where each line has two columns separated by tab.
    We use the second column (delay value).
    """
    samples = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                samples.append(float(parts[1]))  # Use the second column
    return samples

def compute_mean_std(samples):
    """
    Given a list of samples, return mean and std deviation.
    """
    mean = np.mean(samples)
    std = np.std(samples)
    return mean, std


def plot_monte_carlo_pdf(mc_total_delays):
    """
    Plot the normalized histogram (PDF) of Monte Carlo total delays.
    """
    plt.figure()
    plt.hist(mc_total_delays, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    plt.xlabel('Total Delay (s)')
    plt.ylabel('Probability Density')
    plt.title('Monte Carlo Delay Distribution')
    plt.grid(True)
    plt.show()

def plot_overlay(mc_total_delays, total_mean, total_std):
    """
    Plot Monte Carlo histogram + Gaussian overlay
    """
    plt.figure()
    count, bins, ignored = plt.hist(mc_total_delays, bins=100, density=True, alpha=0.7, label="Monte Carlo")
    
    # Plot Gaussian
    gaussian_pdf = (1 / (total_std * np.sqrt(2 * np.pi))) * \
                   np.exp(- (bins - total_mean)**2 / (2 * total_std**2))
    plt.plot(bins, gaussian_pdf, linewidth=2, label="Gaussian")

    plt.xlabel("Total Delay (ns)")
    plt.ylabel("Probability Density")
    plt.title("Delay Distribution Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def gaussian_test():
    # 1. Define the paths with Gaussian delays (mean, stddev) for XOR, AND, OR gates
    xor_mu, xor_sigma = 2.0, 0.1  # example values for the delays
    and_mu, and_sigma = 1.5, 0.05
    or_mu, or_sigma = 2.5, 0.2

    # Create a Gaussian Timing Graph and add paths
    gtg = GaussianTimingGraph()
    gtg.add_path("xor", "and", gate_delays=[(xor_mu, xor_sigma)])
    gtg.add_path("and", "or", gate_delays=[(and_mu, and_sigma)])
    gtg.add_path("or", "out", gate_delays=[(or_mu, or_sigma)])

    # Step 1: Compute the total delays statistics
    start_gauss = time.time()
    delay_stats = gtg.compute_total_delays()
    gaussian_runtime = time.time() - start_gauss

    # Step 2: Compute total mean and stddev for the entire path xor -> and -> or -> out
    total_mean = xor_mu + and_mu + or_mu
    total_var = xor_sigma**2 + and_sigma**2 + or_sigma**2
    total_std = np.sqrt(total_var)

    # Step 3: Plot the Gaussian delay distribution
    x = np.linspace(total_mean - 4 * total_std, total_mean + 4 * total_std, 1000)
    pdf = norm.pdf(x, total_mean, total_std)
    
    plt.figure(figsize=(8,5))
    plt.plot(x, pdf, label=f"Gaussian N({total_mean:.2f}, {total_std**2:.2f}²)", color='red', linestyle='--')
    plt.title("Gaussian PDF")
    plt.xlabel("Delay (ns)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 4: Find the max frequency
    max_freq_gauss = gtg.find_max_frequency(max_failure_rate=0.01)
    print(f"Gaussian Max Frequency: {max_freq_gauss:.3f} GHz")
    print(f"Gaussian Runtime: {gaussian_runtime:.6f} seconds")

if __name__ == "__main__":
    gaussian_test()

    # 1. Read the Monte Carlo data
    xor_samples = read_samples("xor.txt")
    and_samples = read_samples("and.txt")
    or_samples = read_samples("or.txt")

    ###### --- Monte Carlo-based --- ######
    print("\n--- Monte Carlo TimingGraph ---")
    tg = MonteCarloTimingGraph()
    tg.add_path("xor", "and", xor_samples)
    tg.add_path("and", "or", and_samples)
    tg.add_path("or", "out", or_samples)  # dummy output node

    start_mc = time.time()
    mean_mc, std_mc, max_freq_mc = tg.find_maximum_frequency(num_trials=10000)
    mc_runtime = time.time() - start_mc

    # Get all total delays for plotting
    critical_path = tg.find_critical_path()
    mc_total_delays = tg.calculate_total_delay(critical_path, num_trials=10000)

    plot_monte_carlo_pdf(mc_total_delays)

    print(f"Monte Carlo Mean Delay: {mean_mc:.3f} ns")
    print(f"Monte Carlo Std Delay: {std_mc:.3f} ns")
    print(f"Monte Carlo Max Frequency: {max_freq_mc:.3f} GHz")
    print(f"Monte Carlo Runtime: {mc_runtime:.3f} seconds")

    ###### --- Gaussian-based --- ######
    print("\n--- Gaussian TimingGraph ---")
    xor_mu, xor_sigma = compute_mean_std(xor_samples)
    and_mu, and_sigma = compute_mean_std(and_samples)
    or_mu, or_sigma = compute_mean_std(or_samples)

    gtg = GaussianTimingGraph()
    gtg.add_path("xor", "and", gate_delays=[(xor_mu, xor_sigma)])
    gtg.add_path("and", "or", gate_delays=[(and_mu, and_sigma)])
    gtg.add_path("or", "out", gate_delays=[(or_mu, or_sigma)])

    start_gauss = time.time()
    delay_stats = gtg.compute_total_delays()
    gaussian_runtime = time.time() - start_gauss

    # Sum total mean and variance for entire path xor -> and -> or -> out
    total_mean = xor_mu + and_mu + or_mu
    total_var = xor_sigma**2 + and_sigma**2 + or_sigma**2
    total_std = np.sqrt(total_var)

    plot_overlay(mc_total_delays, total_mean, total_std)

    max_freq_gauss = gtg.find_max_frequency(max_failure_rate=0.01)
    print(f"Gaussian Max Frequency: {max_freq_gauss:.3f} GHz")
    print(f"Gaussian Runtime: {gaussian_runtime:.6f} seconds")