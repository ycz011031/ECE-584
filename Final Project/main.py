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
    return np.loadtxt(filename).tolist()

def compute_mean_std(samples):
    return np.mean(samples), np.std(samples)

def plot_monte_carlo_pdf(delays):
    plt.figure(figsize=(8,5))
    plt.hist(delays, bins=100, density=True, alpha=0.7, label="Monte Carlo", color='blue')
    plt.title("Monte Carlo Total Delay PDF")
    plt.xlabel("Delay (ns)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def plot_overlay(monte_carlo_delays, gaussian_mean, gaussian_std):
    x = np.linspace(min(monte_carlo_delays) - 1, max(monte_carlo_delays) + 1, 1000)
    mc_hist, mc_bins = np.histogram(monte_carlo_delays, bins=100, density=True)
    mc_bin_centers = 0.5 * (mc_bins[1:] + mc_bins[:-1])

    plt.figure(figsize=(8,5))
    plt.plot(mc_bin_centers, mc_hist, label="Monte Carlo", color='blue')
    plt.plot(x, (1 / (gaussian_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - gaussian_mean) / gaussian_std)**2),
             label=f"Gaussian N({gaussian_mean:.2f}, {gaussian_std:.2f}²)", color='red', linestyle='--')
    plt.title("Monte Carlo vs Gaussian PDF")
    plt.xlabel("Delay (ns)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


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
    xor_samples = read_samples("and.txt")
    and_samples = read_samples("and.txt")
    or_samples = read_samples("and.txt")

    ###### --- Monte Carlo-based --- ######
    print("\n--- Monte Carlo TimingGraph ---")
    tg = TimingGraph()
    tg.add_path("xor", "and", gate_delays=xor_samples)
    tg.add_path("and", "or", gate_delays=and_samples)
    tg.add_path("or", "out", gate_delays=or_samples)  # dummy output node

    start_mc = time.time()
    violations, histograms = tg.simulate(clock_period=10.0, num_trials=1000)
    mc_runtime = time.time() - start_mc

    # Total delay is from "xor" to "out"
    mc_total_delays = []
    for i in range(len(histograms[("xor", "and")])):
        total = (
            histograms[("xor", "and")][i]
            + histograms[("and", "or")][i]
            + histograms[("or", "out")][i]
        )
        mc_total_delays.append(total)

    plot_monte_carlo_pdf(mc_total_delays)
    print ("debug")

    max_freq_mc = tg.find_max_frequency(max_failure_rate=0.01, num_trials=1000)
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