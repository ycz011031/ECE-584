# statistical_timing_verification.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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
            max_violation = max(violations.values(), default=0)

            if max_violation <= max_failure_rate:
                best_period = mid
                high = mid
            else:
                low = mid

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

if __name__ == "__main__":
    tg = TimingGraph()

    # Example: use Gaussian, sample list, and a custom function
    delay_samples = [1.9, 2.1, 2.0, 2.2, 1.8]
    custom_pdf = lambda: np.random.exponential(scale=1.5)

    tg.add_path("RegA", "RegB", gate_delays=[(1.0, 0.2), delay_samples])
    tg.add_path("RegB", "RegC", gate_delays=[custom_pdf])

    clock_period = 3.0
    violations, histograms = tg.simulate(clock_period=clock_period, num_trials=10000)

    print("Probability of timing violations:")
    for path, prob in violations.items():
        print(f"Path {path}: {prob:.4f}")

    tg.visualize_delay_histograms(histograms)

    max_freq = tg.find_max_frequency(max_failure_rate=0.01, num_trials=5000)
    print(f"Maximum frequency with ≤1% failure rate: {max_freq:.3f} GHz")

    print("\nSlack values at 3.0 ns clock period:")
    slacks = tg.compute_edge_slacks(clock_period=clock_period)
    for path, slack in slacks.items():
        print(f"Path {path}: {slack:.3f} ns slack")
