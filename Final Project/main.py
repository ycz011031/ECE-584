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