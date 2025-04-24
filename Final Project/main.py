# statistical_timing_verification.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class TimingGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_path(self, src, dst, gate_delays):
        """
        gate_delays: list of (mean, stddev) tuples for each gate on the edge
        """
        self.graph.add_edge(src, dst, gate_delays=gate_delays)

    def simulate(self, clock_period, num_trials=10000):
        violations = defaultdict(int)
        delay_histograms = defaultdict(list)

        for _ in range(num_trials):
            sampled_delays = {}

            for u, v in self.graph.edges:
                gate_delays = self.graph[u][v]['gate_delays']
                total_delay = 0
                for mu, sigma in gate_delays:
                    gate_sample = np.random.normal(mu, sigma)
                    total_delay += max(gate_sample, 0)
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

    # Example pipeline: A -> B -> C with multiple gates per path
    tg.add_path("RegA", "RegB", gate_delays=[(1.0, 0.2), (1.0, 0.1)])  # total ~2.0 ± combined
    tg.add_path("RegB", "RegC", gate_delays=[(0.8, 0.1), (0.7, 0.1)])  # total ~1.5

    clock_period = 3.0
    violations, histograms = tg.simulate(clock_period=clock_period, num_trials=10000)

    print("Probability of timing violations:")
    for path, prob in violations.items():
        print(f"Path {path}: {prob:.4f}")

    tg.visualize_delay_histograms(histograms)