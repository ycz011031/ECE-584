# statistical_timing_verification.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class TimingGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_path(self, src, dst, mean_delay, stddev_delay):
        self.graph.add_edge(src, dst, delay_dist=(mean_delay, stddev_delay))

    def simulate(self, clock_period, num_trials=10000):
        violations = defaultdict(int)
        delay_histograms = defaultdict(list)

        for _ in range(num_trials):
            sampled_delays = {}

            for u, v in self.graph.edges:
                mu, sigma = self.graph[u][v]['delay_dist']
                delay_sample = np.random.normal(mu, sigma)
                sampled_delays[(u, v)] = max(delay_sample, 0)

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

    # Example pipeline: A -> B -> C
    tg.add_path("RegA", "RegB", mean_delay=2.0, stddev_delay=0.3)
    tg.add_path("RegB", "RegC", mean_delay=1.5, stddev_delay=0.2)

    clock_period = 3.0
    violations, histograms = tg.simulate(clock_period=clock_period, num_trials=10000)

    print("Probability of timing violations:")
    for path, prob in violations.items():
        print(f"Path {path}: {prob:.4f}")

    tg.visualize_delay_histograms(histograms)