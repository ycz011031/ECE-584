import numpy as np
import matplotlib.pyplot as plt



def plot_perturbation(lb,margin):
# Define perturbation values
    perturbations = [0, 0.001, 0.003, 0.01, 0.03]

    # Initialize lb (LP lower bound) and margin (MILP exact solution) manually
    # Each list should contain 10 lists (one for each objective)

    # Ensure all objectives (except 7) have the same length as perturbations
    for i in range(10):
        if i == 7:  # Skip ground truth objective
            continue
        if len(lb[i]) != len(perturbations) or len(margin[i]) != len(perturbations):
            raise ValueError(f"Objective {i}: lb and margin arrays must match perturbation length!")

    # Plot MILP (Exact Margin) for 9 objectives (excluding ground truth objective 7)
    plt.figure(figsize=(8, 6))
    for obj_idx in range(10):
        if obj_idx == 7:
            continue  # Skip ground truth objective
        plt.plot(perturbations, margin[obj_idx], marker='o', linestyle='-', label=f"MILP Obj {obj_idx}")

    plt.xlabel("Perturbation Size (s)")
    plt.ylabel("Verification Objective Value")
    plt.title("MILP (Exact Margin) for 9 Objectives (Excluding Objective 7)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot LP (Lower Bound) for 9 objectives (excluding ground truth objective 7)
    plt.figure(figsize=(8, 6))
    for obj_idx in range(10):
        if obj_idx == 7:
            continue  # Skip ground truth objective
        plt.plot(perturbations, lb[obj_idx], marker='s', linestyle='--', label=f"LP Obj {obj_idx}")

    plt.xlabel("Perturbation Size (s)")
    plt.ylabel("Verification Objective Value")
    plt.title("LP (Lower Bound) for 9 Objectives (Excluding Objective 7)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return