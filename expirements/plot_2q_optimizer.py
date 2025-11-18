# experiments/plot_2q_optimizers.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from src.optimize_2q import run_optimization, run_spsa_optimization

def main():
    print("Running Nelder-Mead optimization...")
    # Nelder-Mead run
    nm_out = run_optimization(
        maxiter=80,
        verbose=False
    )
    nm_hist = nm_out["history"]

    print("Running SPSA optimization...")
    # SPSA run
    spsa_out = run_spsa_optimization(
        maxiter=50,
        verbose=False
    )
    spsa_hist = spsa_out["history"]

    # Plot
    plt.figure()
    plt.plot(nm_hist, label="Nelder-Mead", marker="o")
    plt.plot(spsa_hist, label="SPSA", marker="x")
    plt.xlabel("Iteration")
    plt.ylabel("Entanglement entropy")
    plt.title("2-qubit entanglement optimization (simulator)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("expirements/2q_optimizers_comparison.png", dpi=200)
    print("Saved plot to expirements/2q_optimizers_comparison.png")

    print("Nelder-Mead final entanglement:", nm_out["optimal_entanglement"])
    print("SPSA final entanglement:", spsa_out["optimal_entanglement"])

if __name__ == "__main__":
    main()
