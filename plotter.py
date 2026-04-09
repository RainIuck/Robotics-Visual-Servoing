import numpy as np
import matplotlib.pyplot as plt


class ErrorPlotter:
    def __init__(self):
        self.error_norms = []

    def update(self, e):
        """Record current pixel error vector e = s - s_star."""
        self.error_norms.append(float(np.linalg.norm(e)))

    def plot(self, save_path="error_convergence.png"):
        """Plot and save the error convergence curve."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.error_norms, linewidth=2, color="steelblue")
        ax.axhline(y=5.0, color="red", linestyle="--", linewidth=1, label="threshold (5 px)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel error norm ||e|| (px)")
        ax.set_title("IBVS Feature Error Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"[Plotter] Saved to {save_path}")
