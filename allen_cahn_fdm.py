import numpy as np
import matplotlib.pyplot as plt

def compute_laplacian(u, dx):
    """Computes the 2D Laplacian using a 5-point stencil with periodic boundaries."""
    u_up = np.roll(u, -1, axis=0)
    u_down = np.roll(u, 1, axis=0)
    u_left = np.roll(u, -1, axis=1)
    u_right = np.roll(u, 1, axis=1)
    return (u_up + u_down + u_left + u_right - 4.0 * u) / (dx ** 2)

def run_allen_cahn():
    # --- Parameters ---
    Lx, Ly = 200, 200
    dx = 1.0
    dt = 0.05        # Time step (must be small for explicit Euler stability)
    epsilon = 1.0    # Gradient energy coefficient
    W = 1.0          # Wall height
    L = 1.0          # Mobility coefficient
    total_steps = 2000
    plot_interval = 500

    # --- Initial Condition ---
    # for Allen-Cahn a mean of 0.0
    # maze-like structure before coarsening.
    average_u = 0.0
    noise_magnitude = 0.1

    # Generate random noise: u is between (average - noise) and (average + noise)
    np.random.seed(42) # a fixed seed
    u = average_u + noise_magnitude * (np.random.rand(Lx, Ly) * 2.0 - 1.0)

    # Set up plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    plot_idx = 0

    # Plot initial state
    axes[plot_idx].imshow(u, cmap='coolwarm', vmin=-1, vmax=1)
    axes[plot_idx].set_title("Step 0")
    axes[plot_idx].axis('off')
    plot_idx += 1

    # --- Time Integration Loop (Explicit Euler) ---
    for step in range(1, total_steps + 1):
        # 1. Compute Laplacian
        laplacian = compute_laplacian(u, dx)

        # 2. Compute the derivative of the double-well potential: f'(u) = u^3 - u
        dfdu = W * ((u ** 3) - u)

        # 3. Update the phase field
        # Equation: du/dt = epsilon^2 * Laplacian(u) - f'(u)
        du_dt = (epsilon ** 2) * laplacian - dfdu
        u = u + L * dt * du_dt

        # Plotting
        if step % plot_interval == 0:
            axes[plot_idx].imshow(u, cmap='coolwarm', vmin=-1, vmax=1)
            axes[plot_idx].set_title(f"Step {step}")
            axes[plot_idx].axis('off')
            plot_idx += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_allen_cahn()
