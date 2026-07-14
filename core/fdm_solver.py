import numpy as np

def compute_laplacian(u, dx=1.0):
    """Computes the 2D Laplacian using finite difference."""
    return (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + 
            np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4.0 * u) / (dx ** 2)

def evolve_allen_cahn(u, steps, dx=1.0, dt=0.05, epsilon=1.0, W=1.0, L=1.0):
    """
    Evolves the phase-field order parameter 'u'
    using the explicit Euler scheme.
    """
    u_current = u.copy()
    for _ in range(steps):
        laplacian = compute_laplacian(u_current, dx)
        dfdu = W * ((u_current ** 3) - u_current)
        u_current = u_current + L * dt * ((epsilon ** 2) * laplacian - dfdu)
    return u_current
