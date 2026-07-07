import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import concurrent.futures # <-- Added for multi-core processing

# ==========================================
# 1. Data Generation (Multi-Core FDM)
# ==========================================
def compute_laplacian(u, dx=1.0):
    return (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + 
            np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4.0 * u) / (dx ** 2)

def _generate_single_sample(seed):
    """Helper function to generate one sample so we can parallelize it."""
    np.random.seed(seed)
    Lx, Ly = 200, 200
    dx, dt, epsilon, W, L = 1.0, 0.05, 1.0, 1.0, 1.0
    
    u = 0.1 * (np.random.rand(Lx, Ly) * 2.0 - 1.0)
    
    # Frame 1 (t=300)
    for _ in range(300):
        u = u + L * dt * ((epsilon ** 2) * compute_laplacian(u, dx) - W * ((u ** 3) - u))
    frame1 = u.copy()
    
    # Frame 2 (t=400)
    for _ in range(100):
        u = u + L * dt * ((epsilon ** 2) * compute_laplacian(u, dx) - W * ((u ** 3) - u))
    frame2 = u.copy()
    
    # Target (t=500)
    for _ in range(100):
        u = u + L * dt * ((epsilon ** 2) * compute_laplacian(u, dx) - W * ((u ** 3) - u))
    target = u.copy()
    
    return frame1, frame2, target

def generate_multi_frame_data(num_samples=50):
    # Detect how many CPU cores you have
    num_cores = os.cpu_count()
    print(f"Generating {num_samples} samples using {num_cores} CPU cores in parallel...")
    
    X_data = np.zeros((num_samples, 2, 200, 200), dtype=np.float32)
    Y_data = np.zeros((num_samples, 1, 200, 200), dtype=np.float32)
    
    # Prepare unique seeds for each sample
    seeds = [42 + i for i in range(num_samples)]
    
    # Launch parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # map() runs the helper function on all seeds simultaneously across your cores
        results = list(executor.map(_generate_single_sample, seeds))
        
    # Unpack the parallel results back into our tensors
    for i, (f1, f2, tgt) in enumerate(results):
        X_data[i, 0, :, :] = f1
        X_data[i, 1, :, :] = f2
        Y_data[i, 0, :, :] = tgt
            
    print("Data generation complete!")
    return torch.tensor(X_data), torch.tensor(Y_data)
