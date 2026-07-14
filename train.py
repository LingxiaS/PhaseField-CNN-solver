import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import concurrent.futures
from core.fdm_solver import evolve_allen_cahn
from core.model import PhaseFieldUNet

def _generate_train_sample(seed):
    np.random.seed(seed)
    Lx, Ly = 200, 200
    u = 0.1 * (np.random.rand(Lx, Ly) * 2.0 - 1.0)
    
    frame_300 = evolve_allen_cahn(u, steps=300)
    frame_400 = evolve_allen_cahn(frame_300, steps=100)
    return frame_300, frame_400

def generate_data(num_samples):
    num_cores = min(os.cpu_count(), 16)
    print(f"Generating {num_samples} samples using {num_cores} cores...")
    
    X_data = np.zeros((num_samples, 1, 200, 200), dtype=np.float32)
    Y_data = np.zeros((num_samples, 1, 200, 200), dtype=np.float32)
    seeds = [100 + i for i in range(num_samples)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(_generate_train_sample, seeds))
        
    for i, (f300, f400) in enumerate(results):
        X_data[i, 0, :, :] = f300
        Y_data[i, 0, :, :] = f400
            
    return torch.tensor(X_data), torch.tensor(Y_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase-Field U-Net")
    parser.add_argument("--samples", type=int, default=50, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, Y_train = generate_data(args.samples)
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    model = PhaseFieldUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train.size()[0], args.batch_size):
            indices = permutation[i:i+args.batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / (X_train.size()[0] / args.batch_size)
            print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {avg_loss:.6f}")

    save_path = os.path.join(args.save_dir, "unet_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
