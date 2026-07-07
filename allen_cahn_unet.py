import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Data Generation (FDM Background)
# ==========================================
def compute_laplacian(u, dx=1.0):
    return (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + 
            np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4.0 * u) / (dx ** 2)

def generate_multi_frame_data(num_samples=50):
    print(f"Generating {num_samples} samples using FDM...")
    Lx, Ly = 200, 200
    dx, dt, epsilon, W, L = 1.0, 0.05, 1.0, 1.0, 1.0
    
    # Input X: 2 channels (t=300, t=400)
    # Output Y: 1 channel (t=500)
    X_data = np.zeros((num_samples, 2, Lx, Ly), dtype=np.float32)
    Y_data = np.zeros((num_samples, 1, Lx, Ly), dtype=np.float32)
    
    for i in range(num_samples):
        np.random.seed(42 + i)
        u = 0.1 * (np.random.rand(Lx, Ly) * 2.0 - 1.0)
        
        # Burn-in to t=300 (Frame 1)
        for _ in range(300):
            laplacian = compute_laplacian(u, dx)
            u = u + L * dt * ((epsilon ** 2) * laplacian - W * ((u ** 3) - u))
        X_data[i, 0, :, :] = u.copy()
        
        # Evolve to t=400 (Frame 2)
        for _ in range(100):
            laplacian = compute_laplacian(u, dx)
            u = u + L * dt * ((epsilon ** 2) * laplacian - W * ((u ** 3) - u))
        X_data[i, 1, :, :] = u.copy()
        
        # Target t=500 (Target Frame)
        for _ in range(100):
            laplacian = compute_laplacian(u, dx)
            u = u + L * dt * ((epsilon ** 2) * laplacian - W * ((u ** 3) - u))
        Y_data[i, 0, :, :] = u.copy()
        
        if (i+1) % 10 == 0:
            print(f"  -> Generated {i+1}/{num_samples} samples")
            
    return torch.tensor(X_data), torch.tensor(Y_data)

# ==========================================
# 2. U-Net Architecture (2-Channel Input)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class MultiFrameUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Note: in_channels is now 2
        self.inc = DoubleConv(2, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32) 
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(32, 16) 
        
        self.outc = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1) 
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1) 
        x = self.conv2(x)
        
        return self.outc(x)

# ==========================================
# 3. Main Training Execution
# ==========================================
if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    
    # 1. Prepare Data
    # For a real run, increase num_samples to 200+
    num_total_samples = 40
    X_all, Y_all = generate_multi_frame_data(num_samples=num_total_samples)
    
    split_idx = int(num_total_samples * 0.8)
    X_train, Y_train = X_all[:split_idx], Y_all[:split_idx]
    X_test, Y_test = X_all[split_idx:], Y_all[split_idx:]
    
    # 2. Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training U-Net on device: {device}...")
    
    model = MultiFrameUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    
    # 3. Train
    epochs = 400
    batch_size = 8
    
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / (X_train.size()[0] / batch_size)
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.6f}")
            
    # ==========================================
    # 4. Autoregressive Testing & Visualization
    # ==========================================
    print("Running autoregressive inference...")
    model.eval()
    with torch.no_grad():
        # Get the first test sample
        # Shape: (1, 2, 200, 200) representing [t=300, t=400]
        current_input = X_test[0:1].to(device)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot the first frame of the input (t=300)
        axes[0].imshow(current_input[0, 0].cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title("Input (t=300)")
        axes[0].axis('off')
        
        # Plot the second frame of the input (t=400)
        axes[1].imshow(current_input[0, 1].cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title("Input (t=400)")
        axes[1].axis('off')
        
        # Predict the next two steps iteratively
        for i in range(2, 4):
            # Predict next frame (e.g., t=500)
            next_frame = model(current_input)
            
            axes[i].imshow(next_frame[0, 0].cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            axes[i].set_title(f"Prediction (t={300 + i*100})")
            axes[i].axis('off')
            
            # Autoregressive update: 
            # Old Frame 2 becomes New Frame 1
            # Predicted Frame becomes New Frame 2
            new_input = torch.cat([current_input[:, 1:2, :, :], next_frame], dim=1)
            current_input = new_input
            
    plt.tight_layout()
    plt.savefig("unet_predictions.png", dpi=150)
    print("Saved evaluation plot to unet_predictions.png")
