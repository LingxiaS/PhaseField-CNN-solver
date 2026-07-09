import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import concurrent.futures

# ==========================================
# 1. Data Generation (Multi-Core FDM)
# ==========================================
def compute_laplacian(u, dx=1.0):
    return (np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) + 
            np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 4.0 * u) / (dx ** 2)

def _generate_train_sample(seed):
    """Generates a single training sample mapping t=300 to t=400."""
    np.random.seed(seed)
    Lx, Ly = 200, 200
    dx, dt, epsilon, W, L = 1.0, 0.05, 1.0, 1.0, 1.0
    
    u = 0.1 * (np.random.rand(Lx, Ly) * 2.0 - 1.0)
    
    # Evolve from t=0 to t=300 (Input state)
    for _ in range(300):
        u = u + L * dt * ((epsilon ** 2) * compute_laplacian(u, dx) - W * ((u ** 3) - u))
    frame_300 = u.copy()
    
    # Evolve from t=300 to t=400 (Target state)
    for _ in range(100):
        u = u + L * dt * ((epsilon ** 2) * compute_laplacian(u, dx) - W * ((u ** 3) - u))
    frame_400 = u.copy()
    
    return frame_300, frame_400

def generate_training_data(num_samples=50):
    num_cores = os.cpu_count()
    print(f"Generating {num_samples} training samples using {num_cores} cores...")
    
    X_data = np.zeros((num_samples, 1, 200, 200), dtype=np.float32)
    Y_data = np.zeros((num_samples, 1, 200, 200), dtype=np.float32)
    
    # Use seeds 100 to 149 to keep seed 42 strictly for testing
    seeds = [100 + i for i in range(num_samples)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(_generate_train_sample, seeds))
        
    for i, (f300, f400) in enumerate(results):
        X_data[i, 0, :, :] = f300
        Y_data[i, 0, :, :] = f400
            
    print("Training data generation complete!")
    return torch.tensor(X_data), torch.tensor(Y_data)

# ==========================================
# 2. U-Net Architecture (1-Channel Input)
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

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input channel is 1 (t=300) to predict 1 channel (t=400)
        self.inc = DoubleConv(1, 16)
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
# 3. Main Script
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # --- Part A: Train the Model ---
    X_train, Y_train = generate_training_data(num_samples=50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training U-Net on device: {device}...")
    
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    
    epochs = 500
    batch_size = 10
    
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
            
    # --- Part B: Test on Unseen Seed (42) ---
    print("\nStarting inference test on UNSEEN seed (42)...")
    
    # 1. Run FDM to get Ground Truth for seed 42
    np.random.seed(42)
    u_test = 0.1 * (np.random.rand(200, 200) * 2.0 - 1.0)
    dx, dt, epsilon, W, L = 1.0, 0.05, 1.0, 1.0, 1.0
    
    for _ in range(300):
        u_test = u_test + L * dt * ((epsilon ** 2) * compute_laplacian(u_test, dx) - W * ((u_test ** 3) - u_test))
    test_t300_gt = u_test.copy()
    
    for _ in range(100):
        u_test = u_test + L * dt * ((epsilon ** 2) * compute_laplacian(u_test, dx) - W * ((u_test ** 3) - u_test))
    test_t400_gt = u_test.copy()
    
    for _ in range(100):
        u_test = u_test + L * dt * ((epsilon ** 2) * compute_laplacian(u_test, dx) - W * ((u_test ** 3) - u_test))
    test_t500_gt = u_test.copy()

    # 2. Use trained U-Net to predict
    model.eval()
    with torch.no_grad():
        # Prepare input tensor: shape (1, 1, 200, 200)
        input_tensor = torch.tensor(test_t300_gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict t=400
        pred_t400 = model(input_tensor)
        
        # Autoregressive: use predicted t=400 to predict t=500
        pred_t500 = model(pred_t400)
        
        pred_t400_np = pred_t400[0, 0].cpu().numpy()
        pred_t500_np = pred_t500[0, 0].cpu().numpy()

    # --- Part C: Visualization ---
    print("Saving comparison plot...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    axes[0].imshow(test_t300_gt, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title("Input (t=300)\nSeed: 42")
    axes[0].axis('off')
    
    axes[1].imshow(test_t400_gt, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title("FDM Ground Truth (t=400)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_t400_np, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title("U-Net Prediction (t=400)")
    axes[2].axis('off')
    
    axes[3].imshow(test_t500_gt, cmap='coolwarm', vmin=-1, vmax=1)
    axes[3].set_title("FDM Ground Truth (t=500)")
    axes[3].axis('off')
    
    axes[4].imshow(pred_t500_np, cmap='coolwarm', vmin=-1, vmax=1)
    axes[4].set_title("U-Net Prediction (t=500)")
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig("allen_cahn_unet_prediction_compare.png", dpi=150)
    print("Done!.")
