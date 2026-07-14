import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from core.fdm_solver import evolve_allen_cahn
from core.model import PhaseFieldUNet

def save_single_image(data, title, filepath):
    """Helper function to save a single 2D array as an image."""
    plt.figure(figsize=(4, 4))
    plt.imshow(data, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase-Field U-Net and generate sequential images")
    parser.add_argument("--model_path", type=str, default="checkpoints/unet_model.pth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for unseen test data")
    parser.add_argument("--out_dir", type=str, default="results")
    
    # New customizable parameters
    parser.add_argument("--input_time", type=int, default=300, help="FDM steps before ML starts predicting")
    parser.add_argument("--end_time", type=int, default=500, help="Total FDM steps to simulate")
    parser.add_argument("--interval", type=int, default=100, help="Steps between each saved frame")
    
    args = parser.parse_args()

    # Physics constraint warning
    if args.interval != 100:
        print(f"\n[WARNING] The U-Net was trained specifically to predict a delta_t of 100 steps.")
        print(f"You set interval to {args.interval}. The ML model will still predict, but its output ")
        print(f"represents a 100-step jump, which will desync with the FDM interval!\n")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load Model
    model = PhaseFieldUNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded pre-trained model from {args.model_path}")

    # ==========================================
    # Phase 1: Pure FDM (0 -> input_time)
    # ==========================================
    print(f"\n--- Phase 1: Evolving FDM from t=0 to t={args.input_time} ---")
    np.random.seed(args.seed)
    u_current = 0.1 * (np.random.rand(200, 200) * 2.0 - 1.0)
    
    # Save initial state
    save_single_image(u_current, "FDM Input (t=0)", os.path.join(args.out_dir, "t_0000_fdm.png"))
    
    current_step = 0
    while current_step < args.input_time:
        step_size = min(args.interval, args.input_time - current_step)
        u_current = evolve_allen_cahn(u_current, steps=step_size)
        current_step += step_size
        
        filename = f"t_{current_step:04d}_fdm.png"
        save_single_image(u_current, f"FDM (t={current_step})", os.path.join(args.out_dir, filename))
        print(f"Saved: {filename}")

    # ==========================================
    # Phase 2: FDM vs U-Net (input_time -> end_time)
    # ==========================================
    print(f"\n--- Phase 2: Autoregressive ML vs FDM from t={args.input_time} to t={args.end_time} ---")
    
    u_fdm_gt = u_current.copy()
    u_ml_pred = u_current.copy()
    
    while current_step < args.end_time:
        step_size = min(args.interval, args.end_time - current_step)
        
        # Advance Ground Truth FDM
        u_fdm_gt = evolve_allen_cahn(u_fdm_gt, steps=step_size)
        
        # Advance U-Net Prediction
        with torch.no_grad():
            input_tensor = torch.tensor(u_ml_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred_tensor = model(input_tensor)
            u_ml_pred = pred_tensor[0, 0].cpu().numpy()
            
        current_step += step_size
        
        # Save FDM Ground Truth
        fdm_filename = f"t_{current_step:04d}_fdm_gt.png"
        save_single_image(u_fdm_gt, f"FDM GT (t={current_step})", os.path.join(args.out_dir, fdm_filename))
        
        # Save ML Prediction
        ml_filename = f"t_{current_step:04d}_unet_pred.png"
        save_single_image(u_ml_pred, f"U-Net Pred (t={current_step})", os.path.join(args.out_dir, ml_filename))
        
        print(f"Saved: {fdm_filename} & {ml_filename}")

    print(f"\nSuccess! All images have been dumped into the '{args.out_dir}' folder.")