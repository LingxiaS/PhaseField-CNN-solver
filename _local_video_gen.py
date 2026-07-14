import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec
from core.fdm_solver import evolve_allen_cahn
from core.model import PhaseFieldUNet

# ================= Configuration =================
MODEL_PATH = "checkpoints/unet_model.pth"
SEED = 42
OUTPUT_GIF = "demo.gif"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print("Generating data for video (this may take a minute)...")

# 1. Generate FDM Frames (Save every 5 steps for smooth animation)
np.random.seed(SEED)
u_current = 0.1 * (np.random.rand(200, 200) * 2.0 - 1.0)

frames_phase1 = [] # t=0 to t=300 (Input evolution)
for _ in range(60): # 60 * 5 = 300
    u_current = evolve_allen_cahn(u_current, steps=5)
    frames_phase1.append(u_current)

frames_phase2_gt = [] # t=300 to t=500
for _ in range(40): # 40 * 5 = 200
    u_current = evolve_allen_cahn(u_current, steps=5)
    frames_phase2_gt.append(u_current)

# 2. Get U-Net Predictions (Keyframes)
model = PhaseFieldUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

t300 = frames_phase1[-1]
with torch.no_grad():
    in_tensor = torch.tensor(t300, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    pred_400 = model(in_tensor)[0, 0].cpu().numpy()
    pred_500 = model(torch.tensor(pred_400).unsqueeze(0).unsqueeze(0).to(DEVICE))[0, 0].cpu().numpy()

# 3. Create smooth cross-fade for U-Net to match the 40 frames of Phase 2
frames_phase2_ml = []
for i in range(40):
    if i < 20: # t=300 to t=400 (Interpolate t300 -> pred_400)
        alpha = i / 19.0
        frame = (1 - alpha) * t300 + alpha * pred_400
    else:      # t=400 to t=500 (Interpolate pred_400 -> pred_500)
        alpha = (i - 20) / 19.0
        frame = (1 - alpha) * pred_400 + alpha * pred_500
    frames_phase2_ml.append(frame)

# 4. Setup Fancy Matplotlib Layout
fig = plt.figure(figsize=(12, 6), facecolor='black')
gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])

ax_left = fig.add_subplot(gs[:, 0])
ax_top = fig.add_subplot(gs[0, 1])
ax_bot = fig.add_subplot(gs[1, 1])

for ax in [ax_left, ax_top, ax_bot]:
    ax.axis('off')
    ax.set_facecolor('black')

fig.suptitle("Phase-Field Evolution: FDM vs U-Net", color='white', fontsize=16)
text_left = ax_left.set_title("Input State Evolution (FDM)", color='white')
text_top = ax_top.set_title("Ground Truth (FDM)", color='white')
text_bot = ax_bot.set_title("Autoregressive Prediction (U-Net)", color='white')

img_left = ax_left.imshow(frames_phase1[0], cmap='coolwarm', vmin=-1, vmax=1)
img_top = ax_top.imshow(np.zeros((200,200)), cmap='coolwarm', vmin=-1, vmax=1)
img_bot = ax_bot.imshow(np.zeros((200,200)), cmap='coolwarm', vmin=-1, vmax=1)

def update(frame_idx):
    if frame_idx < 60:
        # Phase 1: Only left video plays
        img_left.set_data(frames_phase1[frame_idx])
        ax_top.set_visible(False)
        ax_bot.set_visible(False)
        text_left.set_text(f"Input Phase (t={frame_idx*5})")
    else:
        # Phase 2: Left is frozen, right side starts playing
        idx_p2 = frame_idx - 60
        img_left.set_data(frames_phase1[-1])
        text_left.set_text("Input State (t=300) - Frozen")
        
        ax_top.set_visible(True)
        ax_bot.set_visible(True)
        
        img_top.set_data(frames_phase2_gt[idx_p2])
        img_bot.set_data(frames_phase2_ml[idx_p2])
        
        t_val = 300 + (idx_p2 + 1) * 5
        text_top.set_text(f"Ground Truth FDM (t={t_val})")
        text_bot.set_text(f"U-Net Prediction (t={t_val})")
        
    return img_left, img_top, img_bot, text_left, text_top, text_bot

print("Rendering video...")
total_frames = 60 + 40 # 100 frames total
ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)

# Save as GIF
ani.save(OUTPUT_GIF, writer=PillowWriter(fps=15))
print(f"Saved fancy demo to {OUTPUT_GIF}")
