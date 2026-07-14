# Phase-Field AI Surrogate Solver

This repository implements a Convolutional Neural Network (U-Net) as an autoregressive surrogate solver for the Allen-Cahn equation. It demonstrates how Deep Learning can dramatically accelerate the prediction of stiff nonlinear partial differential equations (PDEs) used in materials science to model microstructural evolution.

*(Place your generated `demo.gif` here)*
![Phase-Field Demo](demo.gif)

## Project Architecture
The project is strictly modularized for clean research and deployment:
* `core/fdm_solver.py`: A highly optimized multi-core Finite Difference Method solver acting as the physics engine for ground truth data generation.
* `core/model.py`: The U-Net PyTorch architecture designed for pixel-to-pixel phase-field mapping.
* `train.py`: The data generation and training pipeline.
* `evaluate.py`: The inference script for autoregressive prediction on unseen random seeds.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YourUsername/PhaseField-UNet.git](https://github.com/YourUsername/PhaseField-UNet.git)
cd PhaseField-UNet
pip install -r requirements.txt