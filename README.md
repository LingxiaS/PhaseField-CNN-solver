# Phase-Field AI Surrogate Solver

This repository implements a Convolutional Neural Network (U-Net) as an autoregressive surrogate solver for the Allen-Cahn equation. It demonstrates how Deep Learning can be used in the prediction of stiff nonlinear partial differential equations (PDEs) employed in materials science to model microstructural evolution.

### The Governing Physics

The model learns and predicts the discrete time evolution of the Allen-Cahn equation. The phase-field order parameter $u$ evolves according to Allen-Cahn dynamics:

$$ \frac{\partial u}{\partial t} = L \left( \epsilon^2 \nabla^2 u - W(u^3 - u) \right) $$

Where:
* $u$: The non-conserved phase-field order parameter.
* $L$: Kinetic mobility coefficient.
* $\epsilon$: Gradient energy coefficient (controls interfacial energy and thickness).
* $W$: Double-well barrier height.
* $\nabla^2 u$: The spatial Laplacian, computed via a 2D 5-point stencil.

![Phase-Field Demo](demo.gif)

## Project Architecture
The project is modularized for clean deployment:
* `core/fdm_solver.py`: A multi-core Finite Difference Method solver for ground truth Allen-Cahn dynamics.
* `core/model.py`: The U-Net PyTorch architecture designed for phase-field evolution.
* `train.py`: The data generation and training pipeline.
* `evaluate.py`: The inference script for autoregressive prediction on unseen random seeds.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/LingxiaS/PhaseField-CNN-solver
cd PhaseField-CNN-solver
pip install -r requirements.txt