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

## Performance & Accuracy Benchmark

To evaluate the efficiency and physical fidelity of the U-Net surrogate model against the traditional explicit Finite Difference Method (FDM), a benchmark test was conducted on an Apple Silicon MacBook Pro. 

The evaluation measures both computation speed (inference vs. numerical iteration) and prediction accuracy of the trained model (taking FDM as the ground truth).

### Quantitative Comparison

| Evaluation Stage | Equivalent FDM Steps | FDM Time | U-Net Time | Speedup Factor | Mean Squared Error (MSE) | Relative $L_2$ Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Step**<br>($t = 300 \rightarrow 400$) | 100 steps | 48.61 ms | 9.05 ms | 5.37x Faster | $1.65 \times 10^{-4}$ | 1.42% |
| **Autoregressive 2-Step**<br>($t = 300 \rightarrow 500$) | 200 steps | 98.59 ms | 18.04 ms | 5.46x Faster | $3.31 \times 10^{-4}$ | 1.99% |

### Key Takeaways

* **Significant Acceleration**: The trained surrogate model provides a consistent **~5.5x speedup** over the numerical FDM solver by skipping 100 explicit time steps in a single forward pass.
* **High Physical Fidelity**: The surrogate model captures complex microstructural evolution with minimal loss of accuracy, maintaining a **Relative $L_2$ Error below 2%** even during autoregressive step.
* **Stable Rollouts**: Error accumulation across multiple autoregressive steps remains remarkably low, demonstrating the model's robustness in handling non-linear phase-field dynamics over long time horizons.


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