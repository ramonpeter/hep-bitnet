# SMEFTNet Regression Task

This repository provides code to train and test **SMEFTNet** (and its quantized variant **SMEFTNet-Bit**) for a regression task. The primary goal is to estimate the decay plane angle \(\phi_{\text{decay}}^\text{jet}\) from particle-level jet information in simulated \(\PW\PZ\) events. For background details and additional context, please see the discussion in [Ref.~\cite{Chatterjee:2024pbp}](#background-information) (provided below).

## Getting Started

### Repository Structure

- **`train_regression.py`**  
  Python script to train the SMEFTNet or SMEFTNet-Bit model for the regression task.
  
- **`test_regression.py`**  
  Python script to evaluate a trained model on test data.

- **`NN/models/`**  
  Directory where model weights are saved after every training epoch.

- **`SMEFT.sbatch`**  
  Example SLURM batch script demonstrating how to run the training and testing jobs on an HPC cluster, including how to specify command-line arguments.

- **`SMEFTNet.py`**  
  The original SMEFTNet model architecture, featuring standard linear layers.

- **`SMEFTNet-Bit.py`**  
  A variant of SMEFTNet where certain (or all) linear layers are replaced by BitLinear layers. Within this file, you can specify which parts of the network should be quantized:
  - **MPNN block only** (Message Passing Neural Network component)
  - **MLP layers only** (feed-forward component)
  - **All linear layers** in the entire model

## Running the Code

1. **Train the model**  
   ```bash
   python3 train_regression.py
   ```
   This command will initialize the SMEFTNet (or SMEFTNet-Bit, depending on how you configure it) and begin training. Model weights are saved to `NN/models` after every epoch.

2. **Test the model**  
   ```bash
   python3 test_regression.py
   ```
   This command loads the saved model weights from `NN/models` and evaluates performance on the test dataset.

3. **Using the SLURM batch script**  
   An example SLURM script, `SMEFT.sbatch`, is included to show how to submit training or testing jobs to a high-performance computing cluster. You can edit the parameters and relevant variables to suit your environment.

## Notes on SMEFTNet vs. SMEFTNet-Bit

- **SMEFTNet**: The standard model architecture with floating-point linear layers.  
- **SMEFTNet-Bit**: A quantized version with BitLinear layers, offering potential memory and computational efficiency at the cost of reduced precision.

In `SMEFTNet-Bit.py`, you can choose to quantize:
- All linear layers (complete quantization),
- Only the MLP layers (about 70% of weights),
- Or just the MPNN block layers (about 30% of weights).

This flexibility allows you to experiment with different trade-offs between model size, inference speed, and performance accuracy.

---

**Contact**: For any questions or issues, please open a GitHub issue or contact the repository owner directly.
