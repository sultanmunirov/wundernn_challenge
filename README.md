# 5th Place Solution

This repository contains the source code for the prize solution. The approach is based on an ensemble of Recurrent Neural Networks (LSTM, GRU, BiLSTM) trained with Chrono Initialization and an Adaptive Meta-Learner for dynamic weight blending.

## 1. Hardware & Performance

*   **Hardware:** 1x NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB VRAM)
*   **OS:** Windows 10 (WSL2 / Ubuntu)
*   **Training Time:** ~45 minutes for the full pipeline
*   **VRAM Usage:** < 4GB

## 2. Environment Setup

The solution uses a virtual environment to ensure reproducibility.

### Prerequisites
*   Python 3.10+
*   Linux or WSL2

### Installation
Run the setup script to create a virtual environment and install pinned dependencies:

```bash
bash setup.sh
```

To activate the environment manually later:
```bash
source venv/bin/activate
```

## 3. Training Pipeline

To retrain all models from scratch using raw data, use the `train.sh` script.

**Command:**
```bash
# Usage: ./train.sh <path_to_train.parquet>
bash train.sh ./path/to/your/train.parquet
```

**What happens inside:**
1.  Creates a symlink to the data in `datasets/`.
2.  Trains **GRU**, **BiLSTM**, and **LSTM** models sequentially using boosting-like stages.
3.  Optimizes Meta-Learner weights on a hold-out set (20%).
4.  Retrains models on the full dataset.
5.  Saves final weights to `models/`.

**Reproducibility:**
All random seeds (Python, Numpy, PyTorch) are fixed to `42` (and specific seeds for models) to ensure the result is deterministic. `CUBLAS_WORKSPACE_CONFIG` is set for reproducible GPU operations.

## 4. Inference & Submission

The file `solution.py` is the entry point for the competition submission. It loads the trained weights from the `models/` directory.

To verify inference locally:
```bash
# 1. Activate env
source venv/bin/activate

# 2. Run check
python submission/solution.py
```

## 5. Model Configuration

The solution consists of three diverse Recurrent Neural Networks. Key hyperparameters are hardcoded in the model classes (`src/models/`) to ensure reproducibility.

### Common Architecture Details
*   **Chrono Initialization:** Implemented for LSTM/GRU gating biases to capture long-term dependencies (`T_max=1000`, log-uniform distribution).
*   **Highway Head:** Custom output head used in LSTM and GRU models.
    *   *Structure:* LayerNorm -> Linear -> GeLU + Gating Mechanism.
    *   *Hidden Dimension:* 512.
    *   *Dropout:* 0.1.

---

### A. LSTM (Chrono Initialized)
*   **Source:** `src/models/lstm.py`
*   **Architecture:**
    *   Input Dim: 96
    *   Hidden Dim: **150**
    *   Layers: 1
    *   Head: Highway Head (512 dim, Gate Bias initialized to **-1**)
*   **Training Strategy:**
    *   **Loss:** MSE
    *   **Features:** "id", "sigmoid", "silu"
    *   **Schedule:** 2 Boost Stages [**68 epochs**, **56 epochs**]
    *   **SWA:** Last-N (k=5) Averaging
    *   **Optimizer:** AdamW (LR: 1e-3, Weight Decay: 1e-1)

### B. GRU (Chrono Initialized)
*   **Source:** `src/models/gru.py`
*   **Architecture:**
    *   Input Dim: 96
    *   Hidden Dim: **128**
    *   Layers: 1
    *   Head: Highway Head (512 dim, Gate Bias initialized to **0**)
*   **Training Strategy:**
    *   **Loss:** MSE
    *   **Features:** "id", "sigmoid", "silu"
    *   **Schedule:** 2 Boost Stages [**64 epochs**, **31 epochs**]
    *   **SWA:** Last-N (k=5) Averaging
    *   **Optimizer:** AdamW (LR: 1e-3, Weight Decay: 1e-1)

### C. BiLSTM (Bidirectional)
*   **Source:** `src/models/bilstm.py`
*   **Architecture:**
    *   Input Dim: 96
    *   Hidden Dim: **256**
    *   Layers: 2 (Bidirectional)
    *   Head: Standard Linear Head (Dropout -> Linear)
*   **Training Strategy:**
    *   **Loss:** MSE
    *   **Features:** "id", "sigmoid", "tanh"
    *   **Schedule:** **7 epochs**
    *   **SWA:** Last-N (k=3) Averaging
    *   **Optimizer:** AdamW (LR: 4e-4, Weight Decay: 1e-4)
  
### Meta-Learner (Adaptive Ensemble)
*   **File:** `src/meta_model.py`
*   **Type:** Online Adaptive Blending (Feature-Wise)
*   **Ensemble Components:** 3 Models (LSTM, GRU, BiLSTM) × 32 Features
*   **Offline Optimization (Initial Weights):**
    *   *Data:* Out-Of-Fold (OOF) predictions on 20% validation set
    *   *Optimizer:* AdamW (LR: **0.01**)
    *   *Epochs:* **200**
    *   *Constraint:* Softmax normalization across models
*   **Online Adaptation (Inference):**
    *   **Adaptation Rate ($\alpha$):** **0.1** (Controls how fast weights react to errors)
    *   *Update Logic:* Exponential decay based on squared error

## 6. Directory Structure

```text
.
├── src/                 # Source code library
│   ├── models/          # Neural network architectures
│   ├── features.py      # Feature engineering logic
│   ├── meta_model.py    # Ensemble logic
│   └── data_utils.py    # Data loading utilities
├── scripts/             # Training orchestration
│   └── train.py
├── models/              # Trained weights (artifacts)
├── train.sh             # Entry point for training
├── setup.sh             # Environment setup script
├── submission/          # Exact submission
├── requirements.txt     # Pinned dependencies
└── README.md            # Documentation
```