import os
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=r".*torch\.ao\.quantization.*"
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time

from meta import PredictionModel

np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{current_dir}")
from utils import DataPoint

BASE_OUT_DIM = 32
FEATURE_NAMES = ("id", "tanh", "sigmoid")
TANH_SCALE = 3.0

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(f"{current_dir}/..")
    from utils import ScorerStepByStep

    model = PredictionModel()

    dataset_path = f"{current_dir}/../datasets/train.parquet"
    df = pd.read_parquet(dataset_path)
    seq_ids = sorted(df["seq_ix"].unique())
    np.random.seed(42)
    np.random.shuffle(seq_ids)
    sub_size = int(0.01 * len(seq_ids))
    sub_seqs = seq_ids[:sub_size]
    print(
        f"Subsample: {len(sub_seqs)} seqs ({sub_size / max(1, len(seq_ids)) * 100:.1f}% of total)"
    )

    sub_df = df[df["seq_ix"].isin(sub_seqs)].sort_values(["seq_ix", "step_in_seq"])
    sub_file = "sub_val.parquet"
    sub_df.to_parquet(sub_file, index=False)

    scorer = ScorerStepByStep(sub_file)
    results = scorer.score(model)
    print(f"\nSubsample R² results:")
    print(f"Mean R²: {results['mean_r2']:.4f}")
    for feature, r2 in list(results.items())[:5]:
        if feature != "mean_r2":
            print(f"  {feature}: {r2:.4f}")

    os.remove(sub_file)
    print("Subsample scoring complete.")
