import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.lstm import LSTMFullModel
from src.models.gru import GRUFullModel
from src.models.bilstm import BiLSTMModel
from src.meta_model import PredictionModel

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

if __name__ == "__main__":
    print("Loading data")
    df = pd.read_parquet("datasets/train.parquet")

    all_seqs = sorted(df["seq_ix"].unique())

    np.random.shuffle(all_seqs)
    split_80 = int(0.8 * len(all_seqs))
    train_80 = all_seqs[:split_80]
    val_20 = all_seqs[split_80:]
    print(f"80% train ({len(train_80)} seqs), 20% val ({len(val_20)} seqs)")

    temp_model = PredictionModel()
    temp_model._reset_state()

    print("=== Train/Load on 80% data ===")

    gru_80_path = "models/gru_80.pt"
    if not os.path.exists(gru_80_path):
        temp_model.gru._train_model(train_80, val_20)
        temp_model.gru.save_model(gru_80_path)
    else:
        temp_model.gru.load_model(gru_80_path)

    bilstm_80_path = "models/bilstm_80.pt"
    if not os.path.exists(bilstm_80_path):
        temp_model.bilstm._train_model(train_80, val_20)
        temp_model.bilstm.save_model(bilstm_80_path)
    else:
        temp_model.bilstm.load_model(bilstm_80_path)

    lstm_80_path = "models/lstm_80.pt"
    if not os.path.exists(lstm_80_path):
        temp_model.lstm._train_model(train_80, val_20)
        temp_model.lstm.save_model(lstm_80_path)
    else:
        temp_model.lstm.load_model(lstm_80_path)

    print("=== OOF & Meta-opt on 20% ===")
    temp_model._optimize_weights(val_20)

    print("=== Retrain on FULL data with best epochs from 80/20 ===")
    temp_model._reset_state()

    full_lstm_path = "models/lstm.pt"
    if not os.path.exists(full_lstm_path):
        full_lstm = LSTMFullModel(max_epochs_per_boost=[68, 56])
        full_lstm._train_model(all_seqs, val_20)
        full_lstm.save_model(full_lstm_path)
    else:
        temp_model.lstm.load_model(full_lstm_path)

    full_gru_path = "models/gru.pt"
    if not os.path.exists(full_gru_path):
        full_gru = GRUFullModel(max_epochs_per_boost=[64, 22])
        full_gru._train_model(all_seqs, val_20)
        full_gru.save_model(full_gru_path)
    else:
        temp_model.gru.load_model(full_gru_path)

    full_bilstm_path = "models/bilstm.pt"
    if not os.path.exists(full_bilstm_path):
        full_bilstm = BiLSTMModel(num_epochs=7)
        full_bilstm._train_model(all_seqs, val_20)
        full_bilstm.save_model(full_bilstm_path)
    else:
        temp_model.bilstm.load_model(full_bilstm_path)

    print("Training complete. Meta-model ready for inference.")
