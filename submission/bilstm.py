import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from datetime import datetime
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import deque
from feature_hub import (
    FeatureEngineer,
    BASE_OUT_DIM,
    TANH_SCALE,
    WindowDataset,
)
from utils import DataPoint
from sklearn.model_selection import KFold

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)


class TopKModelQueue:
    def __init__(self, k=5):
        self.k = k
        self.models = []

    def update(self, score, state_dict):
        sd_copy = {key: val.cpu().clone() for key, val in state_dict.items()}
        self.models.append((score, sd_copy))
        self.models.sort(key=lambda x: x[0], reverse=True)
        if len(self.models) > self.k:
            self.models = self.models[: self.k]

    def get_averaged_state_dict(self):
        if not self.models:
            return None
        ref_sd = self.models[0][1]
        avg_sd = {}
        n = len(self.models)
        for key, val in ref_sd.items():
            if torch.is_floating_point(val):
                sum_tensor = torch.zeros_like(val)
                for _, sd in self.models:
                    sum_tensor += sd[key]
                avg_sd[key] = sum_tensor / n
            else:
                avg_sd[key] = val
        return avg_sd


class BiLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, BASE_OUT_DIM)

    def forward(self, x, hidden=None):
        y, h = self.lstm(x, hidden)
        out = self.fc(self.dropout(y[:, -1, :]))
        return out, h


class BiLSTMModel:
    def __init__(
        self,
        feature_names=("id", "tanh", "sigmoid"),
        hidden_dim=256,
        num_layers=2,
        num_epochs=40,
        maxlen=100,
        batch_size=64,
        tanh_scale=TANH_SCALE,
    ):
        self.feature_names = tuple(feature_names)
        self.fe = FeatureEngineer(self.feature_names, tanh_scale)
        self.input_dim = self.fe.in_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.maxlen = maxlen
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMPredictor(self.input_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.model.lstm.flatten_parameters()

        self.current_seq_ix = None
        self.sequence_history = deque(maxlen=maxlen)
        self.is_trained = False
        self.best_val_r2 = -float("inf")
        self.patience = 5
        self.best_epoch = 0

        self.model_name = f"bilstm_swa_h{hidden_dim}_l{num_layers}_m{maxlen}_b{batch_size}_e{num_epochs}"
        self.best_model_path = f"models/{self.model_name}.pt"
        self.log_file = f"logs/lstm/{self.model_name}.log"

    def _evaluate(self, dataloader):
        """Helper to evaluate model on a dataloader."""
        if dataloader is None:
            return 0.0, 0.0

        self.model.eval()
        preds, targs = [], []
        criterion = nn.MSELoss()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                out, _ = self.model(x, None)
                loss = criterion(out, y)
                total_loss += loss.item()
                count += 1

                preds.append(out.cpu().numpy())
                targs.append(y.cpu().numpy())

        val_loss = total_loss / max(1, count)
        val_r2 = 0.0
        if preds:
            P = np.concatenate(preds, 0)
            Y = np.concatenate(targs, 0)
            r2s = [r2_score(Y[:, i], P[:, i]) for i in range(BASE_OUT_DIM)]
            val_r2 = float(np.mean(r2s))

        return val_loss, val_r2

    def _train_model(self, train_seqs, val_seqs):
        print(f"Training {self.model_name} on {len(train_seqs)} seqs with SWA...")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "w") as f:
            f.write(f"[Config] {self.model_name} | Start: {now}\n")
            f.write(f"Features: {self.feature_names} | input_dim={self.input_dim}\n")

        df = pd.read_parquet(f"../../datasets/train.parquet")
        seqs = df.groupby("seq_ix")

        optimizer = optim.AdamW(self.model.parameters(), lr=4e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_windows, train_targets = [], []
        for seq_id in train_seqs:
            seq_df = seqs.get_group(seq_id).sort_values("step_in_seq")
            states = seq_df.iloc[:, 3:].values.astype(np.float32)
            L = len(states)
            if L < self.maxlen + 1:
                continue
            feats_all = self.fe.featurize_seq(states)
            for t in range(self.maxlen, L):
                train_windows.append(feats_all[t - self.maxlen : t])
                train_targets.append(states[t])

        if not train_windows:
            raise ValueError("No training windows")

        train_loader = DataLoader(
            WindowDataset(train_windows, train_targets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_seqs:
            val_windows, val_targets = [], []
            for seq_id in val_seqs:
                seq_df = seqs.get_group(seq_id).sort_values("step_in_seq")
                states = seq_df.iloc[:, 3:].values.astype(np.float32)
                L = len(states)
                if L < self.maxlen + 1:
                    continue
                feats_all = self.fe.featurize_seq(states)
                for t in range(self.maxlen, L):
                    val_windows.append(feats_all[t - self.maxlen : t])
                    val_targets.append(states[t])
            if val_windows:
                val_loader = DataLoader(
                    WindowDataset(val_windows, val_targets),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

        top_k_queue = TopKModelQueue(k=3)
        patience_counter = 0

        for ep in range(self.num_epochs):
            t0 = time.time()
            self.model.train()
            tr_loss, nb = 0.0, 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                out, _ = self.model(x, None)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                nb += 1
            tr_loss /= max(1, nb)

            val_loss, val_r2 = self._evaluate(val_loader)

            raw_state_dict = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

            top_k_queue.update(val_r2, raw_state_dict)

            avg_state_dict = top_k_queue.get_averaged_state_dict()
            avg_r2 = -float("inf")

            if avg_state_dict is not None:
                self.model.load_state_dict(avg_state_dict)
                _, avg_r2 = self._evaluate(val_loader)

            if avg_r2 > val_r2:
                step_winner_r2 = avg_r2
                step_winner_state = avg_state_dict
                used_swa = True
            else:
                step_winner_r2 = val_r2
                step_winner_state = raw_state_dict
                used_swa = False

            dt = time.time() - t0

            swa_tag = "[SWA]" if used_swa else "[Raw]"
            print(
                f"[BiLSTM] Ep {ep+1}/{self.num_epochs} | Tr {tr_loss:.4f} | Val {val_loss:.4f} | "
                f"R2 {val_r2:.4f} (BestStep {step_winner_r2:.4f} {swa_tag}) | {dt:.2f}s"
            )

            with open(self.log_file, "a") as f:
                f.write(
                    f"Ep {ep+1} | Tr {tr_loss:.5f} | Val {val_loss:.5f} | RawR2 {val_r2:.5f} | SwaR2 {avg_r2:.5f} | Used {swa_tag}\n"
                )

            if step_winner_r2 > self.best_val_r2:
                self.best_val_r2 = step_winner_r2
                self.best_epoch = ep + 1
                torch.save(
                    {
                        "state_dict": step_winner_state,
                        "best_epoch": self.best_epoch,
                        "best_val_r2": self.best_val_r2,
                    },
                    self.best_model_path,
                )
                patience_counter = 0
                print(f"   >>> New Best R2: {self.best_val_r2:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[BiLSTM] Early stop at epoch {ep+1}")
                    self.model.load_state_dict(raw_state_dict)
                    break

            self.model.load_state_dict(raw_state_dict)

        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            print(
                f"Loaded best model (Epoch {checkpoint['best_epoch']}, R2 {checkpoint['best_val_r2']:.4f})"
            )

        self.is_trained = True

    def reset_state(self):
        self.current_seq_ix = None
        self.sequence_history.clear()

    @torch.no_grad()
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.reset_state()
            self.current_seq_ix = data_point.seq_ix
            self.model.lstm.flatten_parameters()
        self.sequence_history.append(np.asarray(data_point.state, np.float32))
        if not data_point.need_prediction:
            return None
        if len(self.sequence_history) < self.maxlen:
            return np.mean(np.array(list(self.sequence_history)), axis=0).astype(
                np.float32
            )

        hist = np.array(list(self.sequence_history)[-self.maxlen :], np.float32)
        feats = self.fe.featurize_seq(hist)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        out, _ = self.model(x, None)
        return out.squeeze(0).cpu().numpy()

    def load_model(self, model_path: str):
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.best_epoch = checkpoint["best_epoch"]
            self.best_val_r2 = checkpoint["best_val_r2"]
            self.model.to(self.device)
            self.model.eval()
            self.is_trained = True
            print(f"Loaded BiLSTM from {model_path} (best_epoch={self.best_epoch})")
        else:
            print(f"No model found at {model_path}")

    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "best_epoch": self.best_epoch,
            "best_val_r2": self.best_val_r2,
        }
        torch.save(checkpoint, model_path)
        print(f"Saved BiLSTM to {model_path}")


if __name__ == "__main__":
    ds_path = os.path.join(os.path.dirname(__file__), "../../datasets/train.parquet")
    assert os.path.exists(ds_path), f"Dataset not found: {ds_path}"

    df = pd.read_parquet(ds_path)
    seq_ids = sorted(df["seq_ix"].unique())

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)

    fold_results = []

    print(
        f"Starting {n_splits}-Fold Cross-Validation on {len(seq_ids)} sequences with SWA..."
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(seq_ids)):
        print(f"\n{'='*20} FOLD {fold+1}/{n_splits} {'='*20}")

        train_seqs = [seq_ids[i] for i in train_idx]
        val_seqs = [seq_ids[i] for i in val_idx]

        model = BiLSTMModel(
            feature_names=("id", "tanh", "sigmoid"),
            hidden_dim=256,
            num_layers=2,
            num_epochs=25,
            maxlen=100,
            batch_size=64,
            tanh_scale=TANH_SCALE,
        )

        model.model_name = f"{model.model_name}_fold{fold}"
        model.log_file = f"logs/lstm/{model.model_name}.log"
        model.best_model_path = f"models/{model.model_name}.pt"

        model._train_model(train_seqs, val_seqs)

        fold_results.append(model.best_val_r2)
        print(f"Fold {fold+1} Best Val R2: {model.best_val_r2:.4f}")

    print(f"\n{'='*20} CROSS-VALIDATION RESULTS {'='*20}")
    print(f"Fold R2s: {fold_results}")
    print(f"Mean R2: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
