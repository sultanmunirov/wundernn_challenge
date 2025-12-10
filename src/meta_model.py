import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import r2_score
import torch.nn.functional as F

from src.models.lstm import LSTMFullModel
from src.models.gru import GRUFullModel
from src.models.bilstm import BiLSTMModel
from src.data_utils import DataPoint

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)


class PredictionModel:
    def __init__(self):
        self.input_dim = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = LSTMFullModel()
        self.gru = GRUFullModel()
        self.bilstm = BiLSTMModel()

        self.weights = nn.Parameter(
            torch.ones(3, 32, device=self.device), requires_grad=True
        )

        self.adaptation_rate = 0.1
        self.seq_len = 899

        self.current_seq_ix = None
        self.adaptive_weights_np = None
        self.last_raw_preds = None
        self.is_trained = False

    def _collect_oof_preds(self, model, val_data, model_name):
        preds = []
        targets = []
        model.reset_state()
        for i in tqdm(range(len(val_data)), desc=model_name):
            row = val_data[i]
            seq_ix = int(row[0])
            step = int(row[1])
            need = bool(row[2])
            state = row[3:].astype(np.float32)

            dp = DataPoint(seq_ix, step, need, state)
            pred = model.predict(dp)

            if need and pred is not None and i + 1 < len(val_data):
                next_row = val_data[i + 1]
                if int(next_row[0]) == seq_ix and int(next_row[1]) == step + 1:
                    next_state = next_row[3:].astype(np.float32)
                    preds.append(pred)
                    targets.append(next_state)

        if len(preds) == 0:
            print(f"No preds for {model_name}")
            return None, None
        return np.array(preds), np.array(targets)

    def _optimize_weights(self, val_seqs):
        df = pd.read_parquet("datasets/train.parquet")

        val_df = df[df["seq_ix"].isin(val_seqs)].sort_values(["seq_ix", "step_in_seq"])
        val_data = val_df.values

        oof_dir = "models/oof"
        paths = {
            "lstm": f"{oof_dir}/oof_lstm.npy",
            "gru": f"{oof_dir}/oof_gru.npy",
            "bilstm": f"{oof_dir}/oof_bilstm.npy",
            "targets": f"{oof_dir}/oof_targets.npy",
        }

        preds = {}
        targets = (
            np.load(paths["targets"]) if os.path.exists(paths["targets"]) else None
        )

        if targets is not None:
            print(f"Loaded targets: {targets.shape}")

        jobs = [
            ("LSTM", "lstm", "lstm"),
            ("GRU", "gru", "gru"),
            ("BiLSTM", "bilstm", "bilstm"),
        ]

        for nice_name, attr_name, key in jobs:
            pth = paths[key]
            if os.path.exists(pth):
                arr = np.load(pth)
                print(f"Loaded {nice_name}: {arr.shape} from {pth}")
            else:
                arr, tgt = self._collect_oof_preds(
                    getattr(self, attr_name), val_data, nice_name
                )
                if arr is None:
                    raise RuntimeError(f"No OOF for {nice_name}")
                if targets is None:
                    targets = tgt
                    np.save(paths["targets"], targets)
                np.save(pth, arr)
                print(f"Saved {nice_name}: {arr.shape} -> {pth}")
            preds[key] = arr

        preds_lstm = torch.from_numpy(preds["lstm"]).float().to(self.device)
        preds_gf = torch.from_numpy(preds["gru"]).float().to(self.device)
        preds_bi = torch.from_numpy(preds["bilstm"]).float().to(self.device)
        targets_t = torch.from_numpy(targets).float().to(self.device)

        print("Optimizing 3x32 Global Feature-Wise Weights...")
        optimizer = optim.AdamW([self.weights], lr=0.01, weight_decay=0)
        criterion = nn.MSELoss()

        for epoch in range(200):
            optimizer.zero_grad()
            w = F.softmax(self.weights, dim=0).unsqueeze(0)
            preds_stack = torch.stack([preds_lstm, preds_gf, preds_bi], dim=1)
            blended = (w * preds_stack).sum(dim=1)
            loss = criterion(blended, targets_t)
            loss.backward()
            optimizer.step()

        final_w = F.softmax(self.weights, dim=0)
        global_w_np = final_w.detach().cpu().numpy()
        print(f"Global Weights optimized shape: {global_w_np.shape}")
        self.is_trained = True

        print("\nSimulating Adaptive Inference (Feature-Wise) on OOF Data...")

        n_samples = targets.shape[0]
        n_seq = n_samples // self.seq_len

        if n_samples % self.seq_len != 0:
            print(
                f"[WARNING] Data length {n_samples} is not divisible by {self.seq_len}. Simulation might be inaccurate."
            )
            n_samples = n_seq * self.seq_len

        p_l = preds["lstm"][:n_samples].reshape(n_seq, self.seq_len, 32)
        p_g = preds["gru"][:n_samples].reshape(n_seq, self.seq_len, 32)
        p_b = preds["bilstm"][:n_samples].reshape(n_seq, self.seq_len, 32)
        t_t = targets[:n_samples].reshape(n_seq, self.seq_len, 32)

        curr_w = np.tile(global_w_np, (n_seq, 1, 1))

        simulated_preds = []

        for t in range(self.seq_len):
            m0, m1, m2 = p_l[:, t], p_g[:, t], p_b[:, t]
            truth = t_t[:, t]

            w0 = curr_w[:, 0, :]
            w1 = curr_w[:, 1, :]
            w2 = curr_w[:, 2, :]

            pred_step = w0 * m0 + w1 * m1 + w2 * m2
            simulated_preds.append(pred_step)

            e0 = (m0 - truth) ** 2
            e1 = (m1 - truth) ** 2
            e2 = (m2 - truth) ** 2

            step_errors = np.stack([e0, e1, e2], axis=1)

            update_factor = np.exp(-self.adaptation_rate * step_errors)
            curr_w *= update_factor

            row_sums = curr_w.sum(axis=1, keepdims=True)
            curr_w /= row_sums + 1e-10

        simulated_preds = np.array(simulated_preds).transpose(1, 0, 2).reshape(-1, 32)
        flat_targets = t_t.reshape(-1, 32)

        total_r2 = r2_score(flat_targets.flatten(), simulated_preds.flatten())

        print(f"OOF RESULTS (Adaptive Rate={self.adaptation_rate}, Feature-Wise=True):")
        print(f"Global R2 (flattened): {total_r2:.5f}")
        np.save("models/meta_weights.npy", self.weights.cpu().detach().numpy())

    def _reset_state(self):
        for m in (self.lstm, self.gru, self.bilstm):
            m.reset_state()
        self.current_seq_ix = None
        self.adaptive_weights_np = None
        self.last_raw_preds = None

    @torch.inference_mode()
    def predict(self, data_point) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_state()
            self.current_seq_ix = data_point.seq_ix
            self.adaptive_weights_np = F.softmax(self.weights, dim=0).cpu().numpy()

        if self.last_raw_preds is not None:
            actual_state = data_point.state
            errors = (self.last_raw_preds - actual_state) ** 2

            update_factor = np.exp(-self.adaptation_rate * errors)
            self.adaptive_weights_np *= update_factor

            self.adaptive_weights_np /= np.sum(self.adaptive_weights_np, axis=0) + 1e-10

        p_l = self.lstm.predict(data_point)
        p_g = self.gru.predict(data_point)
        p_b = self.bilstm.predict(data_point)

        if any(p is None for p in [p_l, p_g, p_b]):
            return None

        self.last_raw_preds = np.stack([p_l, p_g, p_b])

        blended = np.sum(self.adaptive_weights_np * self.last_raw_preds, axis=0)

        return blended
