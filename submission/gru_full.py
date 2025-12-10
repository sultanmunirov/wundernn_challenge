import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import r2_score
from collections import deque

from feature_hub import FeatureEngineer, OnlineFeatureMaker, BASE_OUT_DIM, TANH_SCALE
from torch.utils.data import Dataset, DataLoader
from utils import DataPoint


def set_seed():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    np.random.seed(63)
    torch.manual_seed(63)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


set_seed()


class AsymmetricMSELoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):
        errors = pred - target
        weights = torch.where(errors > 0, self.pos_weight, self.neg_weight)
        return torch.mean(weights * errors**2)


class HighwayHead(nn.Module):
    def __init__(self, d_in=256, d_mid=128, d_out=BASE_OUT_DIM, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.h = nn.Linear(d_in, d_mid)
        self.t = nn.Linear(d_in, d_mid)
        self.proj_x = nn.Linear(d_in, d_mid, bias=False)
        self.proj_h = nn.Linear(d_mid, d_out)
        self.drop = nn.Dropout(p)
        nn.init.constant_(self.t.bias, 0)

    def forward(self, x):
        x = self.ln(x)
        H = F.gelu(self.h(x))
        T = torch.sigmoid(self.t(x))
        Xs = self.proj_x(x)
        y = T * H + (1.0 - T) * Xs
        return self.proj_h(self.drop(y))


class _GRUFullPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_after_gru = nn.LayerNorm(hidden_dim)
        self.head = HighwayHead(hidden_dim, 512, BASE_OUT_DIM, p=dropout)

        self._apply_chrono_init(hidden_dim, T_max=1000)

    def _apply_chrono_init(self, hidden_size, T_max):
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "bias_hh" in name:
                    T = torch.logspace(
                        0, np.log10(T_max), hidden_size, base=10.0, dtype=torch.float64
                    )
                    idx = torch.randperm(hidden_size)
                    T = T[idx]
                    init_bias = torch.log(T - 1 + 1e-6)
                    param[hidden_size : 2 * hidden_size].copy_(init_bias)
                elif "bias_ih" in name:
                    param[hidden_size : 2 * hidden_size].fill_(0.0)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        out = self.dropout(out)
        out = self.layer_norm_after_gru(out)
        preds = self.head(out)
        return preds, hidden

    def step(self, x_t, hidden=None):
        x_t = x_t.unsqueeze(1)
        out, hidden = self.gru(x_t, hidden)
        last = out[:, -1, :]
        last = self.dropout(last)
        last = self.layer_norm_after_gru(last)
        pred = self.head(last)
        return pred, hidden


class LastNModelQueue:
    def __init__(self, k=5):
        self.k = k
        self.models = []

    def update(self, score, state_dict):
        sd_copy = {key: val.cpu().clone() for key, val in state_dict.items()}
        self.models.append((score, sd_copy))
        if len(self.models) > self.k:
            self.models.pop(0)

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


class GRUFullModel:
    def __init__(
        self,
        feature_names=("id", "sigmoid", "silu", ),
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        num_epochs=20,
        batch_size=64,
        tanh_scale=TANH_SCALE,
        num_boosts=2,
        max_epochs_per_boost=[200, 160],
        learning_rates=[1e-3, 1e-2],
        weight_decays=[1e-1, 1e-1],
        pos_weight=1.0,
        neg_weight=1.0,
    ):
        set_seed()
        self.feature_names = feature_names
        self.fe = FeatureEngineer(self.feature_names, tanh_scale)
        self.fm = OnlineFeatureMaker(self.feature_names, tanh_scale)
        self.input_dim = self.fe.in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.max_epochs = (
            max_epochs_per_boost if max_epochs_per_boost else [num_epochs] * num_boosts
        )
        self.batch_size = batch_size
        self.num_boosts = num_boosts
        self.learning_rates = learning_rates
        self.weight_decays = weight_decays
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [
            _GRUFullPredictor(self.input_dim, hidden_dim, num_layers, dropout).to(
                self.device
            )
            for _ in range(self.num_boosts)
        ]
        for model in self.models:
            model.gru.flatten_parameters()

        lr_str = "_".join([f"{lr:.0e}" for lr in self.learning_rates])
        wd_str = "_".join([f"{wd:.0e}" for wd in self.weight_decays])
        self.model_name = f"gru_chrono_full_h{hidden_dim}_lr{lr_str}_wd{wd_str}"
        self.best_models_path = f"models/{self.model_name}.pt"
        self.per_boost_paths = [
            f"models/{self.model_name}_boost{i}.pt" for i in range(self.num_boosts)
        ]
        self.best_val_r2 = -float("inf")
        self.best_epoch = 0
        self.patience = 20
        self.best_stage_r2s = [-float("inf")] * self.num_boosts
        self.current_seq_ix = None
        self.hiddens = [None] * self.num_boosts
        self.sequence_history = deque()
        self.is_trained = False
        self.tanh_scale = tanh_scale

    def _get_mask(self, batch_size, seq_len, lengths, device=None):
        if device is None:
            device = self.device
        return torch.arange(seq_len, device=device).expand(
            batch_size, seq_len
        ) < lengths.to(device).unsqueeze(1)

    def _compute_stage_metrics(self, loader, model, prev_models, crit, device):
        val_loss = 0.0
        nbv = 0
        targets_list = []
        total_preds_list = []

        model.eval()
        with torch.inference_mode():
            for px, py, pmask, L in loader:
                if px.numel() == 0:
                    continue
                px = px.to(device, non_blocking=True)
                py = py.to(device, non_blocking=True)

                len_mask = self._get_mask(px.size(0), px.size(1), L, device=device)
                final_mask = len_mask & pmask.to(device)

                cum_prev_data = torch.zeros_like(py)
                for prev_m in prev_models:
                    p_data, _ = prev_m(px)
                    cum_prev_data += p_data

                current_p, _ = model(px)
                residual_data = py - cum_prev_data

                if final_mask.sum() > 0:
                    val_loss += crit(
                        current_p[final_mask], residual_data[final_mask]
                    ).item()

                total_pred = cum_prev_data + current_p

                targets_list.append(py[final_mask].cpu().numpy())
                total_preds_list.append(total_pred[final_mask].cpu().numpy())
                nbv += 1

        val_loss = val_loss / max(1, nbv)

        if targets_list:
            true_arr = np.concatenate(targets_list, axis=0)
            pred_arr = np.concatenate(total_preds_list, axis=0)
            if len(true_arr) > 0:
                stage_r2 = float(
                    np.mean(
                        [
                            r2_score(true_arr[:, i], pred_arr[:, i])
                            for i in range(BASE_OUT_DIM)
                        ]
                    )
                )
            else:
                stage_r2 = 0.0
        else:
            stage_r2 = 0.0

        return val_loss, stage_r2

    def _collect_targets(self, loader):
        targets_list = []
        for _, Ypad, pmask, L in loader:
            if Ypad.numel() == 0:
                continue
            mask = self._get_mask(Ypad.size(0), Ypad.size(1), L, device=Ypad.device)
            final_mask = mask & pmask
            targets_list.append(Ypad[final_mask].cpu().numpy())
        return (
            np.concatenate(targets_list, axis=0)
            if targets_list
            else np.empty((0, BASE_OUT_DIM))
        )

    def _train_model(self, train_seqs, val_seqs=None):
        set_seed()
        print(f"Training {self.model_name} with Multi-Augmentation (5x Data + Noise)...")
        df = pd.read_parquet(
            os.path.join(os.path.dirname(__file__), "../../datasets/train.parquet")
        )
        df = df.sort_values(["seq_ix", "step_in_seq"])
        seqs = df.groupby("seq_ix")

        train_df = df[df["seq_ix"].isin(train_seqs)].copy()
        seq_vars_dict = {}
        print("Calculating sequence variances...")
        for seq_id, group in train_df.groupby("seq_ix"):
            states = group.iloc[:, 3:].values
            if len(states) >= 10:
                mean_var = np.mean(np.var(states, axis=0))
                seq_vars_dict[seq_id] = mean_var

        seq_vars = np.array(list(seq_vars_dict.values()))
        median_var = np.median(seq_vars)
        print(f"Dataset Median Variance: {median_var:.6f}")
        print(f"Dataset Min Var: {np.min(seq_vars):.6f}, Max Var: {np.max(seq_vars):.6f}")

        def seq_arrays(sid):
            seq_df = seqs.get_group(sid).sort_values("step_in_seq")
            needs_pred = seq_df.iloc[:, 2].values.astype(bool)
            states = seq_df.iloc[:, 3:].values.astype(np.float32)
            feats = self.fe.featurize_seq(states)
            return states, feats, needs_pred

        train_pairs, val_pairs = [], []
        
        for sid in train_seqs:
            st, ft, npred = seq_arrays(sid)
            if len(st) >= 2:
                train_pairs.append((st, ft, npred))
        
        if val_seqs:
            for sid in val_seqs:
                st, ft, npred = seq_arrays(sid)
                if len(st) >= 2:
                    val_pairs.append((st, ft, npred))

        aug_train_pairs = []
        aug_vars_log = [] 
        
        NUM_AUGMENTATIONS = 4 
        NOISE_LEVEL = 0.05

        print(f"Generating {NUM_AUGMENTATIONS}x Augmented Data with Random Scaling + Noise...")
        
        for sid in train_seqs:
            st, _, npred = seq_arrays(sid)
            current_var = seq_vars_dict.get(sid, median_var)
            
            scale_to_median = np.sqrt(median_var / (current_var + 1e-8))
            
            for _ in range(NUM_AUGMENTATIONS):
                random_scale_jitter = np.random.uniform(0.6, 1.4)
                
                final_scale = scale_to_median * random_scale_jitter
                
                final_scale = np.clip(final_scale, 0.2, 5.0)
                
                st_scaled = st * final_scale
                
                seq_std = np.std(st_scaled)
                noise_sigma = NOISE_LEVEL * seq_std
                noise = np.random.normal(0, noise_sigma, st_scaled.shape)
                
                st_aug = (st_scaled + noise).astype(np.float32)
                
                ft_aug = self.fe.featurize_seq(st_aug)
                
                if len(st_aug) >= 2:
                    aug_train_pairs.append((st_aug, ft_aug, npred))
                    aug_vars_log.append(np.mean(np.var(st_aug, axis=0)))

        train_pairs.extend(aug_train_pairs)

        aug_vars_log = np.array(aug_vars_log)
        print("\n--- Augmentation Statistics ---")
        print(f"Original Median Var : {median_var:.6f}")
        print(f"Augmented Median Var: {np.median(aug_vars_log):.6f}")
        print(f"Augmented Min Var   : {np.min(aug_vars_log):.6f}")
        print(f"Augmented Max Var   : {np.max(aug_vars_log):.6f}")
        print(f"Total Train Samples : {len(train_pairs)} (Original: {len(train_pairs) - len(aug_train_pairs)})")
        print("-------------------------------\n")

        val_pairs_full = val_pairs

        class _SeqDS(Dataset):
            def __init__(self, pairs):
                self.pairs = pairs

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, i):
                return self.pairs[i]

        def collate_shift1(pairs):
            xs, ys, np_masks, lens = [], [], [], []
            for states, feats, needs_pred in pairs:
                xs.append(torch.tensor(feats[:-1], dtype=torch.float32))
                ys.append(torch.tensor(states[1:], dtype=torch.float32))
                np_masks.append(torch.tensor(needs_pred[1:], dtype=torch.bool))
                lens.append(len(states) - 1)
            return (
                pad_sequence(xs, batch_first=True),
                pad_sequence(ys, batch_first=True),
                pad_sequence(np_masks, batch_first=True),
                torch.tensor(lens, dtype=torch.long),
            )

        train_dataset = _SeqDS(train_pairs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_shift1,
            pin_memory=self.device.type == "cuda",
        )

        val_loader_full = (
            DataLoader(
                _SeqDS(val_pairs_full),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_shift1,
                pin_memory=self.device.type == "cuda",
            )
            if val_pairs
            else None
        )

        full_Y_full = self._collect_targets(val_loader_full)

        for boost_idx in range(self.num_boosts):
            print(f"\n--- Boost stage {boost_idx + 1}/{self.num_boosts} ---")
            for prev_idx in range(boost_idx):
                if os.path.exists(self.per_boost_paths[prev_idx]):
                    self.models[prev_idx].load_state_dict(
                        torch.load(
                            self.per_boost_paths[prev_idx], map_location=self.device
                        )
                    )
                    self.models[prev_idx].eval()

            model = self.models[boost_idx]
            optimiz = optim.AdamW(
                model.parameters(),
                lr=self.learning_rates[boost_idx],
                weight_decay=self.weight_decays[boost_idx],
            )
            crit = AsymmetricMSELoss(self.pos_weight, self.neg_weight)
            best_stage_r2 = self.best_stage_r2s[boost_idx]
            stage_patience = 0
            best_stage_state = None

            last_n_queue = LastNModelQueue(k=5)
            prev_models = self.models[:boost_idx]

            for epoch in range(self.max_epochs[boost_idx]):
                t0 = time.time()
                model.train()
                tot_loss, nb = 0.0, 0
                for px, py, pmask, L in train_loader:
                    if px.numel() == 0:
                        continue
                    px, py = px.to(self.device, non_blocking=True), py.to(
                        self.device, non_blocking=True
                    )

                    mask = self._get_mask(px.size(0), px.size(1), L, device=self.device)

                    with torch.no_grad():
                        cum_prev_data = torch.zeros_like(py)
                        for prev_m in prev_models:
                            prev_m.eval()
                            p_data, _ = prev_m(px)
                            cum_prev_data += p_data

                    residual_data = py - cum_prev_data
                    optimiz.zero_grad(set_to_none=True)
                    current_p, _ = model(px)

                    loss = crit(current_p[mask], residual_data[mask])
                    loss.backward()
                    optimiz.step()
                    tot_loss += loss.item()
                    nb += 1

                train_loss = tot_loss / max(1, nb)

                vl, stage_r2_full = 0.0, 0.0
                if val_loader_full:
                    vl, stage_r2_full = self._compute_stage_metrics(
                        val_loader_full, model, prev_models, crit, self.device
                    )

                dt = time.time() - t0
                print(
                    f"Boost {boost_idx+1} Epoch {epoch+1}: TrainLoss={train_loss:.4f} ValLoss={vl:.4f} GlobalR2_full={stage_r2_full:.4f} ({dt:.2f}s)"
                )

                raw_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

                last_n_queue.update(stage_r2_full, raw_state_dict)

                avg_state_dict = last_n_queue.get_averaged_state_dict()
                avg_r2_full = -float("inf")

                if avg_state_dict is not None:
                    model.load_state_dict(avg_state_dict)
                    _, avg_r2_full = self._compute_stage_metrics(
                        val_loader_full, model, prev_models, crit, self.device
                    )
                    print(f" [SWA-N] Avg Model Full Global R2: {avg_r2_full:.4f}")

                if avg_r2_full > stage_r2_full:
                    step_winner_r2 = avg_r2_full
                    step_winner_state = avg_state_dict
                else:
                    step_winner_r2 = stage_r2_full
                    step_winner_state = raw_state_dict

                if step_winner_r2 > best_stage_r2:
                    best_stage_r2 = step_winner_r2
                    best_stage_state = {
                        k: v.clone() for k, v in step_winner_state.items()
                    }
                    self.best_stage_r2s[boost_idx] = best_stage_r2
                    stage_patience = 0
                    torch.save(best_stage_state, self.per_boost_paths[boost_idx])
                    print(f" -> New best Global R2 full: {best_stage_r2:.4f}")
                else:
                    stage_patience += 1
                    if stage_patience >= self.patience:
                        print(f"Early stop at epoch {epoch+1} for boost {boost_idx+1}")
                        model.load_state_dict(raw_state_dict)
                        break

                model.load_state_dict(raw_state_dict)

            if best_stage_state is not None:
                model.load_state_dict(best_stage_state)
                model.eval()

            cum_r2_full = best_stage_r2
            print(f"Boost {boost_idx + 1} Cumulative Full Val R2: {cum_r2_full:.4f}")

            if cum_r2_full > self.best_val_r2:
                self.best_val_r2 = cum_r2_full
                self.best_epoch = epoch + 1
                torch.save([m.state_dict() for m in self.models], self.best_models_path)

        self.is_trained = True

    def reset_state(self):
        self.current_seq_ix = None
        self.hiddens = [None] * self.num_boosts
        self.sequence_history.clear()
        self.fm = OnlineFeatureMaker(self.feature_names, self.tanh_scale)

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            return False
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dicts" in ckpt:
            for m, sd in zip(self.models, ckpt["state_dicts"]):
                m.load_state_dict(sd)
        elif isinstance(ckpt, list):
            for m, sd in zip(self.models, ckpt):
                m.load_state_dict(sd)
        elif isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt)
            for m in self.models:
                m.load_state_dict(sd)
        else:
            raise ValueError(f"Unknown checkpoint type")

        if isinstance(ckpt, dict):
            self.best_epoch = ckpt.get("best_epoch", self.best_epoch)
            self.best_val_r2 = ckpt.get("best_val_r2", self.best_val_r2)
        for m in self.models:
            m.eval()
        self.is_trained = True
        return True

    @torch.no_grad()
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.reset_state()
            self.current_seq_ix = data_point.seq_ix

        state_arr = np.asarray(data_point.state, np.float32)
        self.sequence_history.append(state_arr)

        feat_vec = self.fm.make_single(state_arr)
        x_t = torch.tensor(feat_vec, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

        pred_tensor = torch.zeros(BASE_OUT_DIM, dtype=torch.float32, device=self.device)

        for i, model in enumerate(self.models):
            y_next, self.hiddens[i] = model.step(x_t, self.hiddens[i])
            pred_tensor += y_next.squeeze(0)

        if not data_point.need_prediction:
            return None

        return pred_tensor.cpu().numpy()

    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                "state_dicts": [m.state_dict() for m in self.models],
                "best_epoch": self.best_epoch,
                "best_val_r2": self.best_val_r2,
            },
            model_path,
        )
        print(f"Saved {self.model_name} to {model_path}")


if __name__ == "__main__":
    pass