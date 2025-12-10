import numpy as np
from scipy.special import erf
from torch.utils.data import Dataset
from collections import deque
import torch
import warnings
import itertools
import random

BASE_OUT_DIM = 32
TANH_SCALE = 3.0
ROLLING_KS = [5, 10, 20]


def np_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def np_silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def np_sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def np_cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def np_arcsin(x: np.ndarray) -> np.ndarray:
    return np.arcsin(np.clip(x, -1.0, 1.0))


def np_log1p_abs(x: np.ndarray) -> np.ndarray:
    """Log(1 + |x|) * sign(x) — симметричный логарифм."""
    return np.log1p(np.abs(x)) * np.sign(x)


def np_softplus(x: np.ndarray) -> np.ndarray:
    """ln(1 + exp(x))"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def np_safe_div(x: np.ndarray, y: np.ndarray, eps=1e-6) -> np.ndarray:
    return x / (y + eps)


class FeatureEngineer:
    """
    Мощный генератор фич.
    Поддерживает:
    - Base: id, tanh, sigmoid, sq, gelu, silu, sin, cos, arcsin, log, exp, sign, abs, softplus
    - Diff: diff1 (x_t - x_{t-1}), diff2 (ускорение)
    - Rolling (по окнам ROLLING_KS): mean, std, max, min
    - Cross (поперек 32 фич): cross_mean, cross_std, cross_max, cross_min
    - Pairs: средние случайных пар
    """

    def __init__(
        self,
        feature_names,
        tanh_scale: float = TANH_SCALE,
        rolling_ks: list = ROLLING_KS,
        num_pair_avgs: int = 0,
    ):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        elif len(feature_names) == 0:
            warnings.warn("Empty feature_names; fallback to identity.")
            feature_names = ["id"]

        self.feature_names = tuple(feature_names)
        self.tanh_scale = float(tanh_scale)
        self.rolling_ks = sorted(set(rolling_ks))

        self.has_rolling = any(
            k in self.feature_names
            for k in ["rolling_mean", "rolling_std", "rolling_max", "rolling_min"]
        )
        self.has_cross = any(
            k in self.feature_names
            for k in ["cross_mean", "cross_std", "cross_max", "cross_min", "cross_skew"]
        )

        self.in_dim = 0

        self.pointwise_feats = [
            f
            for f in self.feature_names
            if not f.startswith("rolling_")
            and not f.startswith("cross_")
            and f != "pair_avgs"
        ]
        self.in_dim += len(self.pointwise_feats) * BASE_OUT_DIM

        if "rolling_mean" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "rolling_std" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "rolling_max" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "rolling_min" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM

        if "cross_mean" in self.feature_names:
            self.in_dim += 1
        if "cross_std" in self.feature_names:
            self.in_dim += 1
        if "cross_max" in self.feature_names:
            self.in_dim += 1
        if "cross_min" in self.feature_names:
            self.in_dim += 1

        self.num_pair_avgs = num_pair_avgs
        self.pair_indices = []
        if self.num_pair_avgs > 0:
            self.in_dim += self.num_pair_avgs
            np.random.seed(42)
            all_pairs = list(itertools.combinations(range(BASE_OUT_DIM), 2))
            if len(all_pairs) < self.num_pair_avgs:
                self.pair_indices = all_pairs
            else:
                self.pair_indices = random.sample(all_pairs, self.num_pair_avgs)

    def featurize_seq(self, states: np.ndarray) -> np.ndarray:
        states = states.astype(np.float32, copy=False)
        T, D = states.shape
        feats = []

        if "id" in self.pointwise_feats:
            feats.append(states)
        if "tanh" in self.pointwise_feats:
            feats.append(np.tanh(states / self.tanh_scale))
        if "sigmoid" in self.pointwise_feats:
            feats.append(1.0 / (1.0 + np.exp(-states / self.tanh_scale)))
        if "sq" in self.pointwise_feats:
            feats.append(states**2)
        if "sqrt" in self.pointwise_feats:
            feats.append(np.sqrt(np.abs(states)) * np.sign(states))
        if "gelu" in self.pointwise_feats:
            feats.append(np_gelu(states))
        if "silu" in self.pointwise_feats:
            feats.append(np_silu(states))
        if "sin" in self.pointwise_feats:
            feats.append(np_sin(states))
        if "cos" in self.pointwise_feats:
            feats.append(np_cos(states))
        if "arcsin" in self.pointwise_feats:
            feats.append(np_arcsin(states))
        if "log" in self.pointwise_feats:
            feats.append(np_log1p_abs(states))
        if "exp" in self.pointwise_feats:
            feats.append(np.exp(-(states**2)))
        if "sign" in self.pointwise_feats:
            feats.append(np.sign(states))
        if "abs" in self.pointwise_feats:
            feats.append(np.abs(states))
        if "softplus" in self.pointwise_feats:
            feats.append(np_softplus(states))

        if "diff1" in self.pointwise_feats:
            d1 = np.zeros_like(states)
            d1[1:] = states[1:] - states[:-1]
            feats.append(d1)

        if "diff2" in self.pointwise_feats:
            d1 = np.zeros_like(states)
            d1[1:] = states[1:] - states[:-1]
            d2 = np.zeros_like(states)
            d2[1:] = d1[1:] - d1[:-1]
            feats.append(d2)

        if self.has_cross:
            if "cross_mean" in self.feature_names:
                feats.append(np.mean(states, axis=1, keepdims=True))
            if "cross_std" in self.feature_names:
                feats.append(np.std(states, axis=1, keepdims=True))
            if "cross_max" in self.feature_names:
                feats.append(np.max(states, axis=1, keepdims=True))
            if "cross_min" in self.feature_names:
                feats.append(np.min(states, axis=1, keepdims=True))

        if self.num_pair_avgs > 0:
            pair_vals = np.zeros((T, self.num_pair_avgs), dtype=np.float32)
            for idx, (i, j) in enumerate(self.pair_indices):
                pair_vals[:, idx] = (states[:, i] + states[:, j]) * 0.5
            feats.append(pair_vals)

        if self.has_rolling:
            n_ks = len(self.rolling_ks)
            if "rolling_mean" in self.feature_names:
                feats.append(np.zeros((T, D * n_ks), dtype=np.float32))
            if "rolling_std" in self.feature_names:
                feats.append(np.zeros((T, D * n_ks), dtype=np.float32))
            if "rolling_max" in self.feature_names:
                feats.append(np.zeros((T, D * n_ks), dtype=np.float32))
            if "rolling_min" in self.feature_names:
                feats.append(np.zeros((T, D * n_ks), dtype=np.float32))

            ptr_map = {}
            curr_ptr = len(feats) - 1
            if "rolling_min" in self.feature_names:
                ptr_map["min"] = curr_ptr
                curr_ptr -= 1
            if "rolling_max" in self.feature_names:
                ptr_map["max"] = curr_ptr
                curr_ptr -= 1
            if "rolling_std" in self.feature_names:
                ptr_map["std"] = curr_ptr
                curr_ptr -= 1
            if "rolling_mean" in self.feature_names:
                ptr_map["mean"] = curr_ptr
                curr_ptr -= 1

            for i, k in enumerate(self.rolling_ks):
                for t in range(T):
                    start = max(0, t - k + 1)
                    window = states[start : t + 1]

                    start_col = i * D
                    end_col = (i + 1) * D

                    if "mean" in ptr_map:
                        feats[ptr_map["mean"]][t, start_col:end_col] = np.mean(
                            window, axis=0
                        )
                    if "std" in ptr_map:
                        feats[ptr_map["std"]][t, start_col:end_col] = np.std(
                            window, axis=0
                        )
                    if "max" in ptr_map:
                        feats[ptr_map["max"]][t, start_col:end_col] = np.max(
                            window, axis=0
                        )
                    if "min" in ptr_map:
                        feats[ptr_map["min"]][t, start_col:end_col] = np.min(
                            window, axis=0
                        )

        if not feats:
            feats = [states]

        out = np.hstack(feats).astype(np.float32)
        if out.shape[1] != self.in_dim:
            pass
        return out


class OnlineFeatureMaker:
    """Stateful feature maker for inference."""

    def __init__(
        self,
        feature_names,
        tanh_scale: float = TANH_SCALE,
        rolling_ks: list = ROLLING_KS,
        num_pair_avgs: int = 0,
    ):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        elif len(feature_names) == 0:
            warnings.warn("Empty feature_names; fallback to identity.")
            feature_names = ["id"]

        self.feature_names = tuple(feature_names)
        self.tanh_scale = float(tanh_scale)
        self.rolling_ks = sorted(set(rolling_ks))

        self.has_rolling = any(
            k in self.feature_names
            for k in ["rolling_mean", "rolling_std", "rolling_max", "rolling_min"]
        )
        self.has_cross = any(
            k in self.feature_names
            for k in ["cross_mean", "cross_std", "cross_max", "cross_min"]
        )

        self.pointwise_feats = [
            f
            for f in self.feature_names
            if not f.startswith("rolling_")
            and not f.startswith("cross_")
            and f != "pair_avgs"
        ]

        self.in_dim = 0
        self.in_dim += len(self.pointwise_feats) * BASE_OUT_DIM
        if "rolling_mean" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "rolling_std" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "rolling_max" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "rolling_min" in self.feature_names:
            self.in_dim += len(self.rolling_ks) * BASE_OUT_DIM
        if "cross_mean" in self.feature_names:
            self.in_dim += 1
        if "cross_std" in self.feature_names:
            self.in_dim += 1
        if "cross_max" in self.feature_names:
            self.in_dim += 1
        if "cross_min" in self.feature_names:
            self.in_dim += 1

        self.num_pair_avgs = num_pair_avgs
        self.pair_indices = []
        if self.num_pair_avgs > 0:
            self.in_dim += self.num_pair_avgs
            np.random.seed(42)
            all_pairs = list(itertools.combinations(range(BASE_OUT_DIM), 2))
            if len(all_pairs) < self.num_pair_avgs:
                self.pair_indices = all_pairs
            else:
                self.pair_indices = random.sample(all_pairs, self.num_pair_avgs)

        max_k = max(self.rolling_ks) if self.rolling_ks else 0
        hist_len = max(max_k, 2) + 1
        self.history = deque(maxlen=hist_len)

    def update_history(self, x: np.ndarray):
        self.history.append(x.astype(np.float32))

    def make_single(self, x: np.ndarray) -> np.ndarray:
        self.update_history(x)

        hist_arr = np.array(list(self.history))
        cx = x.astype(np.float32)
        feats = []

        if "id" in self.pointwise_feats:
            feats.append(cx)
        if "tanh" in self.pointwise_feats:
            feats.append(np.tanh(cx / self.tanh_scale))
        if "sigmoid" in self.pointwise_feats:
            feats.append(1.0 / (1.0 + np.exp(-cx / self.tanh_scale)))
        if "sq" in self.pointwise_feats:
            feats.append(cx**2)
        if "sqrt" in self.pointwise_feats:
            feats.append(np.sqrt(np.abs(cx)) * np.sign(cx))
        if "gelu" in self.pointwise_feats:
            feats.append(np_gelu(cx))
        if "silu" in self.pointwise_feats:
            feats.append(np_silu(cx))
        if "sin" in self.pointwise_feats:
            feats.append(np_sin(cx))
        if "cos" in self.pointwise_feats:
            feats.append(np_cos(cx))
        if "arcsin" in self.pointwise_feats:
            feats.append(np_arcsin(cx))
        if "log" in self.pointwise_feats:
            feats.append(np_log1p_abs(cx))
        if "exp" in self.pointwise_feats:
            feats.append(np.exp(-(cx**2)))
        if "sign" in self.pointwise_feats:
            feats.append(np.sign(cx))
        if "abs" in self.pointwise_feats:
            feats.append(np.abs(cx))
        if "softplus" in self.pointwise_feats:
            feats.append(np_softplus(cx))

        if "diff1" in self.pointwise_feats:
            if len(self.history) >= 2:
                d1 = self.history[-1] - self.history[-2]
            else:
                d1 = np.zeros_like(cx)
            feats.append(d1)

        if "diff2" in self.pointwise_feats:
            if len(self.history) >= 3:
                d2 = self.history[-1] - 2 * self.history[-2] + self.history[-3]
            else:
                d2 = np.zeros_like(cx)
            feats.append(d2)

        if self.has_cross:
            if "cross_mean" in self.feature_names:
                feats.append([np.mean(cx)])
            if "cross_std" in self.feature_names:
                feats.append([np.std(cx)])
            if "cross_max" in self.feature_names:
                feats.append([np.max(cx)])
            if "cross_min" in self.feature_names:
                feats.append([np.min(cx)])

        if self.num_pair_avgs > 0:
            p_avgs = np.zeros(self.num_pair_avgs, dtype=np.float32)
            for idx, (i, j) in enumerate(self.pair_indices):
                p_avgs[idx] = (cx[i] + cx[j]) * 0.5
            feats.append(p_avgs)

        if self.has_rolling:
            for k in self.rolling_ks:
                if len(hist_arr) == 0:
                    window = np.zeros((1, BASE_OUT_DIM), dtype=np.float32)
                elif len(hist_arr) < k:
                    window = hist_arr
                else:
                    window = hist_arr[-k:]

                if "rolling_mean" in self.feature_names:
                    feats.append(np.mean(window, axis=0))
                if "rolling_std" in self.feature_names:
                    feats.append(np.std(window, axis=0))
                if "rolling_max" in self.feature_names:
                    feats.append(np.max(window, axis=0))
                if "rolling_min" in self.feature_names:
                    feats.append(np.min(window, axis=0))

        if not feats:
            return cx

        final_feats = []
        for f in feats:
            if isinstance(f, (list, tuple)):
                final_feats.append(np.array(f, dtype=np.float32))
            elif isinstance(f, np.ndarray):
                final_feats.append(f.flatten())
            else:
                final_feats.append(np.array([f], dtype=np.float32))

        out = np.concatenate(final_feats).astype(np.float32)
        return out

    def reset(self):
        self.history.clear()


class WindowDataset(Dataset):
    def __init__(self, windows, targets, fe=None):
        self.fe = fe
        if fe:
            self.windows = [fe.featurize_seq(w) for w in windows]
        else:
            self.windows = windows
        self.targets = targets

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32), torch.tensor(
            self.targets[idx], dtype=torch.float32
        )
