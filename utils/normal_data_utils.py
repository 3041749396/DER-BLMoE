# utils/normal_data_utils.py
import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import scipy.signal

from configs.config_moe_bls import MoeBLSConfig


# -------------------- 工具函数 -------------------- #

def _join(root: str, fname: str) -> str:
    return os.path.join(root, fname)


def _infer_label_name_from_x(x_name: str, is_train: bool) -> str:
    basename = os.path.basename(x_name)
    prefix = "Y"
    if "X_" in basename:
        return basename.replace("X_", prefix + "_")
    return "Y_train_10Class.npy" if is_train else "Y_test_10Class.npy"


def complex_from_concat(X_raw: np.ndarray) -> np.ndarray:
    """将拼接格式或 3D 格式的信号转为复数数组"""
    if np.iscomplexobj(X_raw):
        if X_raw.ndim != 2:
            raise ValueError(f"Expected 2D complex array, got {X_raw.shape}")
        return X_raw.astype(np.complex128)

    if X_raw.ndim == 3 and X_raw.shape[2] == 2:
        return X_raw[..., 0] + 1j * X_raw[..., 1]

    if X_raw.ndim != 2:
        raise ValueError(f"X_raw 必须是二维数组 (N, 2*L) 或 (N, L, 2)，现在是 {X_raw.shape}")

    if X_raw.shape[1] % 2 != 0:
        raise ValueError(f"Real+Imag 拼接格式特征维度必须是 2*L，目前是 {X_raw.shape[1]}")

    L = X_raw.shape[1] // 2
    real = X_raw[:, :L]
    imag = X_raw[:, L:]
    return real.astype(np.float64) + 1j * imag.astype(np.float64)


def _ensure_2d_concat(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3 and X.shape[2] == 2:
        return np.concatenate([X[..., 0], X[..., 1]], axis=1)
    return X


def _load_xy(root, x_name, y_name_hint, is_train):
    """
    通用加载函数：加载 X 和 Y。
    """
    x_path = _join(root, x_name)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Data file not found: {x_path}")

    print(f"[Data] Loading: {x_name} ...")
    X = np.load(x_path)
    X = _ensure_2d_concat(X)

    # 尝试加载 Y
    y_path = _join(root, y_name_hint)
    if not os.path.exists(y_path):
        inferred_name = _infer_label_name_from_x(x_name, is_train)
        y_path = _join(root, inferred_name)

    if os.path.exists(y_path):
        Y = np.load(y_path).astype(int).ravel()
        if len(Y) != len(X):
            if len(Y) > len(X):
                Y = Y[:len(X)]
            else:
                raise ValueError(f"Not enough labels for {x_name}: Data={len(X)}, Label={len(Y)}")
    else:
        print(f"  [Warn] Label file not found for {x_name}, returning None for Y.")
        Y = None

    return X, Y


def compute_bandwidth_importance(X_bands: list, method: str = "variance") -> list:
    raw_scores = []
    for band_data in X_bands:
        if np.iscomplexobj(band_data):
            data_flat = np.abs(band_data).ravel()
        else:
            data_flat = band_data.ravel()

        if method == "variance":
            score = np.var(data_flat)
        elif method == "energy":
            score = np.mean(data_flat ** 2)
        elif method == "max":
            score = np.max(np.abs(data_flat))
        else:
            score = 1.0
        raw_scores.append(float(score))

    raw_scores = np.array(raw_scores)
    if raw_scores.max() > raw_scores.min():
        scores_norm = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    else:
        scores_norm = np.ones_like(raw_scores)
    scores_final = scores_norm + 0.1
    return scores_final.tolist()


# -------------------- 核心处理逻辑 -------------------- #

def calculate_global_crop_indices(X_raw: np.ndarray, cfg: MoeBLSConfig, energy_ratio: float = 0.95) -> Tuple[int, int]:
    """
    【修改点 1】基于全量数据计算基于全局平均能量的裁剪索引。
    不再进行采样，而是统计所有输入样本的平均功率谱。
    """
    print(f"[Calibration] Estimating {energy_ratio*100}% energy bounds from ALL {len(X_raw)} samples...")
    
    # 预处理：转复数 & 去直流
    X_complex = complex_from_concat(X_raw)
    X_complex = X_complex - np.mean(X_complex, axis=1, keepdims=True)
    
    nperseg = getattr(cfg, "STFT_NPERSEG", 128)
    nfft = getattr(cfg, "STFT_NFFT", 256)
    noverlap = getattr(cfg, "STFT_NO_OVERLAP", nperseg // 2)
    
    # 累加所有样本的频谱能量
    total_psd_accum = None
    
    for sig in X_complex:
        f, t, Zxx = scipy.signal.stft(
            sig, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window="hamming", boundary=None, padded=False, return_onesided=False
        )
        Z_shift = np.fft.fftshift(Zxx, axes=0)
        
        # 计算该样本在频率轴上的能量分布 (Sum over Time axis)
        sample_freq_energy = np.sum(np.abs(Z_shift)**2, axis=1)
        
        if total_psd_accum is None:
            total_psd_accum = sample_freq_energy
        else:
            total_psd_accum += sample_freq_energy
            
    # 计算累积能量分布 (CDF)
    avg_psd = total_psd_accum / len(X_raw)
    total_energy = np.sum(avg_psd)
    
    if total_energy == 0:
        print("  [Warn] Signal energy is zero, using full band.")
        return 0, len(avg_psd)

    cdf = np.cumsum(avg_psd) / total_energy
    
    # 寻找边界
    drop_ratio = (1.0 - energy_ratio) / 2.0
    lower_idx = np.searchsorted(cdf, drop_ratio)
    upper_idx = np.searchsorted(cdf, 1.0 - drop_ratio)
    
    # 安全检查
    if upper_idx - lower_idx < 8: 
        print(f"  [Warn] Calculated bandwidth too narrow ({lower_idx}->{upper_idx}), using full band.")
        return 0, len(avg_psd)
        
    print(f"  [Calibration] Global Energy Crop: Freq Bins {lower_idx} -> {upper_idx} (Total: {len(avg_psd)})")
    return int(lower_idx), int(upper_idx)


def stft_features_from_raw(X_raw: np.ndarray, cfg: MoeBLSConfig, crop_indices: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    提取 STFT 特征，支持基于能量的频段裁剪。
    【修改点 2】引入 Zero-Padding (零填充)，确保频段可以被 num_bands 整除，不再丢弃数据。
    """
    X_complex = complex_from_concat(X_raw)
    X_complex = X_complex - np.mean(X_complex, axis=1, keepdims=True)
    
    nperseg = getattr(cfg, "STFT_NPERSEG", 128)
    noverlap = getattr(cfg, "STFT_NO_OVERLAP", nperseg // 2)
    nfft = getattr(cfg, "STFT_NFFT", 256)
    num_bands = getattr(cfg, "NUM_LOGICAL_EXPERTS", 1)

    feats: List[np.ndarray] = []
    for sig in X_complex:
        f, t, Zxx = scipy.signal.stft(
            sig, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            window="hamming", boundary=None, padded=False, return_onesided=False
        )
        Z_shift = np.fft.fftshift(Zxx, axes=0)
        
        # --- 应用裁剪 ---
        if crop_indices is not None:
            s, e = crop_indices
            Z_use_raw = Z_shift[s:e, :] # 只保留有效能量频段
        else:
            Z_use_raw = Z_shift
        # ----------------

        F, T = Z_use_raw.shape

        # --- 零填充逻辑：确保 F 能被 num_bands 整除 ---
        # 如果 num_bands > 0，计算余数并填充，而不是截断
        if num_bands > 1:
            remainder = F % num_bands
            if remainder != 0:
                pad_width = num_bands - remainder
                # 在频率轴 (axis=0) 的末尾填充 0
                Z_use = np.pad(Z_use_raw, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            else:
                Z_use = Z_use_raw
        else:
            # num_bands 为 0 或 1 时不需要切分或填充
            Z_use = Z_use_raw
            num_bands = 1

        # 分割并拼接特征
        band_chunks = np.split(Z_use, num_bands, axis=0)
        band_feats = []
        for chunk in band_chunks:
            chunk_stacked = np.concatenate([chunk.real, chunk.imag], axis=0)
            band_feats.append(chunk_stacked.reshape(-1).astype(np.float32))
        
        feat = np.concatenate(band_feats)
        feats.append(feat)

    X_stft = np.stack(feats, axis=0)
    return X_stft


# -------------------- 主入口 -------------------- #

def load_normal_data(cfg: MoeBLSConfig):
    """
    同时加载两套数据：
    1. Main Data (MoE/BLS): X_train_10Class -> STFT (with Energy Crop)
    2. SFEBLN Data: X_train_10Class_for_sfebln -> Raw
    """
    root = cfg.DATA_ROOT

    # 1. 加载主数据集 (Main / Standard)
    x_tr_name = getattr(cfg, "X_TRAIN_RAW_FILE", "X_train_10Class.npy")
    x_te_name = getattr(cfg, "X_TEST_RAW_FILE", "X_test_10Class.npy")
    y_tr_name = getattr(cfg, "Y_TRAIN_FILE", "Y_train_10Class.npy")
    y_te_name = getattr(cfg, "Y_TEST_FILE", "Y_test_10Class.npy")

    X_tr_main, Y_tr_main = _load_xy(root, x_tr_name, y_tr_name, is_train=True)
    X_te_main, Y_te_main = _load_xy(root, x_te_name, y_te_name, is_train=False)

    # -------------------------------------------------------------
    # Step 1.1: 计算全局能量裁剪边界 (基于全量训练集)
    # -------------------------------------------------------------
    # 这里使用全量 X_tr_main 进行计算，不再进行采样
    crop_indices = calculate_global_crop_indices(X_tr_main, cfg, energy_ratio=0.95)

    # -------------------------------------------------------------
    # Step 1.2: 使用相同的裁剪边界计算 STFT
    # -------------------------------------------------------------
    print(f"[Data] Computing STFT for Main Data (Train)...")
    X_tr_stft = stft_features_from_raw(X_tr_main, cfg, crop_indices=crop_indices)
    
    print(f"[Data] Computing STFT for Main Data (Test)...")
    X_te_stft = stft_features_from_raw(X_te_main, cfg, crop_indices=crop_indices)

    # 2. 加载 SFEBLN 专用数据集 (保持原有逻辑)
    sfebln_tr_name = getattr(cfg, "SFEBLN_X_TRAIN_FILE", "X_train_10Class_for_sfebln.npy")
    sfebln_te_name = getattr(cfg, "SFEBLN_X_TEST_FILE", "X_test_10Class_for_sfebln.npy")
    sfebln_y_tr_name = getattr(cfg, "SFEBLN_Y_TRAIN_FILE", "Y_train_10Class_for_sfebln.npy")
    sfebln_y_te_name = getattr(cfg, "SFEBLN_Y_TEST_FILE", "Y_test_10Class_for_sfebln.npy")

    try:
        X_tr_sfebln, Y_tr_sfebln = _load_xy(root, sfebln_tr_name, sfebln_y_tr_name, is_train=True)
        X_te_sfebln, Y_te_sfebln = _load_xy(root, sfebln_te_name, sfebln_y_te_name, is_train=False)

        # 标签回退逻辑
        if Y_tr_sfebln is None and len(X_tr_sfebln) == len(X_tr_main):
            Y_tr_sfebln = Y_tr_main
        if Y_te_sfebln is None and len(X_te_sfebln) == len(X_te_main):
            Y_te_sfebln = Y_te_main

        if Y_tr_sfebln is None:
            fallback_y_path = _join(root, y_tr_name)
            if os.path.exists(fallback_y_path):
                tmp_y = np.load(fallback_y_path).astype(int).ravel()
                if len(tmp_y) >= len(X_tr_sfebln):
                    Y_tr_sfebln = tmp_y[:len(X_tr_sfebln)]

        if Y_te_sfebln is None:
            fallback_y_path = _join(root, y_te_name)
            if os.path.exists(fallback_y_path):
                tmp_y = np.load(fallback_y_path).astype(int).ravel()
                if len(tmp_y) >= len(X_te_sfebln):
                    Y_te_sfebln = tmp_y[:len(X_te_sfebln)]

    except FileNotFoundError:
        print("[Warn] SFEBLN specific data not found. SFEBLN will be skipped or fail.")
        X_tr_sfebln, Y_tr_sfebln = None, None
        X_te_sfebln, Y_te_sfebln = None, None

    return (
        (X_tr_stft, X_tr_main, Y_tr_main),
        (X_te_stft, X_te_main, Y_te_main),
        (X_tr_sfebln, Y_tr_sfebln),
        (X_te_sfebln, Y_te_sfebln)
    )