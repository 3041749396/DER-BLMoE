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


def get_stft_fs(cfg: MoeBLSConfig) -> float:
    """Return the sampling rate used for STFT axis calibration.

    This value calibrates the frequency/time axes returned by scipy.signal.stft.
    Under the current default STFT scaling and unchanged nperseg/noverlap/nfft,
    changing fs does not change the complex STFT coefficient matrix Zxx itself.
    """
    return float(getattr(cfg, "STFT_FS", 1.0))


def get_stft_freq_axis_hz(
    cfg: MoeBLSConfig,
    crop_indices: Optional[Tuple[int, int]] = None,
    padded_total_bins: Optional[int] = None,
) -> np.ndarray:
    """Return fftshifted STFT frequency axis in Hz.

    The returned axis is a baseband frequency-offset axis. For the ADS-B dataset,
    using STFT_FS=50e6 maps the two-sided STFT axis approximately to
    [-25 MHz, +25 MHz). If crop_indices and padded_total_bins are provided,
    the axis is cropped and then extended to match the zero-padded feature rows.
    """
    fs = get_stft_fs(cfg)
    nfft = int(getattr(cfg, "STFT_NFFT", 256))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs)).astype(np.float64)

    if crop_indices is not None:
        s, e = crop_indices
        freq_axis = freq_axis[int(s):int(e)]

    if padded_total_bins is not None and int(padded_total_bins) > len(freq_axis):
        pad = int(padded_total_bins) - len(freq_axis)
        if len(freq_axis) >= 2:
            df = float(freq_axis[1] - freq_axis[0])
        else:
            df = fs / nfft
        extra = freq_axis[-1] + df * np.arange(1, pad + 1, dtype=np.float64)
        freq_axis = np.concatenate([freq_axis, extra], axis=0)

    if padded_total_bins is not None and int(padded_total_bins) < len(freq_axis):
        freq_axis = freq_axis[:int(padded_total_bins)]

    return freq_axis


def get_stft_time_axis_seconds(num_frames: int, cfg: MoeBLSConfig) -> np.ndarray:
    """Return STFT frame-center time axis in seconds.

    This follows scipy.signal.stft with boundary=None and padded=False, where
    frames are centered at nperseg/2, nperseg/2+hop, ... samples.
    """
    fs = get_stft_fs(cfg)
    nperseg = int(getattr(cfg, "STFT_NPERSEG", 128))
    noverlap = int(getattr(cfg, "STFT_NO_OVERLAP", nperseg // 2))
    hop = max(1, nperseg - noverlap)
    centers = nperseg / 2.0 + np.arange(int(num_frames), dtype=np.float64) * hop
    return centers / fs



# -------------------- 核心处理逻辑 -------------------- #

def calculate_global_crop_indices(X_raw: np.ndarray, cfg: MoeBLSConfig, energy_ratio: float = 0.95) -> Tuple[int, int]:
    """
    【修改点 1】基于全量数据计算基于全局平均能量的裁剪索引。
    不再进行采样，而是统计所有输入样本的平均功率谱。

    energy_ratio 表示能量保留比例，例如 0.95 表示保留 95% 累计能量。
    """
    energy_ratio = float(energy_ratio)
    if not (0.0 < energy_ratio <= 1.0):
        raise ValueError(f"energy_ratio must be in (0, 1], got {energy_ratio}")
    print(f"[Calibration] Estimating {energy_ratio*100}% energy bounds from ALL {len(X_raw)} samples...")
    
    # 预处理：转复数 & 去直流
    X_complex = complex_from_concat(X_raw)
    X_complex = X_complex - np.mean(X_complex, axis=1, keepdims=True)
    
    nperseg = getattr(cfg, "STFT_NPERSEG", 128)
    nfft = getattr(cfg, "STFT_NFFT", 256)
    noverlap = getattr(cfg, "STFT_NO_OVERLAP", nperseg // 2)
    fs = get_stft_fs(cfg)
    
    # 累加所有样本的频谱能量
    total_psd_accum = None
    
    for sig in X_complex:
        f, t, Zxx = scipy.signal.stft(
            sig, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
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
    fs = get_stft_fs(cfg)

    feats: List[np.ndarray] = []
    for sig in X_complex:
        f, t, Zxx = scipy.signal.stft(
            sig, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
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

def build_stft_crop_report(cfg: MoeBLSConfig) -> str:
    """
    Build a concise report for STFT energy-boundary cropping.

    Note:
    - crop_indices = (r_L, r_U_exclusive) follows Python slicing [r_L:r_U_exclusive].
    - The printed r_U is converted to the inclusive upper retained bin.
    - After FFTShift, the zero-frequency / baseband center bin is F/2.
    """
    crop_indices = getattr(cfg, "_STFT_CROP_INDICES", None)
    energy_ratio = getattr(
        cfg,
        "_STFT_ENERGY_RATIO",
        getattr(cfg, "STFT_ENERGY_RATIO", 0.95)
    )

    F = int(getattr(cfg, "STFT_NFFT", 256))
    center_bin = F // 2

    if crop_indices is None:
        return (
            "Retained frequency-bin range: Not available\n"
            f"Center bin: {center_bin}\n"
            "Whether center bin is retained: Not available\n"
            f"Retained energy ratio: {energy_ratio}"
        )

    r_L, r_U_exclusive = crop_indices
    r_L = int(r_L)
    r_U_exclusive = int(r_U_exclusive)

    # Convert Python slicing upper bound [r_L:r_U_exclusive]
    # to an inclusive bin index for easier reporting.
    r_U = r_U_exclusive - 1

    center_retained = (r_L <= center_bin <= r_U)

    return (
        f"Retained frequency-bin range: {r_L} to {r_U}\n"
        f"Center bin: {center_bin}\n"
        f"Whether center bin is retained: {center_retained}\n"
        f"Retained energy ratio: {energy_ratio}"
    )



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
    energy_ratio = getattr(cfg, "STFT_ENERGY_RATIO", 0.95)
    crop_indices = calculate_global_crop_indices(X_tr_main, cfg, energy_ratio=energy_ratio)
    # Store crop indices for visualization/export scripts that need physical frequency axes.
    cfg._STFT_CROP_INDICES = crop_indices
    cfg._STFT_ENERGY_RATIO = energy_ratio

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